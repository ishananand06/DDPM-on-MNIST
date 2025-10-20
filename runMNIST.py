import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

os.makedirs("results_mnist", exist_ok=True)

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 128
IMG_SIZE = 32  
TIMESTEPS = 1000 
EPOCHS = 100
LR = 0.001

print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Diffusion Schedule ---
def get_diffusion_schedule(timesteps):
    betas = torch.linspace(0.0001, 0.02, timesteps, device=DEVICE)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), alphas_cumprod[:-1]])
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }

schedule = get_diffusion_schedule(TIMESTEPS)

def q_sample(x_0, t, schedule):
    noise = torch.randn_like(x_0, device=DEVICE)
    
    sqrt_alphas_cumprod_t = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    
    noisy_image = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image, noise

# --- U-Net Model ---
class SinosoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=DEVICE) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.bn2(self.relu(self.conv2(h)))
        return self.transform(h)
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        img_channels = 1
        down_channels = (32, 64, 128)
        up_channels = (128, 64, 32)
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinosoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(img_channels, down_channels[0], 3, padding=1)
        
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], img_channels, 1)
        
    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
    
# ---Sampling Function ---
@torch.no_grad()
def p_sample(model, x, t, schedule):
    predicted_noise = model(x, t)
    
    alphas_t = (1. - schedule["betas"][t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    
    model_mean = (1 / torch.sqrt(alphas_t)) * (x - ((1 - alphas_t) / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)
    
    if t.min() > 0:
        noise = torch.randn_like(x, device=DEVICE)
        posterior_variance_t = schedule["posterior_variance"][t].view(-1, 1, 1, 1)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    else:
        return model_mean
    
@torch.no_grad()
def p_sample_loop(model, schedule, epoch, num_images=16):
    img_size = IMG_SIZE
    channels = 1
    x = torch.randn((num_images, channels, img_size, img_size), device=DEVICE)
    
    for i in tqdm(reversed(range(TIMESTEPS)), desc="Sampling", total=TIMESTEPS, leave=False):
        t = torch.full((num_images,), i, device=DEVICE, dtype=torch.long)
        x = p_sample(model, x, t, schedule)
        
    x = (x + 1) * 0.5 
    x = make_grid(x, nrow=4)
    save_image(x, f"results_mnist/sample_epoch_{epoch}.png")

# --- Training ---
def train():
    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, (images, _) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            x_0 = images.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (x_0.shape[0],), device=DEVICE).long()
            noisy_images, real_noise = q_sample(x_0, t, schedule)
            predicted_noise = model(noisy_images, t)
            loss = criterion(real_noise, predicted_noise)
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        p_sample_loop(model, schedule, epoch)
        
    print("Training finished.")

# --- Run Training ---
if __name__ == "__main__":
    train()