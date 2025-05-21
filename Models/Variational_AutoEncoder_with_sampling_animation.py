import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.stats import chi2
from PIL import Image

# Thư mục lưu kết quả
output_dir = os.path.join('Kết_quả_huấn_luyện_Variational_Autoecoder', '2D_latent_VAE')
os.makedirs(output_dir, exist_ok=True)

# 1. Tham số siêu (Hyperparameters)
batch_size = 32
lr         = 1e-3
epochs     = 100
latent_dim = 2     # Không gian ẩn 2 chiều
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Chuẩn bị dữ liệu MNIST
transform = transforms.ToTensor()
train_ds  = datasets.MNIST('.', train=True,  download=True, transform=transform)
test_ds   = datasets.MNIST('.', train=False, download=True, transform=transform)
train_ld  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ld   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# 3. Định nghĩa kiến trúc VAE\
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 28*28),       nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h          = self.enc(x)
        mu, logvar = h.chunk(2, dim=1)
        z          = self.reparam(mu, logvar)
        recon      = self.dec(z)
        return recon, mu, logvar

model = VAE().to(device)
opt   = optim.Adam(model.parameters(), lr=lr)
mse   = nn.MSELoss(reduction='sum')

def plot_posterior(z, y, epoch, output_dir):
    """Plot and save posterior distribution for a given epoch"""
    plt.figure(figsize=(6,6))
    sc = plt.scatter(z[:,0], z[:,1], c=y, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(sc, ticks=range(10), label='Digit')
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.title(f'2D Latent Space of VAE - Epoch {epoch}')
    
    # Set consistent axis limits
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    # Add Gaussian contours
    levels = chi2.ppf([0.6827, 0.9545, 0.9973], df=2)
    radii = np.sqrt(levels)
    xs = np.linspace(-4, 4, 200)
    ys = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(xs, ys)
    for r in radii:
        plt.contour(X, Y, X**2 + Y**2, levels=[r**2], linestyles='--')
    
    # Save individual frame
    frame_path = os.path.join(output_dir, f'posterior_epoch_{epoch:04d}.png')
    plt.savefig(frame_path, dpi=150)
    plt.close()
    return frame_path

def plot_training_posterior(z, y, epoch, output_dir):
    """Plot and save posterior distribution during training (without contours)"""
    plt.figure(figsize=(6,6))
    sc = plt.scatter(z[:,0], z[:,1], c=y, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(sc, ticks=range(10), label='Digit')
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.title(f'2D Latent Space of VAE - Epoch {epoch}')
    
    # Set consistent axis limits
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    # Save individual frame
    frame_path = os.path.join(output_dir, f'training_posterior_{epoch:04d}.png')
    plt.savefig(frame_path, dpi=150)
    plt.close()
    return frame_path

# Create directories for saving training progress
posterior_dir = os.path.join(output_dir, 'posterior_frames')
os.makedirs(posterior_dir, exist_ok=True)

# Save pre-training state
model.eval()
with torch.no_grad():
    all_z = []
    all_y = []
    for data, labels in test_ld:
        data = data.to(device)
        h = model.enc(data)
        mu, logvar = h.chunk(2, dim=1)
        z = model.reparam(mu, logvar)
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
    
    all_z = np.vstack(all_z)
    all_y = np.concatenate(all_y)
    frame_path = plot_training_posterior(all_z, all_y, 0, posterior_dir)
    frame_paths = [frame_path]

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_ld):
        data = data.to(device)
        opt.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Calculate loss
        loss = mse(recon_batch, data) * 0.5 + -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Backward pass
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_ld.dataset)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    # Plot and save posterior distribution every epoch
    model.eval()
    with torch.no_grad():
        all_z = []
        all_y = []
        for data, labels in test_ld:
            data = data.to(device)
            h = model.enc(data)
            mu, logvar = h.chunk(2, dim=1)
            z = model.reparam(mu, logvar)
            all_z.append(z.cpu().numpy())
            all_y.append(labels.numpy())
        
        all_z = np.vstack(all_z)
        all_y = np.concatenate(all_y)
        frame_path = plot_training_posterior(all_z, all_y, epoch + 1, posterior_dir)
        frame_paths.append(frame_path)

# Create GIF from saved frames
if frame_paths:
    try:
        images = [Image.open(fp) for fp in frame_paths]
        gif_path = os.path.join(output_dir, 'posterior_evolution.gif')
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=200,  # 200ms between frames for smoother animation
            loop=0
        )
        print(f"Saved posterior evolution animation to {gif_path}")
    except Exception as e:
        print(f"Error creating GIF: {e}")

# 7. Vẽ và lưu scatter plot với contour Gaussian
model.eval()
all_z, all_y = [], []
with torch.no_grad():
    for imgs, labels in test_ld:
        imgs = imgs.to(device)
        h    = model.enc(imgs)
        mu, _= h.chunk(2, dim=1)
        all_z.append(mu.cpu().numpy())
        all_y.append(labels.numpy())
all_z = np.vstack(all_z)
all_y = np.concatenate(all_y)

plt.figure(figsize=(6,6))
sc = plt.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(sc, ticks=range(10), label='Digit')
plt.xlabel('z₁')
plt.ylabel('z₂')
plt.title('2D Latent Space of VAE')
levels = chi2.ppf([0.6827, 0.9545, 0.9973], df=2)
radii  = np.sqrt(levels)
xs = np.linspace(all_z[:,0].min()-1, all_z[:,0].max()+1, 200)
ys = np.linspace(all_z[:,1].min()-1, all_z[:,1].max()+1, 200)
X, Y = np.meshgrid(xs, ys)
for r in radii:
    plt.contour(X, Y, X**2 + Y**2, levels=[r**2], linestyles='--')
scatter_path = os.path.join(output_dir, 'latent_space.png')
plt.savefig(scatter_path, dpi=150)
plt.close()

# 8. Tạo và lưu animation di chuyển trong không gian ẩn
n_frames = 120
theta    = np.linspace(0, 2*np.pi, n_frames)
radius   = 3.0
path     = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

fig, (ax_sc, ax_im) = plt.subplots(1,2, figsize=(8,4))
ax_sc.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.6)
dot, = ax_sc.plot([], [], 'ro', ms=8)
ax_sc.set(title='2D Latent Space of VAE', xlabel='z₁', ylabel='z₂')
ax_sc.set_xlim(-4, 4)
ax_sc.set_ylim(-4, 4)

im = ax_im.imshow(np.zeros((28,28)), cmap='gray', vmin=0, vmax=1)
ax_im.set(title='VAE Latent Walk'); ax_im.axis('off')

def init():
    dot.set_data([], [])
    im.set_data(np.zeros((28,28)))
    return dot, im

def update(i):
    z = torch.from_numpy(path[i]).unsqueeze(0).to(device).float()
    with torch.no_grad():
        dec = model.dec(z).cpu().view(28,28).numpy()
    dot.set_data([path[i,0]], [path[i,1]])
    im.set_data(dec)
    return dot, im

anim = animation.FuncAnimation(fig, update, frames=range(n_frames), init_func=init, interval=50, blit=True)

# Save animations using Pillow
try:
    # Save latent walk animation
    latent_walk_gif = os.path.join(output_dir, 'vae_latent_walk.gif')
    anim.save(latent_walk_gif, writer='pillow', fps=20)
    print(f"Saved latent walk animation to {latent_walk_gif}")
except Exception as e:
    print(f"Error saving latent walk animation: {e}")

plt.close()
print(f"Saved: {scatter_path}")

# After saving the latent walk animation, add new square grid sampling visualization
def create_square_grid_samples(n_samples=15, z1_range=(-2, 2), z2_range=(0, 2)):
    """Create a grid of samples in the specified range"""
    z1 = np.linspace(z1_range[0], z1_range[1], n_samples)
    z2 = np.linspace(z2_range[0], z2_range[1], n_samples)
    Z1, Z2 = np.meshgrid(z1, z2)
    return np.stack([Z1.flatten(), Z2.flatten()], axis=1)

# Create square grid samples with specified frequency
n_samples = 15  # 15x15 grid
grid_samples = create_square_grid_samples(n_samples, z1_range=(-2, 2), z2_range=(0, 2))

# Create figure for grid sampling visualization
plt.figure(figsize=(15, 15))  # Adjusted figure size for 15x15 grid
model.eval()

# Plot the grid points
plt.scatter(grid_samples[:, 0], grid_samples[:, 1], c='red', s=50, alpha=0.6, label='Sampling Points')

# Generate and plot reconstructed images
for i, z in enumerate(grid_samples):
    z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model.dec(z_tensor).cpu().view(28, 28).numpy()
    
    # Calculate position for the subplot
    row = i // n_samples
    col = i % n_samples
    
    # Create subplot for the reconstructed image
    plt.subplot(n_samples, n_samples, i + 1)
    plt.imshow(recon, cmap='gray')
    plt.axis('off')

# Save the grid sampling visualization
grid_path = os.path.join(output_dir, 'latent_space_grid_sampling.png')
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved grid sampling visualization to {grid_path}")