import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.stats import chi2

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

# 4. Vòng lặp huấn luyện
train_elbo, val_elbo = [], []
for epoch in range(1, epochs+1):
    model.train()
    total = 0
    for imgs, _ in train_ld:
        imgs = imgs.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(imgs)
        rec_loss = mse(recon, imgs) * 0.5
        kl       = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss     = rec_loss + kl
        loss.backward()
        opt.step()
        total += loss.item()
    train_elbo.append(total / len(train_ld.dataset))

    model.eval()
    total = 0
    with torch.no_grad():
        for imgs, _ in test_ld:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            rec_loss = mse(recon, imgs) * 0.5
            kl       = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total   += (rec_loss + kl).item()
    val_elbo.append(total / len(test_ld.dataset))

    print(f"Epoch {epoch}/{epochs} — Train ELBO: {train_elbo[-1]:.4f}, Val ELBO: {val_elbo[-1]:.4f}")

# 5. Lưu đồ thị ELBO
plt.figure(figsize=(8,4))
plt.plot(range(1, epochs+1), train_elbo, label='Train ELBO')
plt.plot(range(1, epochs+1), val_elbo,   label='Val ELBO')
plt.xlabel('Epoch')
plt.ylabel('Average ELBO')
plt.legend()
plt.title('Train vs. Validation ELBO')
elbo_path = os.path.join(output_dir, 'elbo_curve.png')
plt.savefig(elbo_path, dpi=150)
plt.close()

# 6. Trích xuất giá trị trung bình không gian ẩn
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

# 7. Vẽ và lưu scatter plot với contour Gaussian
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
mp4_path = os.path.join(output_dir, 'vae_latent_walk.mp4')
gif_path = os.path.join(output_dir, 'vae_latent_walk.gif')
anim.save(mp4_path, fps=20, dpi=150)
anim.save(gif_path, writer='imagemagick', fps=20)
print(f"Saved: {elbo_path}, {scatter_path}, {mp4_path}, {gif_path}")