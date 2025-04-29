import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# Thư mục lưu kết quả
output_dir = os.path.join('Kết_quả_huấn_luyện_Autoencoder', '2D_latent_AE')
os.makedirs(output_dir, exist_ok=True)

# 1. Các tham số siêu hình (Hyperparameters)
batch_size = 32
lr         = 1e-3
epochs     = 100
latent_dim = 2  # Không gian ẩn 2 chiều
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Chuẩn bị dữ liệu MNIST
transform = transforms.ToTensor()
train_ds  = datasets.MNIST('.', train=True,  download=True, transform=transform)
test_ds   = datasets.MNIST('.', train=False, download=True, transform=transform)
train_ld  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ld   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# 3. Định nghĩa kiến trúc Autoencoder với nút thắt 2D
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 28*28),      nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
    def encode(self, x):
        return self.enc(x)
    def decode(self, z):
        return self.dec(z)

model = Autoencoder().to(device)
opt   = optim.Adam(model.parameters(), lr=lr)
mse   = nn.MSELoss(reduction='sum')

# 4. Vòng lặp huấn luyện
train_losses, val_losses = [], []
for epoch in range(1, epochs+1):
    # Huấn luyện
    model.train()
    running = 0
    for imgs, _ in train_ld:
        imgs = imgs.to(device)
        opt.zero_grad()
        recon = model(imgs)
        loss  = mse(recon, imgs) * 0.5
        loss.backward()
        opt.step()
        running += loss.item()
    train_losses.append(running/len(train_ld.dataset))

    # Đánh giá
    model.eval()
    running = 0
    with torch.no_grad():
        for imgs, _ in test_ld:
            imgs = imgs.to(device)
            recon = model(imgs)
            running += (mse(recon, imgs)*0.5).item()
    val_losses.append(running/len(test_ld.dataset))

    print(f"Epoch {epoch}/{epochs} — Train MSE: {train_losses[-1]:.4f}, Val MSE: {val_losses[-1]:.4f}")

# 5. Vẽ và lưu đồ thị hàm mất mát
plt.figure(figsize=(8,4))
plt.plot(range(1, epochs+1), train_losses, label='Train MSE')
plt.plot(range(1, epochs+1), val_losses,   label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('Summed MSE')
plt.legend()
plt.title('Train vs. Validation Loss')
loss_plot = os.path.join(output_dir, 'loss_curve.png')
plt.savefig(loss_plot, dpi=150)
plt.close()

# 6. Kiểm tra khả năng tái tạo và lưu ảnh
model.eval()
imgs, _ = next(iter(test_ld))
imgs    = imgs.to(device)[:8]
with torch.no_grad():
    recons = model(imgs).cpu()

fig, axes = plt.subplots(2, 8, figsize=(12,3))
for i in range(8):
    axes[0,i].imshow(imgs[i].cpu().squeeze(), cmap='gray'); axes[0,i].axis('off')
    axes[1,i].imshow(recons[i].squeeze(),   cmap='gray'); axes[1,i].axis('off')
fig.tight_layout()
recon_path = os.path.join(output_dir, 'reconstructions.png')
plt.savefig(recon_path, dpi=150)
plt.close()

# 7. Trích xuất và lưu scatter plot không gian ẩn 2D
model.eval()
all_z, all_y = [], []
with torch.no_grad():
    for imgs, labels in test_ld:
        imgs = imgs.to(device)
        z    = model.encode(imgs)
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
all_z = np.vstack(all_z)
all_y = np.concatenate(all_y)

plt.figure(figsize=(6,6))
sc = plt.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(sc, ticks=range(10), label='Digit')
plt.xlabel('z₁')
plt.ylabel('z₂')
plt.title('Không gian ẩn 2D của Auto-Encoder')
scatter_path = os.path.join(output_dir, 'latent_space.png')
plt.savefig(scatter_path, dpi=150)
plt.close()

# 8. Tạo và lưu animation lấy mẫu từ không gian ẩn
n_frames = 120
theta    = np.linspace(0, 2*np.pi, n_frames)
radius   = 20
path     = np.stack([20 + radius*np.cos(theta), 20 + radius*np.sin(theta)], axis=1)

fig, (ax_sc, ax_im) = plt.subplots(1,2, figsize=(8,4))
ax_sc.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.6)
dot, = ax_sc.plot([], [], 'ro', ms=8)
ax_sc.set(title='Không gian ẩn 2D Auto-Encoder', xlabel='z₁', ylabel='z₂')

im = ax_im.imshow(np.zeros((28,28)), cmap='gray', vmin=0, vmax=1)
ax_im.set(title='Kết quả lấy mẫu từ không gian ẩn')
ax_im.axis('off')

def init():
    dot.set_data([], [])
    im.set_data(np.zeros((28,28)))
    return dot, im

def update(i):
    z = torch.from_numpy(path[i]).unsqueeze(0).to(device).float()
    with torch.no_grad():
        dec = model.decode(z).cpu().view(28,28).numpy()
    dot.set_data([path[i,0]], [path[i,1]])
    im.set_data(dec)
    return dot, im

anim = animation.FuncAnimation(fig, update, frames=range(n_frames), init_func=init, interval=50, blit=True)

# Lưu animation
mp4_path = os.path.join(output_dir, 'ae_latent_flythrough.mp4')
gif_path = os.path.join(output_dir, 'ae_latent_flythrough.gif')
anim.save(mp4_path, fps=20, dpi=150)
anim.save(gif_path, writer='imagemagick', fps=20)
print(f"Saved: {loss_plot}, {recon_path}, {scatter_path}, {mp4_path}, {gif_path}")
