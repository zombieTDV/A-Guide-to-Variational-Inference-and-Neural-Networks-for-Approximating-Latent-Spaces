"""
Variational Autoencoder (VAE) với MNIST và Hoạt ảnh Lấy mẫu Không gian Ẩn

Mô tả:
    Chương trình này triển khai một Variational Autoencoder (VAE) trên tập dữ liệu MNIST,
    với không gian ẩn 2 chiều. Nó bao gồm:
    1. Huấn luyện VAE để nén ảnh chữ số MNIST
    2. Trực quan hóa không gian ẩn 2D với đường viền Gaussian
    3. Tạo hoạt ảnh lấy mẫu từ không gian ẩn

Tác giả: Trương Đỗ Vương
Ngày: 29/4/2025
"""

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.stats import chi2

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

# 3. Định nghĩa kiến trúc VAE
class VAE(nn.Module):
    """
    Variational Autoencoder với không gian ẩn 2 chiều.
    Bộ mã hóa (encoder) nén ảnh 28x28 thành vector 2D.
    Bộ giải mã (decoder) khôi phục ảnh từ vector 2D.
    """
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
        std = torch.exp(0.5 * logvar)  # Tính độ lệch chuẩn
        eps = torch.randn_like(std)  # Tạo noise ngẫu nhiên
        return mu + eps * std  # Trả về giá trị mẫu
    def forward(self, x):
        h      = self.enc(x)  # Chạy encoder
        mu, logvar = h.chunk(2, dim=1)  # Tách đầu ra thành mu và logvar
        z      = self.reparam(mu, logvar)  # Lấy mẫu từ phân phối
        recon  = self.dec(z)  # Chạy decoder
        return recon, mu, logvar

# Khởi tạo mô hình và optimizer
model = VAE().to(device)
opt   = optim.Adam(model.parameters(), lr=lr)
mse   = nn.MSELoss(reduction='sum')

# 4. Vòng lặp huấn luyện
train_elbo, val_elbo = [], []
for epoch in range(1, epochs+1):
    # Huấn luyện
    model.train()
    total = 0
    for imgs, _ in train_ld:
        imgs = imgs.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(imgs)
        rec_loss = mse(recon, imgs)*0.5  # Mất mát tái tạo
        kl       = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
        loss     = rec_loss + kl
        loss.backward(); opt.step()
        total += loss.item()
    train_elbo.append(total/len(train_ld.dataset))

    # Đánh giá
    model.eval()
    total = 0
    with torch.no_grad():
        for imgs, _ in test_ld:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            rec_loss = mse(recon, imgs)*0.5
            kl       = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total   += (rec_loss+kl).item()
    val_elbo.append(total/len(test_ld.dataset))

    print(f"Epoch {epoch}/{epochs} — Train ELBO: {train_elbo[-1]:.4f}, Val ELBO: {val_elbo[-1]:.4f}")

# 5. Trích xuất giá trị trung bình không gian ẩn 2D và nhãn
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

# 6. Vẽ scatter plot 2D tĩnh với đường viền Gaussian
plt.figure(figsize=(6,6))
sc = plt.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(sc, ticks=range(10), label='Digit')
plt.xlabel('z₁'); plt.ylabel('z₂'); plt.title('Không gian ẩn 2D của Variational Auto-Encoder')
levels = chi2.ppf([0.997], df=2)  # Mức độ tin cậy 99.7%
radii  = np.sqrt(levels)
xs = np.linspace(all_z[:,0].min()-1, all_z[:,0].max()+1, 200)
ys = np.linspace(all_z[:,1].min()-1, all_z[:,1].max()+1, 200)
X,Y = np.meshgrid(xs,ys)
plt.contour(X, Y, X**2+Y**2, levels=radii**2, colors=['r','g','b'], linestyles=['--','-.',':'])
plt.show()

# 7. Tạo hoạt ảnh - di chuyển điểm qua không gian ẩn 2D
n_frames = 120
theta    = np.linspace(0,2*np.pi,n_frames)
radius   = 3.0
path     = np.stack([radius*np.cos(theta), radius*np.sin(theta)], axis=1)

# Thiết lập figure với scatter plot và ảnh
fig, (ax_sc, ax_im) = plt.subplots(1,2,figsize=(8,4))
ax_sc.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.6)
dot, = ax_sc.plot([], [], 'ro', ms=8)
ax_sc.set(title='Không gian ẩn 2D của Variational Auto-Encoder', xlabel='z₁', ylabel='z₂')
im = ax_im.imshow(np.zeros((28,28)), cmap='gray', vmin=0, vmax=1)
ax_im.set(title='Kết quả lấy mẫu từ không gian ẩn'); ax_im.axis('off')

# Hàm khởi tạo cho animation
def init():
    dot.set_data([],[])
    im.set_data(np.zeros((28,28)))
    return dot, im

# Hàm cập nhật cho mỗi frame
def update(i):
    z = torch.from_numpy(path[i]).unsqueeze(0).to(device).float()
    with torch.no_grad():
        dec = model.dec(z).cpu().view(28,28).numpy()
    dot.set_data([path[i, 0]], [path[i, 1]])
    im.set_data(dec)
    return dot, im

# Tạo animation
anim = animation.FuncAnimation(fig, update, frames=range(n_frames),
                               init_func=init, interval=50, blit=True)
plt.tight_layout(); plt.show()

# Lưu animation
anim.save('vae_latent_walk.mp4', fps=20)
anim.save('vae_latent_walk.gif', writer='imagemagick', fps=20)
