import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np

# Thư mục lưu kết quả
output_dir = os.path.join('Kết_quả_huấn_luyện_Variational_Autoecoder', 'VAE_UMAP')
os.makedirs(output_dir, exist_ok=True)

# 1. Tham số siêu (Hyperparameters)
batch_size = 32
lr = 1e-3
epochs = 100
latent_dim = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Chuẩn bị dữ liệu MNIST
transform = transforms.ToTensor()
train_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root='.', train=False, download=True, transform=transform)
train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# 3. Định nghĩa mô hình VAE
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 2*latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 28*28), nn.Sigmoid()
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparam(mu, logvar)
        out = self.dec(z).view(-1, 1, 28, 28)
        return out, mu, logvar

# Khởi tạo mô hình và optimizer
model = VAE().to(device)
opt = optim.Adam(model.parameters(), lr=lr)
mse = nn.MSELoss(reduction='sum')

# 4. Huấn luyện và ghi nhận ELBO
epochs_list = range(1, epochs+1)
train_elbo, val_elbo = [], []
for epoch in epochs_list:
    model.train()
    total_train = 0
    for imgs, _ in train_ld:
        imgs = imgs.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(imgs)
        rec_loss = mse(recon, imgs) * 0.5
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + kl
        loss.backward()
        opt.step()
        total_train += loss.item()
    train_elbo.append(total_train / len(train_ld.dataset))

    model.eval()
    total_val = 0
    with torch.no_grad():
        for imgs, _ in test_ld:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            rec_loss = mse(recon, imgs) * 0.5
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_val += (rec_loss + kl).item()
    val_elbo.append(total_val / len(test_ld.dataset))
    print(f"Epoch {epoch}/{epochs} — Train ELBO: {train_elbo[-1]:.4f}, Val ELBO: {val_elbo[-1]:.4f}")

# 5. Lưu biểu đồ ELBO
plt.figure(figsize=(8,4))
plt.plot(epochs_list, train_elbo, label='Train ELBO')
plt.plot(epochs_list, val_elbo,   label='Val ELBO')
plt.xlabel('Epoch')
plt.ylabel('ELBO per image')
plt.title('Train vs. Validation ELBO')
plt.legend()
elbo_path = os.path.join(output_dir, 'elbo_loss.png')
plt.savefig(elbo_path, dpi=150)
plt.close()

# 6. Lưu ví dụ tái tạo
model.eval()
imgs, _ = next(iter(test_ld))
imgs = imgs.to(device)[:8]
with torch.no_grad():
    recons, _, _ = model(imgs)
fig, axs = plt.subplots(2, 8, figsize=(12,3))
for i in range(8):
    axs[0,i].imshow(imgs[i].cpu().squeeze(), cmap='gray'); axs[0,i].axis('off')
    axs[1,i].imshow(recons[i].cpu().squeeze(), cmap='gray'); axs[1,i].axis('off')
recon_path = os.path.join(output_dir, 'reconstruction_examples.png')
plt.savefig(recon_path, dpi=150)
plt.close()

# 7. Trích xuất mã tiềm ẩn và giảm chiều với UMAP
model.eval()
all_z, all_y = [], []
with torch.no_grad():
    for imgs, labels in test_ld:
        imgs = imgs.to(device)
        h = model.enc(imgs)
        mu, _ = h.chunk(2, dim=1)
        all_z.append(mu.cpu().numpy())
        all_y.append(labels.numpy())
all_z = np.concatenate(all_z, axis=0)
all_y = np.concatenate(all_y, axis=0)

reducer = umap.UMAP(n_components=2, random_state=42)
z_2d = reducer.fit_transform(all_z)

# 8. Vẽ và lưu UMAP latent space
plt.figure(figsize=(8,6))
sc = plt.scatter(z_2d[:,0], z_2d[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(sc, ticks=range(10), label='Digit')
plt.title('UMAP of 32D VAE Latent Space')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
umap_path = os.path.join(output_dir, 'umap_latent_space.png')
plt.savefig(umap_path, dpi=150)
plt.close()

# 9. Thông báo
print(f"Saved: {elbo_path}, {recon_path}, {umap_path}")
