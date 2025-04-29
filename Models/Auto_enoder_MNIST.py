import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap

# Thư mục lưu kết quả
output_dir = os.path.join('Kết_quả_huấn_luyện_Autoencoder', 'AE_UMAP')
os.makedirs(output_dir, exist_ok=True)

# 1. Tham số siêu (Hyperparameters)
batch_size = 32
lr = 1e-3
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Chuẩn bị dữ liệu MNIST
transform = transforms.ToTensor()
train_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root='.', train=False, download=True, transform=transform)
train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# 3. Định nghĩa kiến trúc Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 32),     nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 128),    nn.ReLU(),
            nn.Linear(128, 28*28),  nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

# Khởi tạo mô hình và optimizer
model = Autoencoder().to(device)
opt = optim.Adam(model.parameters(), lr=lr)
mse = nn.MSELoss(reduction='sum')

# 4. Huấn luyện và ghi nhận MSE
epochs_list = range(1, epochs+1)
train_losses, val_losses = [], []
for epoch in epochs_list:
    model.train()
    total_train = 0
    for imgs, _ in train_ld:
        imgs = imgs.to(device)
        opt.zero_grad()
        recon = model(imgs)
        loss = mse(recon, imgs) * 0.5
        loss.backward()
        opt.step()
        total_train += loss.item()
    train_losses.append(total_train / len(train_ld.dataset))

    model.eval()
    total_val = 0
    with torch.no_grad():
        for imgs, _ in test_ld:
            imgs = imgs.to(device)
            recon = model(imgs)
            total_val += (mse(recon, imgs) * 0.5).item()
    val_losses.append(total_val / len(test_ld.dataset))
    print(f"Epoch {epoch}/{epochs} — Train MSE: {train_losses[-1]:.4f}, Val MSE: {val_losses[-1]:.4f}")

# 5. Lưu đồ thị MSE
plt.figure(figsize=(8,4))
plt.plot(epochs_list, train_losses, label='Train MSE')
plt.plot(epochs_list, val_losses,   label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('Summed MSE per image')
plt.title('Train vs. Validation MSE Loss')
plt.legend()
mse_path = os.path.join(output_dir, 'mse_loss.png')
plt.savefig(mse_path, dpi=150)
plt.close()

# 6. Lưu ví dụ tái tạo
model.eval()
imgs, _ = next(iter(test_ld))
imgs = imgs.to(device)[:8]
with torch.no_grad():
    recons = model(imgs).cpu()
fig, axs = plt.subplots(2, 8, figsize=(12,3))
for i in range(8):
    axs[0,i].imshow(imgs[i].cpu().squeeze(), cmap='gray'); axs[0,i].axis('off')
    axs[1,i].imshow(recons[i].squeeze(), cmap='gray');   axs[1,i].axis('off')
recon_path = os.path.join(output_dir, 'reconstruction_examples.png')
plt.savefig(recon_path, dpi=150)
plt.close()

# 7. Trích xuất mã tiềm ẩn và giảm chiều với UMAP
model.eval()
all_z, all_y = [], []
with torch.no_grad():
    for imgs, labels in test_ld:
        imgs = imgs.to(device)
        z = model.enc(imgs)
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())
all_z = np.concatenate(all_z, axis=0)
all_y = np.concatenate(all_y, axis=0)

reducer = umap.UMAP(n_components=2, random_state=42)
z_2d = reducer.fit_transform(all_z)

# 8. Lưu UMAP latent space
plt.figure(figsize=(8,6))
sc = plt.scatter(z_2d[:,0], z_2d[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(sc, ticks=range(10), label='Digit label')
plt.title('UMAP of 32D AE Latent Space')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
umap_path = os.path.join(output_dir, 'umap_latent_space.png')
plt.savefig(umap_path, dpi=150)
plt.close()

# 9. Thông báo
print(f"Saved: {mse_path}, {recon_path}, {umap_path}")
