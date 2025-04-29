"""
Autoencoder với MNIST và UMAP

Mô tả:
    Chương trình này triển khai một Autoencoder trên tập dữ liệu MNIST,
    với không gian ẩn 32 chiều và sử dụng UMAP để giảm chiều xuống 2D. Nó bao gồm:
    1. Huấn luyện Autoencoder để nén ảnh chữ số MNIST
    2. Trực quan hóa không gian ẩn sử dụng UMAP
    3. Kiểm tra khả năng tái tạo của mô hình

Tác giả: Trương Đỗ Vương
Ngày: 29/4/2025
"""

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap

# 1. Các tham số siêu hình (Hyperparameters)
batch_size = 32
lr         = 1e-3
epochs     = 100
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Chuẩn bị dữ liệu MNIST
transform = transforms.ToTensor()
train_ds  = datasets.MNIST(root='.', train=True,  download=True, transform=transform)
test_ds   = datasets.MNIST(root='.', train=False, download=True, transform=transform)
train_ld  = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ld   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# 3. Định nghĩa kiến trúc Autoencoder
class Autoencoder(nn.Module):
    """
    Autoencoder với không gian ẩn 32 chiều.
    Bộ mã hóa (encoder) nén ảnh 28x28 thành vector 32D.
    Bộ giải mã (decoder) khôi phục ảnh từ vector 32D.
    """
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
opt   = optim.Adam(model.parameters(), lr=lr)
mse   = nn.MSELoss(reduction='sum')  # Tổng MSE trên tất cả các pixel

# 4. Huấn luyện với hàm mất mát MSE tổng
train_losses = []
val_losses   = []

for epoch in range(1, epochs+1):
    # Huấn luyện
    model.train()
    running = 0.0
    for imgs, _ in train_ld:
        imgs = imgs.to(device)
        opt.zero_grad()
        recon = model(imgs)
        loss  = mse(recon, imgs) * 0.5
        loss.backward()
        opt.step()
        running += loss.item()
    avg_train = running / len(train_ld.dataset)
    train_losses.append(avg_train)

    # Đánh giá
    model.eval()
    running = 0.0
    with torch.no_grad():
        for imgs, _ in test_ld:
            imgs  = imgs.to(device)
            recon = model(imgs)
            running += (mse(recon, imgs) * 0.5).item()
    avg_test = running / len(test_ld.dataset)
    val_losses.append(avg_test)

    print(f"Epoch {epoch}/{epochs} — Train MSE: {avg_train:.4f}, Val MSE: {avg_test:.4f}")

# 5. Vẽ đồ thị MSE huấn luyện và kiểm tra theo epochs
plt.figure(figsize=(8,4))
plt.plot(range(1, epochs+1), train_losses, label='Train MSE')
plt.plot(range(1, epochs+1), val_losses,   label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Summed MSE per image')
plt.title('Train vs. Validation MSE Loss')
plt.legend()
plt.show()

# 6. Kiểm tra khả năng tái tạo
model.eval()
imgs, _ = next(iter(test_ld))
imgs    = imgs.to(device)[:8]
with torch.no_grad():
    recons = model(imgs).cpu()

fig, axs = plt.subplots(2, 8, figsize=(12,3))
for i in range(8):
    axs[0,i].imshow(imgs[i].cpu().squeeze(), cmap='gray'); axs[0,i].axis('off')
    axs[1,i].imshow(recons[i].squeeze(), cmap='gray');   axs[1,i].axis('off')
plt.show()

# 7. Trích xuất mã tiềm ẩn cho toàn bộ tập kiểm tra
model.eval()
all_z = []
all_y = []
with torch.no_grad():
    for imgs, labels in test_ld:
        imgs = imgs.to(device)
        z    = model.enc(imgs)             # Mã tiềm ẩn 32 chiều
        all_z.append(z.cpu().numpy())
        all_y.append(labels.numpy())

all_z = np.concatenate(all_z, axis=0)
all_y = np.concatenate(all_y, axis=0)

# 8. Giảm chiều xuống 2D sử dụng UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
z_2d    = reducer.fit_transform(all_z)

# 9. Vẽ scatter plot với màu theo nhãn thực
plt.figure(figsize=(8,6))
scatter = plt.scatter(z_2d[:,0], z_2d[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Digit label')
plt.title('UMAP của không gian tiềm ẩn 32D Auto-Encoder trên MNIST')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show() 