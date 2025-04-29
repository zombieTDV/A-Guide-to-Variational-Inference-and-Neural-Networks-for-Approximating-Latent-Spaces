"""
Variational Autoencoder (VAE) với MNIST và UMAP

Mô tả:
    Chương trình này triển khai một Variational Autoencoder (VAE) trên tập dữ liệu MNIST,
    với không gian ẩn 32 chiều và sử dụng UMAP để giảm chiều xuống 2D. Nó bao gồm:
    1. Huấn luyện VAE để nén ảnh chữ số MNIST
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
import umap.umap_ as umap
import numpy as np

# 1. Tham số siêu (Hyperparameters)
batch_size = 32  # Kích thước batch
lr = 1e-3  # Tốc độ học
epochs = 100  # Số lượng epochs
latent_dim = 32  # Kích thước không gian tiềm ẩn (latent space)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Sử dụng GPU nếu có

# 2. Dữ liệu (Data)
transform = transforms.ToTensor()  # Chuyển đổi hình ảnh thành tensor
train_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)  # Tập huấn luyện
test_ds = datasets.MNIST(root='.', train=False, download=True, transform=transform)  # Tập kiểm tra
train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  # DataLoader cho tập huấn luyện
test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)  # DataLoader cho tập kiểm tra

# 3. Mô hình VAE (Variational Autoencoder)
class VAE(nn.Module):
    """
    Variational Autoencoder với không gian ẩn 32 chiều.
    Bộ mã hóa (encoder) nén ảnh 28x28 thành vector 32D.
    Bộ giải mã (decoder) khôi phục ảnh từ vector 32D.
    """
    def __init__(self):
        super().__init__()
        # Encoder: đầu ra là [mu, logvar]
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)
        )
        # Decoder: kết thúc với Sigmoid để giới hạn đầu ra trong khoảng [0,1]
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 28 * 28), nn.Sigmoid()
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Tính độ lệch chuẩn
        eps = torch.randn_like(std)  # Tạo noise ngẫu nhiên
        return mu + eps * std  # Trả về giá trị mẫu

    def forward(self, x):
        h = self.enc(x)  # Chạy encoder
        mu, logvar = h.chunk(2, dim=1)  # Tách đầu ra thành mu và logvar
        z = self.reparam(mu, logvar)  # Lấy mẫu từ phân phối
        out = self.dec(z)  # Chạy decoder
        out = out.view(-1, 1, 28, 28)  # Định hình lại đầu ra
        return out, mu, logvar  # Trả về đầu ra, mu và logvar

# Khởi tạo mô hình và optimizer
model = VAE().to(device)  # Khởi tạo mô hình
opt = optim.Adam(model.parameters(), lr=lr)  # Tối ưu hóa bằng Adam
mse = nn.MSELoss(reduction='sum')  # Hàm mất mát MSE

# 4. Vòng lặp huấn luyện với MSE + KL
train_elbo = []  # Danh sách lưu ELBO huấn luyện
val_elbo = []  # Danh sách lưu ELBO kiểm tra

for epoch in range(1, epochs + 1):
    model.train()  # Chuyển mô hình sang chế độ huấn luyện
    total_loss = 0
    for imgs, _ in train_ld:
        imgs = imgs.to(device)  # Chuyển dữ liệu sang GPU
        opt.zero_grad()  # Đặt gradient về 0

        recon, mu, logvar = model(imgs)  # Chạy mô hình
        # Mất mát tái tạo: MSE giữa đầu ra sigmoid [0,1] và đầu vào
        rec_loss = mse(recon, imgs) * 0.5  # Hệ số 0.5 là tùy chọn cho NLL Gaussian chính xác

        # Thành phần KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = rec_loss + kl  # Tổng hợp mất mát
        loss.backward()  # Tính gradient
        opt.step()  # Cập nhật trọng số

        total_loss += loss.item()  # Cộng dồn mất mát
    train_elbo.append(total_loss / len(train_ld.dataset))  # Lưu ELBO huấn luyện

    # Kiểm tra
    model.eval()  # Chuyển mô hình sang chế độ kiểm tra
    total_val = 0
    with torch.no_grad():
        for imgs, _ in test_ld:
            imgs = imgs.to(device)  # Chuyển dữ liệu sang GPU
            recon, mu, logvar = model(imgs)  # Chạy mô hình
            rec_loss = mse(recon, imgs) * 0.5
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_val += (rec_loss + kl).item()  # Cộng dồn mất mát kiểm tra
    val_elbo.append(total_val / len(test_ld.dataset))  # Lưu ELBO kiểm tra

    print(f"Epoch {epoch}/{epochs} — Train ELBO: {train_elbo[-1]:.4f}, Val ELBO: {val_elbo[-1]:.4f}")

# 5. Vẽ biểu đồ ELBO (giá trị thấp hơn là tốt hơn) qua các epochs
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_elbo, label='Train ELBO')
plt.plot(range(1, epochs + 1), val_elbo, label='Val ELBO')
plt.xlabel('Epoch')
plt.ylabel('ELBO per image')
plt.title('Train vs. Validation ELBO')
plt.legend()
plt.show()

# 6. So sánh tái tạo nhanh
model.eval()  # Chuyển mô hình sang chế độ kiểm tra
imgs, _ = next(iter(test_ld))  # Lấy một batch từ DataLoader
imgs = imgs.to(device)[:8]  # Chọn 8 hình ảnh
with torch.no_grad():
    recons, _, _ = model(imgs)  # Tái tạo hình ảnh

fig, axs = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axs[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray'); axs[0, i].axis('off')  # Hình ảnh gốc
    axs[1, i].imshow(recons[i].cpu().squeeze(), cmap='gray'); axs[1, i].axis('off')  # Hình ảnh tái tạo
plt.show()

# 7. Trích xuất mã tiềm ẩn cho toàn bộ tập kiểm tra
model.eval()
all_z = []  # Danh sách lưu mã tiềm ẩn
all_y = []  # Danh sách lưu nhãn
with torch.no_grad():
    for imgs, labels in test_ld:
        imgs = imgs.to(device)  # Chuyển dữ liệu sang GPU
        h = model.enc(imgs)  # Chạy phần encoder để lấy mu, logvar
        mu, _ = h.chunk(2, dim=1)  # Lấy mean làm embedding
        all_z.append(mu.cpu().numpy())  # Lưu mã tiềm ẩn
        all_y.append(labels.numpy())  # Lưu nhãn

all_z = np.concatenate(all_z, axis=0)  # [N_test, latent_dim]
all_y = np.concatenate(all_y, axis=0)  # [N_test]

# 8. Giảm chiều UMAP xuống 2D
reducer = umap.UMAP(n_components=2, random_state=42)  # Khởi tạo UMAP
z_2d = reducer.fit_transform(all_z)  # [N_test, 2]

# 9. Vẽ biểu đồ phân tán theo nhãn thực
plt.figure(figsize=(8, 6))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_y, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Digit label')  # Thêm colorbar với nhãn
plt.title('UMAP của không gian tiềm ẩn 32D Variational Auto-Encoder trên MNIST')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()