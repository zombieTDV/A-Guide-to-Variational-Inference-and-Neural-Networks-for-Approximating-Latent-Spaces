"""
Tên file: Variational_AutoEncoder.py
Tác giả: Trương Đỗ Vương
Ngày tạo: 23/5/2024

Mô tả:
    File này triển khai mô hình Variational Autoencoder (VAE) trên tập dữ liệu MNIST với các tính năng:
    1. Huấn luyện VAE với không gian ẩn 32 chiều
    2. Sử dụng UMAP để giảm chiều không gian ẩn xuống 2D
    3. Trực quan hóa kết quả huấn luyện
    4. Hỗ trợ đa xử lý và tối ưu hóa GPU

Cấu trúc:
    - Cấu hình GPU và đa xử lý
    - Định nghĩa kiến trúc VAE
    - Huấn luyện mô hình
    - Giảm chiều và trực quan hóa kết quả

Lưu ý:
    - Đảm bảo cài đặt đầy đủ các thư viện: torch, torchvision, matplotlib, numpy, umap-learn
    - Kiểm tra cấu hình GPU trước khi chạy
    - Có thể điều chỉnh các tham số siêu hình (hyperparameters) để tối ưu kết quả
    - Thư mục output sẽ được tạo tự động trong 'Kết_quả_huấn_luyện_Variational_Autoecoder/VAE_UMAP'
"""

# Import các thư viện cần thiết
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np
import multiprocessing

if __name__ == '__main__':
    # Thêm freeze_support cho Windows để hỗ trợ đa xử lý
    multiprocessing.freeze_support()
    
    # Kiểm tra và cấu hình GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
        # Tối ưu hóa cho GPU
        torch.backends.cudnn.benchmark = True
        # Đặt số worker cho DataLoader bằng một nửa số CPU có sẵn
        num_workers = max(1, multiprocessing.cpu_count() // 2)  # Đảm bảo ít nhất 1 worker
    else:
        print("Không tìm thấy GPU, sử dụng CPU")
        num_workers = 0  # Không sử dụng worker khi chạy trên CPU

    # Tạo thư mục lưu kết quả
    output_dir = os.path.join('Kết_quả_huấn_luyện_Variational_Autoecoder', 'VAE_UMAP')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Cấu hình các tham số siêu hình (Hyperparameters)
    batch_size = 128 if torch.cuda.is_available() else 32  # Tăng batch size nếu có GPU
    lr = 1e-3  # Tốc độ học
    epochs = 100  # Số epoch huấn luyện
    latent_dim = 32  # Chiều của không gian ẩn

    # 2. Chuẩn bị dữ liệu MNIST
    transform = transforms.ToTensor()  # Chuyển đổi ảnh thành tensor và chuẩn hóa về [0,1]
    train_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='.', train=False, download=True, transform=transform)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 3. Định nghĩa kiến trúc VAE
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Bộ mã hóa: ánh xạ ảnh đầu vào thành các tham số của phân phối tiềm ẩn
            self.enc = nn.Sequential(
                nn.Flatten(),  # Làm phẳng ảnh 28x28 thành vector 784 chiều
                nn.Linear(28*28, 128), nn.ReLU(),  # Lớp ẩn với hàm kích hoạt ReLU
                nn.Linear(128, 2*latent_dim)  # Đầu ra cho mu và logvar
            )
            # Bộ giải mã: ánh xạ điểm trong không gian ẩn về lại ảnh
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 28*28), nn.Sigmoid()  # Sigmoid để giữ giá trị trong [0,1]
            )

        def reparam(self, mu, logvar):
            """Thực hiện thủ thuật lấy mẫu lại (reparameterization trick)"""
            std = torch.exp(0.5 * logvar)  # Tính độ lệch chuẩn từ logvar
            eps = torch.randn_like(std)  # Lấy mẫu nhiễu từ phân phối chuẩn
            return mu + eps * std  # Trả về điểm trong không gian ẩn

        def forward(self, x):
            """Quá trình forward pass của VAE"""
            h = self.enc(x)  # Mã hóa ảnh đầu vào
            mu, logvar = h.chunk(2, dim=1)  # Tách thành mu và logvar
            z = self.reparam(mu, logvar)  # Lấy mẫu điểm trong không gian ẩn
            out = self.dec(z).view(-1, 1, 28, 28)  # Giải mã và định dạng lại thành ảnh
            return out, mu, logvar

    # Khởi tạo mô hình và optimizer
    model = VAE().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction='sum')  # Hàm mất mát cho phần tái tạo

    # 4. Huấn luyện và ghi nhận ELBO
    epochs_list = range(1, epochs+1)
    train_elbo, val_elbo = [], []  # Lưu lịch sử ELBO
    for epoch in epochs_list:
        # Huấn luyện
        model.train()
        total_train = 0
        for imgs, _ in train_ld:
            imgs = imgs.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(imgs)
            # Tính loss: reconstruction loss + KL divergence
            rec_loss = mse(recon, imgs) * 0.5
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_loss + kl
            loss.backward()
            opt.step()
            total_train += loss.item()
        train_elbo.append(total_train / len(train_ld.dataset))

        # Đánh giá
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

    # 5. Vẽ và lưu biểu đồ ELBO
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
            mu, _ = h.chunk(2, dim=1)  # Chỉ lấy mu cho việc trực quan hóa
            all_z.append(mu.cpu().numpy())
            all_y.append(labels.numpy())
    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # Giảm chiều xuống 2D bằng UMAP
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

    # 9. Thông báo kết quả
    print(f"Saved: {elbo_path}, {recon_path}, {umap_path}")
