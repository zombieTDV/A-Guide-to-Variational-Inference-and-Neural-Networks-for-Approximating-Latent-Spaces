"""
Variational AutoEncoder với hoạt ảnh sampling trên MNIST

- Huấn luyện VAE với không gian ẩn 2 chiều
- Tạo hoạt ảnh sampling và trực quan hóa
- Tích hợp phân loại trên không gian ẩn
- Lưu kết quả vào 'Kết_quả_huấn_luyện_Variational_Autoecoder/2D_latent_VAE'
"""

# -*- coding: utf-8 -*-
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
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    else:
        print("Không tìm thấy GPU, sử dụng CPU")
        num_workers = 0

    output_dir = os.path.join('Kết_quả_huấn_luyện_Variational_Autoecoder', '2D_latent_VAE')
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 128 if torch.cuda.is_available() else 32
    lr         = 1e-3
    epochs     = 100
    latent_dim = 2

    transform = transforms.ToTensor()
    train_ds  = datasets.MNIST('.', train=True,  download=True, transform=transform)
    test_ds   = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_ld  = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_ld   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 3. Định nghĩa kiến trúc VAE (Variational Autoencoder)
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

    # --- Classifier Model Definition --- Định nghĩa mô hình phân loại
    class LatentClassifier(nn.Module):
        def __init__(self, latent_dim=2, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, z):
            x = self.fc1(z)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    classifier_model = LatentClassifier(latent_dim=latent_dim, num_classes=10).to(device)
    classifier_optimizer = optim.Adam(classifier_model.parameters(), lr=lr)
    classifier_criterion = nn.CrossEntropyLoss()

    # --- End Classifier Model Definition ---

    def plot_posterior(z, y, epoch, output_dir):
        plt.figure(figsize=(6,6))
        sc = plt.scatter(z[:,0], z[:,1], c=y, cmap='tab10', s=5, alpha=0.7)
        plt.colorbar(sc, ticks=range(10), label='Digit')
        plt.xlabel('z₁')
        plt.ylabel('z₂')
        plt.title(f'Không Gian Ẩn 2D của VAE - Epoch {epoch}')
        
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        
        levels = chi2.ppf([0.6827, 0.9545, 0.9973], df=2)
        radii = np.sqrt(levels)
        xs = np.linspace(-4, 4, 200)
        ys = np.linspace(-4, 4, 200)
        X, Y = np.meshgrid(xs, ys)
        for r in radii:
            plt.contour(X, Y, X**2 + Y**2, levels=[r**2], linestyles='--')
        
        frame_path = os.path.join(output_dir, f'posterior_epoch_{epoch:04d}.png')
        plt.savefig(frame_path, dpi=150)
        plt.close()
        return frame_path

    def plot_training_posterior(z, y, epoch, output_dir):
        plt.figure(figsize=(6,6))
        sc = plt.scatter(z[:,0], z[:,1], c=y, cmap='tab10', s=5, alpha=0.7)
        plt.colorbar(sc, ticks=range(10), label='Digit')
        plt.xlabel('z₁')
        plt.ylabel('z₂')
        plt.title(f'Không Gian Ẩn 2D của VAE - Epoch {epoch}')
        
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        
        frame_path = os.path.join(output_dir, f'training_posterior_{epoch:04d}.png')
        plt.savefig(frame_path, dpi=150)
        plt.close()
        return frame_path

    posterior_dir = os.path.join(output_dir, 'posterior_frames')
    os.makedirs(posterior_dir, exist_ok=True)

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

    for epoch in range(epochs):
        model.train()
        classifier_model.train()
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(train_ld):
            data = data.to(device)
            labels = labels.to(device)
            
            opt.zero_grad()
            classifier_optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            classifier_outputs = classifier_model(mu)
            
            vae_loss = mse(recon_batch, data) * 0.5 + -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            classifier_loss = classifier_criterion(classifier_outputs, labels)
            
            total_batch_loss = vae_loss + classifier_loss
            
            total_batch_loss.backward()
            opt.step()
            classifier_optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        avg_loss = total_loss / len(train_ld.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        model.eval()
        classifier_model.eval()
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

    if frame_paths:
        try:
            images = [Image.open(fp) for fp in frame_paths]
            posterior_evolution_gif_path = os.path.join(output_dir, 'posterior_evolution.gif')
            images[0].save(
                posterior_evolution_gif_path,
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0
            )
            print(f"Đã lưu hoạt ảnh tiến trình phân phối hậu nghiệm vào {posterior_evolution_gif_path}")
        except Exception as e:
            print(f"Lỗi khi tạo GIF: {e}")

    # 7. Vẽ và lưu scatter plot với contour Gaussian (phân phối hậu nghiệm cuối cùng sau huấn luyện)
    model.eval()
    classifier_model.eval()
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
    plt.title('Không Gian Ẩn 2D của VAE')
    levels = chi2.ppf([0.6827, 0.9545, 0.9973], df=2)
    radii  = np.sqrt(levels)
    xs = np.linspace(all_z[:,0].min()-1, all_z[:,0].max()+1, 200)
    ys = np.linspace(all_z[:,1].min()-1, all_z[:,1].max()+1, 200)
    X, Y = np.meshgrid(xs, ys)
    for r in radii:
        plt.contour(X, Y, X**2 + Y**2, levels=[r**2], linestyles='--')
    final_latent_space_path = os.path.join(output_dir, 'latent_space.png')
    plt.savefig(final_latent_space_path, dpi=150)
    plt.close()

    # 8. Tạo và lưu animation di chuyển trong không gian ẩn (Latent Walk)
    n_frames = 120
    theta    = np.linspace(0, 2*np.pi, n_frames)
    radius   = 3.0
    path     = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    # Tạo figure và axes cho hoạt ảnh
    fig, (ax_sc, ax_im) = plt.subplots(1,2, figsize=(8,4))
    ax_sc.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.6)
    dot, = ax_sc.plot([], [], 'ro', ms=8)
    ax_sc.set(title='Không Gian Ẩn 2D của VAE', xlabel='z₁', ylabel='z₂')
    ax_sc.set_xlim(-4, 4)
    ax_sc.set_ylim(-4, 4)

    im = ax_im.imshow(np.zeros((28,28)), cmap='gray', vmin=0, vmax=1)
    ax_im.set(title='Di Chuyển Trong Không Gian Ẩn VAE'); ax_im.axis('off')

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

    try:
        latent_walk_gif_path = os.path.join(output_dir, 'vae_latent_walk.gif')
        anim.save(latent_walk_gif_path, writer='pillow', fps=20)
        print(f"Đã lưu hoạt ảnh di chuyển trong không gian ẩn vào {latent_walk_gif_path}")
    except Exception as e:
        print(f"Lỗi khi lưu hoạt ảnh di chuyển trong không gian ẩn: {e}")

    plt.close()

    print(f"Đã lưu: {final_latent_space_path}")

    # --- New Visualization: Random Latent Space Samples Classified by Trained Model ---

    total_random_samples = 100000
    sampling_range = (-4, 4)

    print(f"Đang tạo và phân loại {total_random_samples} điểm mẫu ngẫu nhiên từ phạm vi không gian ẩn {sampling_range}...")

    z1_random_samples = torch.rand(total_random_samples) * (sampling_range[1] - sampling_range[0]) + sampling_range[0]
    z2_random_samples = torch.rand(total_random_samples) * (sampling_range[1] - sampling_range[0]) + sampling_range[0]
    random_latent_points = torch.stack([z1_random_samples, z2_random_samples], dim=1).to(device)

    predicted_classes_random = []
    classifier_model.eval()
    with torch.no_grad():
        class_scores_random = classifier_model(random_latent_points)
        _, predicted_labels_random = torch.max(class_scores_random, dim=1)
        predicted_classes_random = predicted_labels_random.cpu().numpy()

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        random_latent_points[:, 0].cpu().numpy(),
        random_latent_points[:, 1].cpu().numpy(),
        c=predicted_classes_random,
        cmap='tab10',
        s=1,
        alpha=0.6
    )
    plt.colorbar(sc, ticks=range(10), label='Predicted Digit')
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.title('Mẫu Ngẫu Nhiên Trong Không Gian Ẩn Được Phân Loại Bởi Mô Hình Phân Loại (Classifier)')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)

    # Lưu biểu đồ các điểm mẫu ngẫu nhiên trong không gian ẩn đã phân loại
    classified_random_path = os.path.join(output_dir, 'latent_space_classified_random_samples.png')
    plt.savefig(classified_random_path, dpi=150)
    plt.close()

    print(f"Đã lưu trực quan hóa các điểm mẫu ngẫu nhiên trong không gian ẩn đã phân loại vào {classified_random_path}")