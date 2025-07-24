"""
AutoEncoder trên MNIST

- Huấn luyện Autoencoder với không gian ẩn 32 chiều
- Giảm chiều bằng UMAP để trực quan hóa
- Lưu kết quả vào 'Kết_quả_huấn_luyện_Autoencoder/32D_latent_AE'
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #Ignore

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
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

    output_dir = os.path.join('Kết_quả_huấn_luyện_Autoencoder', '32D_latent_AE')
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 128 if torch.cuda.is_available() else 32
    lr = 1e-3
    epochs = 100

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='.', train=False, download=True, transform=transform)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 128), nn.ReLU(),
                nn.Linear(128, 32), nn.ReLU()
            )
            self.dec = nn.Sequential(
                nn.Linear(32, 128), nn.ReLU(),
                nn.Linear(128, 28*28), nn.Sigmoid(),
                nn.Unflatten(1, (1,28,28))
            )

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z)

    model = Autoencoder().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction='sum')

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
                total_val += mse(recon, imgs).item()
        val_losses.append(total_val / len(test_ld.dataset))
        print(f"Epoch {epoch}/{epochs} — Train ½ SSE: {train_losses[-1]:.4f}, Val ½ SSE: {val_losses[-1]:.4f}")

    plt.figure(figsize=(8,4))
    plt.plot(epochs_list, train_losses, label='Train ½ SSE')
    plt.plot(epochs_list, val_losses, label='Val ½ SSE')
    plt.xlabel('Epoch')
    plt.ylabel('Summed ½ SSE per image')
    plt.title('Train vs. Validation ½ SSE Loss')
    plt.legend()
    sse_path = os.path.join(output_dir, 'sse_loss.png')
    plt.savefig(sse_path, dpi=150)
    plt.close()

    model.eval()
    imgs, _ = next(iter(test_ld))
    imgs = imgs.to(device)[:8]
    with torch.no_grad():
        recons = model(imgs).cpu()
    fig, axs = plt.subplots(2, 8, figsize=(12,3))
    for i in range(8):
        axs[0,i].imshow(imgs[i].cpu().squeeze(), cmap='gray'); axs[0,i].axis('off')
        axs[1,i].imshow(recons[i].squeeze(), cmap='gray'); axs[1,i].axis('off')
    recon_path = os.path.join(output_dir, 'reconstruction_examples.png')
    plt.savefig(recon_path, dpi=150)
    plt.close()

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

    plt.figure(figsize=(8,6))
    sc = plt.scatter(z_2d[:,0], z_2d[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(sc, ticks=range(10), label='Digit label')
    plt.title('UMAP of 32D AE Latent Space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    umap_path = os.path.join(output_dir, 'umap_latent_space.png')
    plt.savefig(umap_path, dpi=150)
    plt.close()

    print(f"Saved: {sse_path}, {recon_path}, {umap_path}")
