"""
Tên file: Variational_AutoEncoder_with_sampling_animation.py
Tác giả: Trương Đỗ Vương
Ngày tạo: 23/5/2024

Mô tả:
    File này triển khai mô hình Variational Autoencoder (VAE) trên tập dữ liệu MNIST với các tính năng:
    1. Huấn luyện VAE với không gian ẩn 2 chiều
    2. Tạo hoạt ảnh lấy mẫu từ không gian ẩn
    3. Trực quan hóa quá trình huấn luyện và kết quả
    4. Hỗ trợ đa xử lý và tối ưu hóa GPU
    5. Tích hợp mô hình phân loại trên không gian ẩn

Cấu trúc:
    - Cấu hình GPU và đa xử lý
    - Định nghĩa kiến trúc VAE và mô hình phân loại
    - Huấn luyện đồng thời VAE và mô hình phân loại
    - Tạo hoạt ảnh và trực quan hóa kết quả
    - Phân tích không gian ẩn với các mẫu ngẫu nhiên

Lưu ý:
    - Đảm bảo cài đặt đầy đủ các thư viện: torch, torchvision, matplotlib, numpy, scipy, PIL
    - Kiểm tra cấu hình GPU trước khi chạy
    - Có thể điều chỉnh các tham số siêu hình (hyperparameters) để tối ưu kết quả
    - Thư mục output sẽ được tạo tự động trong 'Kết_quả_huấn_luyện_Variational_Autoecoder/2D_latent_VAE'
    - Cần cài đặt ImageMagick để tạo file GIF
"""

# -*- coding: utf-8 -*-
# Import các thư viện cần thiết
import os # Thao tác với hệ điều hành (ví dụ: tạo thư mục)
import torch # Thư viện học sâu PyTorch
from torch import nn, optim # Các module mạng nơ-ron và bộ tối ưu
from torchvision import datasets, transforms # Tải dataset và biến đổi dữ liệu
from torch.utils.data import DataLoader # Tạo DataLoader để quản lý batch dữ liệu
import matplotlib.pyplot as plt # Thư viện vẽ đồ thị
from matplotlib import animation # Tạo hoạt ảnh
import numpy as np # Thư viện tính toán số học
from scipy.stats import chi2 # Sử dụng cho phân phối chi bình phương (để vẽ contour)
from PIL import Image # Thư viện xử lý ảnh (để tạo GIF)
import multiprocessing # Import multiprocessing

if __name__ == '__main__':
    # Add freeze_support for Windows
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

    # Thư mục lưu kết quả huấn luyện và trực quan hóa
    output_dir = os.path.join('Kết_quả_huấn_luyện_Variational_Autoecoder', '2D_latent_VAE')
    os.makedirs(output_dir, exist_ok=True) # Tạo thư mục nếu chưa tồn tại

    # 1. Tham số siêu (Hyperparameters) - Các giá trị cấu hình cho quá trình huấn luyện
    batch_size = 128 if torch.cuda.is_available() else 32 # Tăng batch size nếu có GPU
    lr         = 1e-3 # Tốc độ học cho bộ tối ưu
    epochs     = 100 # Số lượng epoch huấn luyện
    latent_dim = 2     # Chiều của không gian ẩn

    # 2. Chuẩn bị dữ liệu MNIST
    transform = transforms.ToTensor() # Biến đổi ảnh từ PIL Image sang Tensor và chia tỷ lệ về [0, 1]
    train_ds  = datasets.MNIST('.', train=True,  download=True, transform=transform) # Tải dataset MNIST cho huấn luyện
    test_ds   = datasets.MNIST('.', train=False, download=True, transform=transform) # Tải dataset MNIST cho kiểm tra
    train_ld  = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) # DataLoader cho tập huấn luyện, xáo trộn dữ liệu
    test_ld   = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) # DataLoader cho tập kiểm tra, không xáo trộn

    # 3. Định nghĩa kiến trúc VAE (Variational Autoencoder)
    class VAE(nn.Module):
        """Định nghĩa kiến trúc mạng VAE với bộ mã hóa (encoder) và bộ giải mã (decoder)."""
        def __init__(self):
            super().__init__()
            # Bộ mã hóa: ánh xạ ảnh đầu vào (28x28) thành các tham số của phân phối tiềm ẩn (mu, logvar) trong không gian ẩn 2D
            self.enc = nn.Sequential(
                nn.Flatten(), # Làm phẳng ảnh 28x28 thành vector 784 chiều
                nn.Linear(28*28, 128), nn.ReLU(), # Lớp tuyến tính với hàm kích hoạt ReLU
                nn.Linear(128, 2 * latent_dim) # Lớp tuyến tính đầu ra 2 * latent_dim (cho mu và logvar)
            )
            # Bộ giải mã: ánh xạ điểm trong không gian ẩn (2D) về lại ảnh (28x28)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(), # Lớp tuyến tính với hàm kích hoạt ReLU
                nn.Linear(128, 28*28),       nn.Sigmoid(), # Lớp tuyến tính đầu ra kích thước ảnh gốc với hàm sigmoid để giữ giá trị trong [0, 1]
                nn.Unflatten(1, (1,28,28)) # Chuyển vector 784 chiều về lại dạng ảnh 1 kênh 28x28
            )

        def reparam(self, mu, logvar):
            """Thực hiện thủ thuật lấy mẫu lại (reparameterization trick) để lấy mẫu z từ phân phối N(mu, exp(logvar))."""
            std = torch.exp(0.5 * logvar) # Tính độ lệch chuẩn từ logvar
            eps = torch.randn_like(std) # Lấy mẫu từ phân phối chuẩn N(0, 1) có cùng kích thước với std
            return mu + eps * std # Công thức lấy mẫu z = mu + epsilon * std

        def forward(self, x):
            """Quá trình forward pass của VAE: mã hóa, lấy mẫu lại và giải mã."""
            h          = self.enc(x) # Mã hóa ảnh đầu vào x để lấy tham số h
            mu, logvar = h.chunk(2, dim=1) # Tách h thành mu và logvar (mỗi cái latent_dim chiều)
            z          = self.reparam(mu, logvar) # Lấy mẫu z từ mu và logvar
            recon      = self.dec(z) # Giải mã z để tái tạo ảnh
            return recon, mu, logvar # Trả về ảnh tái tạo, mu và logvar

    model = VAE().to(device) # Khởi tạo mô hình VAE và chuyển sang thiết bị đã chọn
    opt   = optim.Adam(model.parameters(), lr=lr) # Khởi tạo bộ tối ưu Adam cho VAE với weight decay
    mse   = nn.MSELoss(reduction='sum') # Hàm lỗi Mean Squared Error cho phần tái tạo

    # --- Classifier Model Definition --- Định nghĩa mô hình phân loại
    class LatentClassifier(nn.Module):
        """Định nghĩa mô hình phân loại đơn giản hoạt động trên không gian ẩn 2D."""
        def __init__(self, latent_dim=2, num_classes=10):
            super().__init__()
            # Mạng feedforward đơn giản
            self.fc1 = nn.Linear(latent_dim, 64) # Lớp tuyến tính từ latent_dim (2) sang 64
            self.relu = nn.ReLU() # Hàm kích hoạt ReLU
            self.fc2 = nn.Linear(64, num_classes) # Lớp tuyến tính từ 64 sang số lớp (10)

        def forward(self, z):
            """Forward pass của mô hình phân loại: nhận vector z trong không gian ẩn và trả về điểm số cho các lớp."""
            x = self.fc1(z)
            x = self.relu(x)
            x = self.fc2(x)
            return x # Trả về điểm số cho 10 lớp

    classifier_model = LatentClassifier(latent_dim=latent_dim, num_classes=10).to(device) # Khởi tạo mô hình phân loại
    classifier_optimizer = optim.Adam(classifier_model.parameters(), lr=lr) # Bộ tối ưu Adam cho mô hình phân loại
    classifier_criterion = nn.CrossEntropyLoss() # Hàm lỗi Cross-Entropy cho phân loại

    # --- End Classifier Model Definition ---

    def plot_posterior(z, y, epoch, output_dir):
        """Vẽ và lưu biểu đồ phân phối hậu nghiệm trong không gian ẩn cho từng epoch huấn luyện (bao gồm contour Gaussian)."""
        plt.figure(figsize=(6,6))
        sc = plt.scatter(z[:,0], z[:,1], c=y, cmap='tab10', s=5, alpha=0.7) # Vẽ scatter plot các điểm latent, tô màu theo nhãn y
        plt.colorbar(sc, ticks=range(10), label='Digit') # Thêm colorbar để hiển thị ánh xạ màu sang chữ số
        plt.xlabel('z₁') # Đặt tên trục x
        plt.ylabel('z₂') # Đặt tên trục y
        plt.title(f'Không Gian Ẩn 2D của VAE - Epoch {epoch}') # Đặt tiêu đề với số epoch
        
        # Đặt giới hạn trục cố định để dễ so sánh các plot qua các epoch
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        
        # Thêm đường contour Gaussian biểu thị các vùng có mật độ xác suất cao dưới giả định Gaussian đơn vị
        levels = chi2.ppf([0.6827, 0.9545, 0.9973], df=2) # Tính các mức độ tin cậy 68%, 95%, 99% cho phân phối chi2 với 2 bậc tự do
        radii = np.sqrt(levels) # Bán kính tương ứng của các đường contour trên không gian 2D Gaussian đơn vị
        xs = np.linspace(-4, 4, 200) # Tạo lưới điểm trên trục x
        ys = np.linspace(-4, 4, 200) # Tạo lưới điểm trên trục y
        X, Y = np.meshgrid(xs, ys) # Tạo lưới 2D từ xs và ys
        for r in radii:
            plt.contour(X, Y, X**2 + Y**2, levels=[r**2], linestyles='--') # Vẽ contour cho X^2 + Y^2 = r^2
        
        # Lưu frame riêng lẻ
        frame_path = os.path.join(output_dir, f'posterior_epoch_{epoch:04d}.png') # Đường dẫn lưu ảnh frame
        plt.savefig(frame_path, dpi=150) # Lưu ảnh
        plt.close() # Đóng figure để giải phóng bộ nhớ
        return frame_path # Trả về đường dẫn ảnh đã lưu

    def plot_training_posterior(z, y, epoch, output_dir):
        """Vẽ và lưu biểu đồ phân phối hậu nghiệm trong không gian ẩn trong quá trình huấn luyện (không có contour)."""
        plt.figure(figsize=(6,6))
        sc = plt.scatter(z[:,0], z[:,1], c=y, cmap='tab10', s=5, alpha=0.7) # Vẽ scatter plot các điểm latent, tô màu theo nhãn y
        plt.colorbar(sc, ticks=range(10), label='Digit') # Thêm colorbar
        plt.xlabel('z₁') # Đặt tên trục x
        plt.ylabel('z₂') # Đặt tên trục y
        plt.title(f'Không Gian Ẩn 2D của VAE - Epoch {epoch}') # Đặt tiêu đề
        
        # Đặt giới hạn trục cố định
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        
        # Lưu frame riêng lẻ
        frame_path = os.path.join(output_dir, f'training_posterior_{epoch:04d}.png') # Đường dẫn lưu ảnh frame
        plt.savefig(frame_path, dpi=150) # Lưu ảnh
        plt.close() # Đóng figure
        return frame_path # Trả về đường dẫn ảnh

    # Tạo thư mục con để lưu các frame ảnh trong quá trình huấn luyện (cho GIF)
    posterior_dir = os.path.join(output_dir, 'posterior_frames')
    os.makedirs(posterior_dir, exist_ok=True) # Tạo thư mục nếu cần

    # Lưu trạng thái trước khi huấn luyện (epoch 0) để có frame đầu tiên cho GIF
    model.eval() # Chuyển mô hình VAE sang chế độ đánh giá
    with torch.no_grad(): # Tắt tính toán gradient
        all_z = [] # Danh sách lưu trữ các điểm latent z
        all_y = [] # Danh sách lưu trữ các nhãn y
        # Lấy tất cả dữ liệu từ test loader để trực quan hóa
        for data, labels in test_ld:
            data = data.to(device) # Chuyển dữ liệu sang thiết bị
            h = model.enc(data) # Mã hóa dữ liệu để lấy h
            mu, logvar = h.chunk(2, dim=1) # Tách mu và logvar
            z = model.reparam(mu, logvar) # Lấy mẫu z
            all_z.append(z.cpu().numpy()) # Lưu điểm z (chuyển về numpy)
            all_y.append(labels.numpy()) # Lưu nhãn y (chuyển về numpy)
        
        all_z = np.vstack(all_z) # Nối các mảng z thành một mảng duy nhất
        all_y = np.concatenate(all_y) # Nối các mảng y thành một mảng duy nhất
        # Vẽ và lưu plot cho trạng thái trước huấn luyện (epoch 0)
        frame_path = plot_training_posterior(all_z, all_y, 0, posterior_dir)
        frame_paths = [frame_path] # Khởi tạo danh sách đường dẫn frame với frame đầu tiên

    # Vòng lặp huấn luyện chính
    for epoch in range(epochs):
        model.train() # Chuyển mô hình VAE sang chế độ huấn luyện
        classifier_model.train() # Chuyển mô hình phân loại sang chế độ huấn luyện
        total_loss = 0 # Tổng lỗi cho epoch hiện tại
        # Lặp qua từng batch dữ liệu trong train loader
        for batch_idx, (data, labels) in enumerate(train_ld): # Lấy cả dữ liệu và nhãn
            data = data.to(device) # Chuyển dữ liệu sang thiết bị
            labels = labels.to(device) # Chuyển nhãn sang thiết bị
            
            opt.zero_grad() # Đặt lại gradient của VAE về 0
            classifier_optimizer.zero_grad() # Đặt lại gradient của mô hình phân loại về 0
            
            # Forward pass VAE: mã hóa dữ liệu
            recon_batch, mu, logvar = model(data)
            
            # Forward pass Classifier: dự đoán lớp từ latent mean (mu)
            classifier_outputs = classifier_model(mu)
            
            # Tính VAE loss: lỗi tái tạo + KL divergence
            vae_loss = mse(recon_batch, data) * 0.5 + -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Tính Classifier loss: lỗi cross-entropy giữa dự đoán và nhãn thật
            classifier_loss = classifier_criterion(classifier_outputs, labels)
            
            # Kết hợp các lỗi (cộng đơn giản cho quá trình đồng huấn luyện)
            total_batch_loss = vae_loss + classifier_loss
            
            # Backward pass và tối ưu hóa
            total_batch_loss.backward() # Lan truyền ngược lỗi
            opt.step() # Cập nhật tham số VAE
            classifier_optimizer.step() # Cập nhật tham số mô hình phân loại
            
            total_loss += total_batch_loss.item() # Cộng dồn lỗi của batch vào tổng lỗi epoch
        
        avg_loss = total_loss / len(train_ld.dataset) # Tính lỗi trung bình trên mỗi mẫu
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}') # In thông tin lỗi
        
        # Vẽ và lưu biểu đồ phân phối hậu nghiệm sau mỗi epoch để tạo hoạt ảnh tiến trình huấn luyện
        model.eval() # Chuyển VAE sang chế độ đánh giá
        # Chuyển mô hình phân loại sang chế độ đánh giá để sử dụng cho việc lấy mẫu z trong plot_training_posterior
        classifier_model.eval()
        with torch.no_grad(): # Tắt tính toán gradient
            all_z = [] # Danh sách lưu trữ điểm latent (mu)
            all_y = [] # Danh sách lưu trữ nhãn thật
            # Lấy dữ liệu từ test loader
            for data, labels in test_ld:
                data = data.to(device) # Chuyển dữ liệu sang thiết bị
                h = model.enc(data) # Mã hóa dữ liệu
                mu, logvar = h.chunk(2, dim=1) # Tách mu và logvar
                z = model.reparam(mu, logvar) # Lấy mẫu z (dùng cho plot scatter)
                all_z.append(z.cpu().numpy()) # Lưu điểm z
                all_y.append(labels.numpy()) # Lưu nhãn
            
            all_z = np.vstack(all_z) # Nối các mảng z
            all_y = np.concatenate(all_y) # Nối các mảng y
            # Vẽ và lưu plot phân phối hậu nghiệm trong quá trình huấn luyện
            frame_path = plot_training_posterior(all_z, all_y, epoch + 1, posterior_dir)
            frame_paths.append(frame_path) # Thêm đường dẫn ảnh frame vào danh sách

    # Tạo GIF từ các frame đã lưu trong quá trình huấn luyện
    if frame_paths:
        try:
            images = [Image.open(fp) for fp in frame_paths] # Mở tất cả các frame ảnh
            # Đường dẫn lưu GIF hoạt ảnh tiến trình phân phối hậu nghiệm
            posterior_evolution_gif_path = os.path.join(output_dir, 'posterior_evolution.gif')
            images[0].save(
                posterior_evolution_gif_path,
                save_all=True,
                append_images=images[1:],
                duration=200,  # Thời gian hiển thị mỗi frame (ms) - 200ms = 5 fps
                loop=0 # Lặp vô hạn
            )
            print(f"Đã lưu hoạt ảnh tiến trình phân phối hậu nghiệm vào {posterior_evolution_gif_path}") # In thông báo
        except Exception as e:
            print(f"Lỗi khi tạo GIF: {e}") # Bắt và in lỗi nếu xảy ra

    # 7. Vẽ và lưu scatter plot với contour Gaussian (phân phối hậu nghiệm cuối cùng sau huấn luyện)
    model.eval() # Chuyển VAE sang chế độ đánh giá
    classifier_model.eval() # Chuyển mô hình phân loại sang chế độ đánh giá
    all_z, all_y = [], [] # Danh sách lưu điểm latent và nhãn thật
    with torch.no_grad(): # Tắt tính toán gradient
        # Lấy dữ liệu từ test loader
        for imgs, labels in test_ld:
            imgs = imgs.to(device) # Chuyển dữ liệu sang thiết bị
            h    = model.enc(imgs) # Mã hóa ảnh
            mu, _= h.chunk(2, dim=1) # Lấy mu (không cần logvar cho plot này)
            all_z.append(mu.cpu().numpy()) # Lưu mu
            all_y.append(labels.numpy()) # Lưu nhãn
    all_z = np.vstack(all_z) # Nối các mảng mu
    all_y = np.concatenate(all_y) # Nối các mảng nhãn

    plt.figure(figsize=(6,6))
    sc = plt.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7) # Vẽ scatter plot mu, tô màu theo nhãn
    plt.colorbar(sc, ticks=range(10), label='Digit') # Thêm colorbar
    plt.xlabel('z₁') # Đặt tên trục x
    plt.ylabel('z₂') # Đặt tên trục y
    plt.title('Không Gian Ẩn 2D của VAE') # Đặt tiêu đề
    levels = chi2.ppf([0.6827, 0.9545, 0.9973], df=2) # Tính các mức tin cậy cho contour Gaussian
    radii  = np.sqrt(levels) # Bán kính contour
    xs = np.linspace(all_z[:,0].min()-1, all_z[:,0].max()+1, 200) # Tạo lưới x cho contour
    ys = np.linspace(all_z[:,1].min()-1, all_z[:,1].max()+1, 200) # Tạo lưới y cho contour
    X, Y = np.meshgrid(xs, ys) # Tạo lưới 2D
    for r in radii:
        plt.contour(X, Y, X**2 + Y**2, levels=[r**2], linestyles='--') # Vẽ contour
    # Đường dẫn lưu ảnh scatter plot không gian ẩn cuối cùng
    final_latent_space_path = os.path.join(output_dir, 'latent_space.png')
    plt.savefig(final_latent_space_path, dpi=150) # Lưu ảnh
    plt.close() # Đóng figure

    # 8. Tạo và lưu animation di chuyển trong không gian ẩn (Latent Walk)
    # Animation này di chuyển một điểm trong không gian ẩn theo đường tròn và hiển thị ảnh được giải mã tại điểm đó.
    n_frames = 120 # Số lượng frame trong hoạt ảnh
    theta    = np.linspace(0, 2*np.pi, n_frames) # Góc cho đường tròn (từ 0 đến 2pi)
    radius   = 3.0 # Bán kính của đường tròn
    path     = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1) # Tạo các điểm trên đường tròn trong không gian ẩn

    # Tạo figure và axes cho hoạt ảnh
    fig, (ax_sc, ax_im) = plt.subplots(1,2, figsize=(8,4)) # Figure với 2 subplot: 1 cho scatter plot, 1 cho ảnh
    ax_sc.scatter(all_z[:,0], all_z[:,1], c=all_y, cmap='tab10', s=5, alpha=0.6) # Vẽ các điểm latent từ tập test (sử dụng dữ liệu test cuối cùng)
    dot, = ax_sc.plot([], [], 'ro', ms=8) # Vẽ điểm đỏ di chuyển trên scatter plot
    ax_sc.set(title='Không Gian Ẩn 2D của VAE', xlabel='z₁', ylabel='z₂') # Đặt tiêu đề và nhãn trục cho scatter plot
    ax_sc.set_xlim(-4, 4) # Đặt giới hạn trục x
    ax_sc.set_ylim(-4, 4) # Đặt giới hạn trục y

    im = ax_im.imshow(np.zeros((28,28)), cmap='gray', vmin=0, vmax=1) # Hiển thị ảnh giải mã (ban đầu là ảnh đen)
    ax_im.set(title='Di Chuyển Trong Không Gian Ẩn VAE'); ax_im.axis('off') # Đặt tiêu đề và ẩn trục cho ảnh giải mã

    def init():
        """Hàm khởi tạo cho hoạt ảnh Latent Walk."""
        dot.set_data([], []) # Đặt lại vị trí điểm đỏ
        im.set_data(np.zeros((28,28))) # Đặt lại ảnh giải mã về ảnh đen
        return dot, im # Trả về các đối tượng cần cập nhật

    def update(i):
        """Hàm cập nhật cho mỗi frame của hoạt ảnh Latent Walk."""
        z = torch.from_numpy(path[i]).unsqueeze(0).to(device).float() # Lấy điểm latent tại frame i, chuyển sang tensor và thiết bị
        with torch.no_grad():
            dec = model.dec(z).cpu().view(28,28).numpy() # Giải mã điểm latent thành ảnh, chuyển về numpy
        dot.set_data([path[i,0]], [path[i,1]]) # Cập nhật vị trí điểm đỏ trên scatter plot
        im.set_data(dec) # Cập nhật dữ liệu ảnh giải mã
        return dot, im # Trả về các đối tượng đã cập nhật

    anim = animation.FuncAnimation(fig, update, frames=range(n_frames), init_func=init, interval=50, blit=True) # Tạo đối tượng hoạt ảnh

    # Lưu hoạt ảnh sử dụng Pillow
    try:
        # Lưu hoạt ảnh di chuyển trong không gian ẩn
        latent_walk_gif_path = os.path.join(output_dir, 'vae_latent_walk.gif')
        anim.save(latent_walk_gif_path, writer='pillow', fps=20) # Lưu hoạt ảnh dưới dạng GIF với 20 fps
        print(f"Đã lưu hoạt ảnh di chuyển trong không gian ẩn vào {latent_walk_gif_path}") # In thông báo
    except Exception as e:
        print(f"Lỗi khi lưu hoạt ảnh di chuyển trong không gian ẩn: {e}") # Bắt và in lỗi nếu xảy ra

    plt.close() # Đóng figure của hoạt ảnh Latent Walk

    # In thông báo về các ảnh scatter plot cuối cùng đã lưu (using English filename)
    print(f"Đã lưu: {final_latent_space_path}")

    # --- New Visualization: Random Latent Space Samples Classified by Trained Model --- # Trực quan hóa mới: Các điểm mẫu ngẫu nhiên trong không gian ẩn được phân loại bởi mô hình phân loại đã huấn luyện

    # Định nghĩa số lượng điểm mẫu ngẫu nhiên và phạm vi lấy mẫu
    total_random_samples = 100000  # Số lượng điểm mẫu ngẫu nhiên
    sampling_range = (-4, 4) # Phạm vi lấy mẫu trên cả trục z1 và z2

    print(f"Đang tạo và phân loại {total_random_samples} điểm mẫu ngẫu nhiên từ phạm vi không gian ẩn {sampling_range}...") # In thông báo

    # Generate random latent points within the specified range
    z1_random_samples = torch.rand(total_random_samples) * (sampling_range[1] - sampling_range[0]) + sampling_range[0]
    z2_random_samples = torch.rand(total_random_samples) * (sampling_range[1] - sampling_range[0]) + sampling_range[0]
    random_latent_points = torch.stack([z1_random_samples, z2_random_samples], dim=1).to(device) # Shape: (total_random_samples, latent_dim)

    # Predict classes using the trained classifier model
    predicted_classes_random = []
    classifier_model.eval() # Chuyển mô hình phân loại sang chế độ đánh giá
    with torch.no_grad(): # Tắt tính toán gradient
        class_scores_random = classifier_model(random_latent_points) # Lấy điểm số cho các lớp
        _, predicted_labels_random = torch.max(class_scores_random, dim=1) # Lấy lớp có điểm số cao nhất
        predicted_classes_random = predicted_labels_random.cpu().numpy() # Chuyển kết quả về numpy

    # Vẽ biểu đồ các điểm mẫu ngẫu nhiên trong không gian ẩn được tô màu theo lớp dự đoán
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        random_latent_points[:, 0].cpu().numpy(), # Tọa độ z1 (chuyển về numpy)
        random_latent_points[:, 1].cpu().numpy(), # Tọa độ z2 (chuyển về numpy)
        c=predicted_classes_random, # Màu sắc dựa trên lớp dự đoán
        cmap='tab10', # Sử dụng colormap 'tab10'
        s=1,  # Kích thước điểm nhỏ hơn cho số lượng lớn
        alpha=0.6 # Độ trong suốt
    )
    plt.colorbar(sc, ticks=range(10), label='Predicted Digit') # Thêm colorbar
    plt.xlabel('z₁') # Nhãn trục x
    plt.ylabel('z₂') # Nhãn trục y
    plt.title('Mẫu Ngẫu Nhiên Trong Không Gian Ẩn Được Phân Loại Bởi Mô Hình Phân Loại (Classifier)') # Tiêu đề
    plt.xlim(-4, 4) # Giới hạn trục x
    plt.ylim(-4, 4) # Giới hạn trục y

    # Lưu biểu đồ các điểm mẫu ngẫu nhiên trong không gian ẩn đã phân loại
    classified_random_path = os.path.join(output_dir, 'latent_space_classified_random_samples.png')
    plt.savefig(classified_random_path, dpi=150) # Lưu ảnh
    plt.close() # Đóng figure

    print(f"Đã lưu trực quan hóa các điểm mẫu ngẫu nhiên trong không gian ẩn đã phân loại vào {classified_random_path}") # In thông báo