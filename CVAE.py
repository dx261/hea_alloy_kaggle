"""
使用Conditional VAE 生成样本 （测试中）
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 128)
        self.fc2_mu = nn.Linear(128, latent_dim)
        self.fc2_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=-1)
        h = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, condition_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, z, condition):
        z = torch.cat((z, condition), dim=-1)
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, condition_dim)
        self.decoder = Decoder(latent_dim, input_dim, condition_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition):
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, condition), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# 超参数
input_dim = 2  # 特征维度
latent_dim = 20  # 潜在空间维度
condition_dim = 2  # 类别维度
num_epochs = 500
learning_rate = 1e-3

# 生成数据集
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)
# 可视化生成的数据
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', alpha=0.6)
plt.title('Generated training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

# 类别标签独热编码
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y_onehot)

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型初始化
model = CVAE(input_dim, latent_dim, condition_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 训练CVAE
for epoch in range(num_epochs):
    for x, condition in train_loader:
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, condition)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


def sample_and_visualize(model, condition, num_samples):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        samples = model.decoder(z, condition)
    return samples


# 选择类别生成样本
condition_class_0 = torch.FloatTensor(encoder.transform(np.array([[0]])))
condition_class_1 = torch.FloatTensor(encoder.transform(np.array([[1]])))

# 生成样本
samples_class_0 = sample_and_visualize(model, condition_class_0.repeat(100, 1), 100)
samples_class_1 = sample_and_visualize(model, condition_class_1.repeat(100, 1), 100)

# 可视化生成的样本
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Training Class 0', alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Training Class 1', alpha=0.6)
plt.scatter(samples_class_0.numpy()[:, 0], samples_class_0.numpy()[:, 1], label='Generated Class 0', marker='x')
plt.scatter(samples_class_1.numpy()[:, 0], samples_class_1.numpy()[:, 1], label='Generated Class 1', marker='x')
plt.title('Generated Samples and Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
