import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedMambaClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, n_layers=4, latent_state_dim=12, expand=2, kernel_size=32, dropout=0.2, num_heads=4):
        """
        改进版的 Mamba 模型，结合了卷积和注意力机制。
        Args:
            input_dim (int): 输入特征维度。
            num_classes (int): 输出类别数量。
            n_layers (int): Mamba 模块层数。
            latent_state_dim (int): 潜在状态维度。
            expand (int): 卷积通道扩展因子。
            kernel_size (int): 卷积核大小。
            dropout (float): Dropout 概率。
            num_heads (int): 多头注意力的头数。
        """
        super(ImprovedMambaClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 输入层：将输入特征映射到潜在空间
        self.input_layer = nn.Conv1d(input_dim, latent_state_dim, kernel_size=1)

        # 多层改进的 Mamba 模块
        self.mamba_layers = nn.ModuleList([
            ImprovedMambaBlock(latent_state_dim, expand, kernel_size, dropout, num_heads)
            for _ in range(n_layers)
        ])

        # 分类输出层
        self.output_layer = nn.Linear(latent_state_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, input_dim, seq_len]
        x = self.input_layer(x)  # [batch_size, latent_state_dim, seq_len]

        for layer in self.mamba_layers:
            x = layer(x)

        # 全局池化：将序列特征压缩为固定长度
        x = torch.mean(x, dim=2)  # [batch_size, latent_state_dim]

        # 分类输出
        x = self.output_layer(x)  # [batch_size, num_classes]
        return x


class ImprovedMambaBlock(nn.Module):
    def __init__(self, latent_state_dim, expand, kernel_size, dropout, num_heads):
        """
        改进版的 Mamba 块，结合了卷积和注意力机制。
        Args:
            latent_state_dim (int): 潜在状态维度。
            expand (int): 卷积通道扩展因子。
            kernel_size (int): 卷积核大小。
            dropout (float): Dropout 概率。
            num_heads (int): 多头注意力的头数。
        """
        super(ImprovedMambaBlock, self).__init__()
        self.conv1 = nn.Conv1d(latent_state_dim, latent_state_dim * expand, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(latent_state_dim * expand, latent_state_dim, kernel_size, padding='same')
        self.norm = nn.BatchNorm1d(latent_state_dim)

        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(latent_state_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))  # 第一层卷积
        x = self.dropout(x)
        x = self.conv2(x)  # 第二层卷积
        x = self.norm(x)  # 归一化

        # 转换为 [seq_len, batch_size, latent_state_dim] 用于注意力
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)  # 自注意力
        x = x.permute(1, 2, 0)  # 转回 [batch_size, latent_state_dim, seq_len]

        return F.relu(x + residual)  # 残差连接
