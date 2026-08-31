import torch
import torch.nn as nn

# 模拟当前的设置
batch_size = 32
proj_dim = 32
num_heads = 4
head_dim = proj_dim // num_heads  # 8

# 模拟输入
eeg_feat = torch.randn(batch_size, 160)  # EEG特征
audio_feat = torch.randn(batch_size, 132)  # 假设audio_dim=132

# 模拟投影层
eeg_query_proj = nn.Linear(160, proj_dim)
audio_key_proj = nn.Linear(132, proj_dim)
audio_value_proj = nn.Linear(132, proj_dim)

# 前向传播
Q = eeg_query_proj(eeg_feat)  # (batch, proj_dim)
K = audio_key_proj(audio_feat)  # (batch, proj_dim)
V = audio_value_proj(audio_feat)  # (batch, proj_dim)

print(f'Q shape after projection: {Q.shape}')
print(f'K shape after projection: {K.shape}')
print(f'V shape after projection: {V.shape}')

# 重塑
Q = Q.view(batch_size, num_heads, head_dim)
K = K.view(batch_size, num_heads, head_dim)
V = V.view(batch_size, num_heads, head_dim)

print(f'Q shape after view: {Q.shape}')
print(f'K shape after view: {K.shape}')
print(f'V shape after view: {V.shape}')

# 继续注意力计算
Q_expanded = Q.unsqueeze(2)  # (batch, num_heads, 1, head_dim)
K_expanded = K.unsqueeze(2)  # (batch, num_heads, 1, head_dim)

print(f'Q_expanded shape: {Q_expanded.shape}')
print(f'K_expanded shape: {K_expanded.shape}')

attn_scores = torch.sum(Q_expanded * K_expanded, dim=-1, keepdim=True)
print(f'attn_scores shape: {attn_scores.shape}')

attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=-1)
print(f'attn_weights shape: {attn_weights.shape}')

# 修复后的方法
print(f'V shape: {V.shape}')
print(f'attn_weights shape: {attn_weights.shape}')

try:
    attn_output = V * attn_weights  # 直接相乘
    print(f'attn_output shape: {attn_output.shape}')
    print('Success!')
except RuntimeError as e:
    print(f'Error: {e}')