# from torch.utils.data import DataLoader, Dataset

# # 定义一个简单的数据集
# class SimpleDataset(Dataset):
#     def __init__(self):
#         self.data = [1, 2, 3, 4, 5, 6, 7, 8]
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         return self.data[idx]

# # 创建 DataLoader
# dataset = SimpleDataset()
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # 定义 cycle 函数
# def cycle(iterable):
#     while True:
#         for x in iterable:
#             yield x

# # 无限循环 DataLoader
# cyclic_loader = cycle(dataloader)
# for _ in range(10):  # 打印前 10 次迭代的内容
#     print(next(cyclic_loader))

import torch.nn as nn
import torch
import torch.nn.functional as F

# pred = torch.randn(1, 5)  
# label = torch.rand(1, 5)  
# label /= label.sum(dim=1, keepdim=True)  
# loss_fn = nn.CrossEntropyLoss()
# loss = loss_fn(pred, label)  
# print(F.log_softmax(pred, dim=1))
# print(label)
# print(torch.sum(F.log_softmax(pred, dim=1)*label, dim=1))
# print(loss)


# 示例数据
B, I_len, K_len, dim = 32, 64, 256, 8
I = torch.randint(0, 10, (B, I_len, dim))  # (B, 64, 8)
K = torch.randint(0, 10, (B, K_len, dim))  # (B, 256, 8)

# Step 1: 扩展维度并比较，找到匹配索引
I_expanded = I.unsqueeze(2)  # (B, 64, 1, 8)
K_expanded = K.unsqueeze(1)  # (B, 1, 256, 8)
matches = torch.all(I_expanded == K_expanded, dim=-1)  # (B, 64, 256)

# 找到匹配索引
indices = torch.argmax(matches.to(dtype=torch.int), dim=-1)  # (B, 64)
is_match = matches.any(dim=-1)  # (B, 64)

# 将失配位置的索引设置为特殊值（如 -1）
indices[~is_match] = -1

# Step 2: 基于索引作加权融合
# 获取匹配的 K tensor
K_matched = torch.gather(K, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, dim))  # (B, 64, 8)

# 对失配的位置进行处理（设置为 0 或保持 I 原始值）
K_matched[indices == -1] = 0  # 设置失配位置为 0

# 定义加权系数
alpha = 0.7  # 权重系数
beta = 1 - alpha

# 融合：I 和匹配的 K tensor
output = alpha * I + beta * K_matched  # (B, 64, 8)

# Step 3: 打印结果
print("I 的形状：", I.shape)
print("K 的形状：", K.shape)
print("匹配索引：", indices)
print("加权融合后的输出：", output.shape)
