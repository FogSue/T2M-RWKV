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


torch.manual_seed(123)
for _ in range(5):
    print(torch.rand(1))
