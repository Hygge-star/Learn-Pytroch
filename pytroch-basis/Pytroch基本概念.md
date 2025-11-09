# PyTorch 基本概念详解

## Tensor（张量）：数据输入的维度是不确定的

### 张量的基本概念
张量是高于标量、向量、矩阵的一种多维数据结构：

- **标量（Scalar）**：零维张量 - 单个数值
- **向量（Vector）**：一维张量 - 数值数组  
- **矩阵（Matrix）**：二维张量 - 数值表格
- **张量（Tensor）**：三维及以上 - 多维数组

### Tensor 与机器学习的关系

#### 1. 数据表示
```python
import torch

# 样本数据的张量表示
# 一条语音数据 - 一维张量
audio_sample = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.25])

# 灰度图像 - 二维张量 (高度 × 宽度)
grayscale_image = torch.randn(28, 28)  # MNIST 手写数字

# 彩色图像 - 三维张量 (通道 × 高度 × 宽度)
color_image = torch.randn(3, 224, 224)  # RGB图像

# 批量数据 - 四维张量 (批量大小 × 通道 × 高度 × 宽度)
batch_images = torch.randn(32, 3, 224, 224)  # 32张彩色图片
```

#### 2. 模型表示
```python
# 模型参数也是张量
# 线性模型: Y = WX + b

# 权重 W - 二维张量
W = torch.randn(10, 5)  # 10个输出特征，5个输入特征

# 偏置 b - 一维张量  
b = torch.randn(10)     # 10个输出特征的偏置

print(f"权重形状: {W.shape}")
print(f"偏置形状: {b.shape}")
```

### Tensor 的类型

```python
# 不同数据类型的张量
float_tensor = torch.FloatTensor([1, 2, 3])      # 32位浮点数
double_tensor = torch.DoubleTensor([1, 2, 3])    # 64位浮点数
int_tensor = torch.IntTensor([1, 2, 3])          # 32位整数
long_tensor = torch.LongTensor([1, 2, 3])        # 64位整数

# 指定数据类型创建
tensor_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
tensor_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)
```

### Tensor 的创建方式

```python
# 1. 从Python列表创建
from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 2. 从NumPy数组创建
import numpy as np
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
from_numpy = torch.from_numpy(numpy_array)

# 3. 特殊张量创建
zeros = torch.zeros(2, 3)        # 全零张量
ones = torch.ones(2, 3)          # 全一张量  
eye = torch.eye(3)               # 单位矩阵
rand = torch.rand(2, 3)          # 均匀分布随机数
randn = torch.randn(2, 3)        # 标准正态分布随机数

# 4. 类似已有张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x_zeros = torch.zeros_like(x)    # 与x形状相同的全零张量
x_ones = torch.ones_like(x)      # 与x形状相同的全一张量
```

### Tensor 的属性

```python
# 创建示例张量
tensor = torch.randn(2, 3, 4)  # 2×3×4的张量

print("张量属性示例:")
print(f"形状 (shape): {tensor.shape}")
print(f"维度 (dim): {tensor.dim()}")
print(f"数据类型 (dtype): {tensor.dtype}")
print(f"设备 (device): {tensor.device}")
print(f"是否要求梯度 (requires_grad): {tensor.requires_grad}")
print(f"张量大小 (size): {tensor.size()}")
print(f"元素总数 (numel): {tensor.numel()}")
```

### Tensor 的运算

#### 1. 数学运算
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 基本运算
print(f"加法: {a + b}")          # 或 torch.add(a, b)
print(f"减法: {a - b}")          # 或 torch.sub(a, b)  
print(f"乘法: {a * b}")          # 或 torch.mul(a, b) - 逐元素乘
print(f"除法: {a / b}")          # 或 torch.div(a, b)
print(f"矩阵乘法: {a @ b}")       # 或 torch.matmul(a, b)

# 其他数学运算
print(f"平方: {a ** 2}")
print(f"指数: {torch.exp(a)}")
print(f"对数: {torch.log(a)}")
print(f"正弦: {torch.sin(a)}")
```

#### 2. 统计运算
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

print(f"求和: {x.sum()}")
print(f"按维度求和: {x.sum(dim=0)}")  # 沿第0维求和
print(f"平均值: {x.mean()}")
print(f"最大值: {x.max()}")
print(f"最小值: {x.min()}")
print(f"标准差: {x.std()}")
```

### Tensor 的操作

#### 1. 形状操作
```python
x = torch.randn(2, 3, 4)

# 改变形状
reshaped = x.reshape(6, 4)       # 改变为6×4
flattened = x.flatten()          # 展平为一维
squeezed = x.squeeze()           # 去除维度为1的维度
unsqueezed = x.unsqueeze(0)      # 在指定位置增加维度

# 转置和维度交换
transposed = x.transpose(0, 1)   # 交换第0和第1维
permuted = x.permute(2, 0, 1)    # 重新排列维度
```

#### 2. 索引和切片
```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"第一行: {x[0]}")
print(f"第二列: {x[:, 1]}")
print(f"子矩阵: {x[0:2, 1:3]}")
print(f"布尔索引: {x[x > 5]}")
print(f"花式索引: {x[[0, 2], [1, 2]]}")
```

#### 3. 连接和分割
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 连接
cat_0 = torch.cat([a, b], dim=0)  # 沿第0维连接
cat_1 = torch.cat([a, b], dim=1)  # 沿第1维连接
stack = torch.stack([a, b])       # 在新维度堆叠

# 分割
chunks = torch.chunk(a, 2, dim=0) # 沿维度分割为2块
split = torch.split(a, 1, dim=0)  # 按指定大小分割
```

### Tensor 与 NumPy 的相互转换

```python
import torch
import numpy as np

# Tensor → NumPy
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
numpy_array = tensor.numpy()
print(f"Tensor转NumPy: {type(numpy_array)}")

# NumPy → Tensor  
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.from_numpy(numpy_array)
print(f"NumPy转Tensor: {type(tensor)}")

# 注意：共享内存
tensor = torch.tensor([1, 2, 3])
numpy_array = tensor.numpy()

# 修改tensor会影响numpy数组
tensor[0] = 100
print(f"共享内存验证 - NumPy数组: {numpy_array}")  # 也会变成100

# 避免共享内存
numpy_array = tensor.detach().numpy()  # 不共享内存
```

### 实际应用示例

```python
# 完整的机器学习数据流程示例
def data_pipeline_example():
    # 1. 创建模拟数据 (批量大小×特征数)
    batch_size, num_features = 32, 10
    X = torch.randn(batch_size, num_features)  # 输入特征
    y = torch.randn(batch_size, 1)             # 目标值
    
    # 2. 定义模型参数
    W = torch.randn(num_features, 1, requires_grad=True)  # 权重
    b = torch.randn(1, requires_grad=True)                # 偏置
    
    # 3. 前向传播
    predictions = X @ W + b                    # Y = WX + b
    
    # 4. 计算损失
    loss = ((predictions - y) ** 2).mean()     # 均方误差
    
    print(f"输入数据形状: {X.shape}")
    print(f"权重形状: {W.shape}")
    print(f"预测值形状: {predictions.shape}")
    print(f"损失值: {loss.item()}")
    
    return loss

# 运行示例
loss = data_pipeline_example()
```

## Variable（变量）：模型最开始参数是未知的

> **注意**：在 PyTorch 0.4.0 之后，Variable 已与 Tensor 合并。现在直接使用 Tensor 并设置 `requires_grad=True` 即可。

```python
# 现代PyTorch中的变量（需要梯度的张量）
# 模型参数在开始时是未知的，需要通过训练学习

# 创建需要梯度的张量（相当于原来的Variable）
W = torch.randn(5, 3, requires_grad=True)  # 权重参数
b = torch.randn(3, requires_grad=True)     # 偏置参数

print(f"W需要梯度: {W.requires_grad}")
print(f"b需要梯度: {b.requires_grad}")

# 在计算过程中自动构建计算图
x = torch.randn(10, 5)  # 输入数据
y = x @ W + b           # 前向计算

# 计算损失
loss = y.sum()

# 反向传播自动计算梯度
loss.backward()

print(f"W的梯度: {W.grad}")
print(f"b的梯度: {b.grad}")
```

## nn.Module：封装了解决计算机视觉问题所需的模型

```python
import torch.nn as nn

# 自定义神经网络模块
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 - 用于提取图像特征
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 全连接层 - 用于分类
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 假设输入图像224×224
        self.fc2 = nn.Linear(128, 10)            # 10个类别
        
        # 激活函数和池化层
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积层1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 112×112
        
        # 卷积层2  
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 56×56
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 使用模型
model = SimpleCNN()
print(model)

# 前向传播示例
input_tensor = torch.randn(4, 3, 224, 224)  # 4张RGB图像
output = model(input_tensor)
print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output.shape}")  # 4×10 (4个样本，10个类别得分)
```

### 总结

- **Tensor**：PyTorch 的核心数据结构，用于表示各种维度的数据
- **Variable**：现在已与 Tensor 合并，通过 `requires_grad=True` 实现
- **nn.Module**：神经网络模型的基类，封装了网络层和前向传播逻辑

这三个概念构成了 PyTorch 深度学习的基础，理解它们对于后续的学习至关重要。
