# Softmax回归入门：用PyTorch实现图像分类

> 本文面向深度学习初学者，通过通俗易懂的方式讲解Softmax回归的原理和PyTorch实现

## 什么是Softmax回归？

想象一下，你要教电脑识别图片中的动物。输入一张图片，电脑需要判断这是🐶狗、🐱猫还是🐔鸡。这就是典型的**多分类问题**，而Softmax回归正是解决这类问题的利器！

## 核心概念：从得分到概率

### 1. 计算类别得分

首先，我们对每个类别计算一个"得分"：

```python
狗得分 = 像素1×权重1 + 像素2×权重2 + 像素3×权重3 + 像素4×权重4 + 偏置1
猫得分 = 像素1×权重5 + 像素2×权重6 + 像素3×权重7 + 像素4×权重8 + 偏置2  
鸡得分 = 像素1×权重9 + 像素2×权重10 + 像素3×权重11 + 像素4×权重12 + 偏置3
```

**问题**：这些得分可能是任意数值，我们无法直观理解它们的含义！

### 2. Softmax的神奇转换

Softmax的作用就是**把任意数值转换为0-1之间的概率**，让我们能直观理解模型的"信心程度"。

```python
# 假设三个类别的得分是：[2.0, 1.0, 0.1]

# Softmax计算过程：
1. 计算e的得分次方：e^2.0=7.39, e^1.0=2.72, e^0.1=1.11
2. 求和：7.39 + 2.72 + 1.11 = 11.22
3. 计算概率：
   狗概率 = 7.39 / 11.22 = 0.66 (66%)
   猫概率 = 2.72 / 11.22 = 0.24 (24%) 
   鸡概率 = 1.11 / 11.22 = 0.10 (10%)
```

**关键特性**：
- 所有概率之和为100%
- 概率越大，模型对该类别越"确信"
- 不改变得分排序（得分最高的仍是概率最高的）

## 为什么需要交叉熵损失？

在训练过程中，我们需要一个标准来衡量模型预测的好坏。交叉熵损失就是这样的"评分标准"。

**举个例子**：
- 真实标签：🐱猫
- 预测情况A：狗20%，猫70%，鸡10% → ✅ 预测正确
- 预测情况B：狗10%，猫80%，鸡10% → ✅ 预测更确信

交叉熵损失能够：
- 当预测正确时，给出较低的惩罚
- 当预测确信且正确时，给出更低的惩罚
- 鼓励模型做出"自信且正确"的预测

## PyTorch实战：完整代码示例

下面让我们用PyTorch实现一个完整的Softmax回归模型。

### 1. 环境准备和数据生成

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子，保证结果可重现
torch.manual_seed(7)

# 生成模拟数据：7张2×2的灰度图片
X = torch.rand((7, 2, 2))
print(f"输入数据形状: {X.shape}")  # torch.Size([7, 2, 2])

# 生成标签：0=狗, 1=猫, 2=鸡
target = torch.randint(0, 3, (7,))
print(f"真实标签: {target}")
```

### 2. 定义神经网络模型

```python
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # 全连接层：4个输入特征（2×2像素），3个输出（3个类别）
        self.fc = nn.Linear(4, 3)
    
    def forward(self, x):
        # 将2×2图片展平为4个像素值
        x = x.view(-1, 4)
        # 计算每个类别的得分
        scores = self.fc(x)
        return scores

# 创建模型实例
model = SimpleClassifier()
print("模型结构:")
print(model)
```

### 3. 设置损失函数和优化器

```python
# 交叉熵损失（内部自动包含softmax）
criterion = nn.CrossEntropyLoss()

# 随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("开始训练模型...")
```

### 4. 模型训练循环

```python
# 训练10个epoch
for epoch in range(10):
    # 前向传播：计算预测值
    outputs = model(X)
    
    # 计算损失
    loss = criterion(outputs, target)
    
    # 反向传播
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()        # 计算当前梯度
    optimizer.step()       # 更新模型参数
    
    # 打印训练信息
    print(f'Epoch [{epoch+1}/10], 损失: {loss.item():.4f}')
```

### 5. 模型预测和评估

```python
print("\n=== 模型预测结果 ===")

# 预测时不需要计算梯度
with torch.no_grad():
    # 获取模型输出
    test_outputs = model(X)
    
    # 转换为概率分布
    probabilities = torch.softmax(test_outputs, dim=1)
    
    # 获取预测类别（概率最大的类别）
    predictions = torch.argmax(probabilities, dim=1)
    
    print(f"真实标签: {target.tolist()}")
    print(f"预测结果: {predictions.tolist()}")
    
    # 计算准确率
    accuracy = (predictions == target).float().mean()
    print(f"准确率: {accuracy.item():.2%}")
    
    # 显示预测概率
    print("\n预测概率分布:")
    for i, prob in enumerate(probabilities):
        print(f"样本{i}: 狗{prob[0]:.1%}, 猫{prob[1]:.1%}, 鸡{prob[2]:.1%}")
```

## 完整可运行代码

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 数据准备
torch.manual_seed(7)
X = torch.rand((7, 2, 2))
target = torch.randint(0, 3, (7,))

# 2. 模型定义
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(4, 3)
    
    def forward(self, x):
        x = x.view(-1, 4)
        return self.fc(x)

model = SimpleClassifier()

# 3. 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 训练循环
print("训练过程:")
for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# 5. 测试评估
print("\n测试结果:")
with torch.no_grad():
    test_outputs = model(X)
    probabilities = torch.softmax(test_outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    print(f"真实标签: {target.tolist()}")
    print(f"预测结果: {predictions.tolist()}")
    accuracy = (predictions == target).float().mean()
    print(f"准确率: {accuracy.item():.2%}")
```

## 关键知识点总结

1. **Softmax函数**：将模型输出转换为概率分布，便于理解和解释
2. **交叉熵损失**：衡量预测概率与真实分布的差异，适合分类问题
3. **训练流程**：
   - 前向传播：计算预测值
   - 计算损失：比较预测与真实值
   - 反向传播：计算梯度
   - 参数更新：优化模型权重

4. **PyTorch核心操作**：
   - `view()`：改变张量形状
   - `zero_grad()`：梯度清零
   - `backward()`：自动求导
   - `step()`：更新参数

## 进阶思考

**为什么`softmax([100, 101, 102])`等于`softmax([-2, -1, 0])`？**

这是因为Softmax只关心数值的**相对大小**，而不是绝对数值。将所有数值减去同一个数（如102），相对大小不变，Softmax结果也不变。这个特性在数值计算中很有用，可以避免数值溢出。

