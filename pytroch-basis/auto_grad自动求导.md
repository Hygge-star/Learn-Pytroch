# PyTorch自动求导系统（autograd）详解

> 让PyTorch自动帮你计算梯度，告别手动求导的烦恼！

## 什么是autograd？

**autograd**是PyTorch的自动求导引擎，它能够自动计算张量的梯度。想象一下有一个智能助手，你只需要告诉它你想要求导的函数，它就能自动算出导数！

## 基本使用：跟踪梯度计算

### 设置requires_grad

```python
import torch

# 创建需要跟踪梯度的张量
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

print(f"w: {w}, requires_grad: {w.requires_grad}")
print(f"x: {x}, requires_grad: {x.requires_grad}")
```

### 构建计算图

```python
# 进行数学运算（自动构建计算图）
a = torch.add(w, x)      # a = w + x = 1 + 2 = 3
b = torch.add(w, 1)      # b = w + 1 = 1 + 1 = 2  
y = torch.mul(a, b)      # y = a * b = 3 * 2 = 6

print(f"a: {a}, grad_fn: {a.grad_fn}")  # 有grad_fn，说明被跟踪
print(f"y: {y}")
```

### 反向传播求梯度

```python
# 自动计算所有梯度
y.backward(retain_graph=True)  # retain_graph=True 保留计算图

print("梯度结果:")
print(f"∂y/∂w = {w.grad}")  # 应该为 5
print(f"∂y/∂x = {x.grad}")  # 应该为 2
```

**手动验证**：
```
y = (w + x) * (w + 1)
∂y/∂w = (w + 1) + (w + x) = 2 + 3 = 5
∂y/∂x = (w + 1) = 2
```

## 两种求导方法

### 方法1：backward() - 最常用

```python
# 重新初始化（梯度会累积，需要重新开始）
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1) 
y = torch.mul(a, b)

# 自动计算梯度（累积到.grad属性中）
y.backward()

print("使用backward():")
print(f"∂y/∂w = {w.grad}")
print(f"∂y/∂x = {x.grad}")
```

### 方法2：autograd.grad() - 直接获取梯度

```python
x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)  # y = x²

# 直接计算一阶导数
grad_1 = torch.autograd.grad(y, x, create_graph=True)
print(f"一阶导数 dy/dx = {grad_1[0]}")  # 2x = 2*3 = 6

# 计算二阶导数（需要create_graph=True）
grad_2 = torch.autograd.grad(grad_1[0], x)
print(f"二阶导数 d²y/dx² = {grad_2[0]}")  # 2
```

## autograd的重要特性

### 1. 梯度不自动清零

```python
x = torch.tensor([2.], requires_grad=True)

# 第一次计算
y1 = x ** 2
y1.backward()
print(f"第一次梯度: {x.grad}")  # 4

# 第二次计算（梯度会累积！）
y2 = x ** 3  
y2.backward()
print(f"第二次梯度: {x.grad}")  # 4 + 12 = 16

# 手动清零梯度
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"清零后梯度: {x.grad}")  # 4
```

### 2. 依赖于叶子节点的节点自动跟踪

```python
w = torch.tensor([1.], requires_grad=True)  # 叶子节点

x = w + 1      # x的requires_grad自动为True
y = x + 2      # y的requires_grad自动为True

print(f"w是叶子节点: {w.is_leaf}")  # True
print(f"x是叶子节点: {x.is_leaf}")  # False  
print(f"x的requires_grad: {x.requires_grad}")  # True
```

### 3. 叶子节点不能in-place操作

```python
w = torch.tensor([1.], requires_grad=True)

# 错误做法（会报错）：
# w += 1  # in-place操作

# 正确做法：
w = w + 1  # 创建新的张量
```

## 实际应用：雅可比向量积

当输出不是标量时，我们需要指定梯度权重：

```python
x = torch.randn(3, requires_grad=True)
print(f"输入x: {x}")

# 构建计算过程
y = x * 2
while y.data.norm() < 1000:  # 直到y的范数大于1000
    y = y * 2

print(f"输出y: {y}")

# 指定梯度权重（雅可比向量积）
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)  # 计算雅可比向量积

print(f"梯度结果: {x.grad}")
```

## 控制梯度计算

### 停止梯度跟踪

```python
x = torch.tensor([2.], requires_grad=True)

print(f"x**2的requires_grad: {(x**2).requires_grad}")  # True

# 方法1：使用detach()
y1 = (x**2).detach()
print(f"detach后: {y1.requires_grad}")  # False

# 方法2：使用torch.no_grad()上下文管理器
with torch.no_grad():
    y2 = x ** 2
    print(f"no_grad内: {y2.requires_grad}")  # False

# 方法3：设置requires_grad=False
x_no_grad = torch.tensor([2.], requires_grad=False)
y3 = x_no_grad ** 2
print(f"requires_grad=False: {y3.requires_grad}")  # False
```

## 练习题详解

**题目**：求 $y = x^2$ 对 $x$ 的一阶偏导和二阶偏导

```python
x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)  # y = x²

# 一阶导数：dy/dx = 2x
grad_1 = torch.autograd.grad(y, x, create_graph=True)
print(f"一阶导数在x=3处的值: {grad_1[0]}")  # 2*3 = 6

# 二阶导数：d²y/dx² = 2  
grad_2 = torch.autograd.grad(grad_1[0], x)
print(f"二阶导数: {grad_2[0]}")  # 2
```

**数学推导**：
- 一阶导数：$\frac{dy}{dx} = 2x$，在 $x=3$ 时为 $6$
- 二阶导数：$\frac{d^2y}{dx^2} = 2$，常数

## 在神经网络中的应用

```python
import torch.nn as nn

# 简单的线性回归
model = nn.Linear(1, 1)  # 输入1维，输出1维
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 模拟数据
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# 训练步骤
for epoch in range(100):
    # 前向传播
    predictions = model(x)
    loss = criterion(predictions, y)
    
    # 反向传播（autograd在这里发挥作用！）
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 自动计算所有梯度
    optimizer.step()       # 更新参数
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## 总结

| 特性 | 说明 | 注意事项 |
|------|------|----------|
| `requires_grad=True` | 开启梯度跟踪 | 只在需要求导的张量上设置 |
| `backward()` | 自动计算梯度 | 梯度会累积，记得清零 |
| `grad`属性 | 存储梯度结果 | 访问前确保已执行backward() |
| `zero_grad()` | 清零梯度 | 每次迭代前调用 |
| `torch.no_grad()` | 禁用梯度跟踪 | 评估模型时使用 |
| `detach()` | 分离计算图 | 创建不需要梯度的张量 |

**核心思想**：只需要定义前向计算，PyTorch会自动构建计算图并计算梯度

