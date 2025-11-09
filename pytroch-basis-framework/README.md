# PyTorch 基础框架

一个模块化设计的 PyTorch 入门框架，包含深度学习项目所需的完整组件和最佳实践。

## 📁 项目结构

```
pytorch-basic-framework/
├── src/                    # 源代码目录
│   ├── data/              # 数据加载模块
│   │   ├── __init__.py
│   │   ├── dataset.py     # 自定义数据集
│   │   └── dataloader.py  # 数据加载器
│   ├── models/            # 模型定义
│   │   ├── __init__.py
│   │   ├── basic_cnn.py   # 基础CNN模型
│   │   └── simple_mlp.py  # 简单MLP模型
│   ├── utils/             # 工具函数
│   │   ├── __init__.py
│   │   ├── config.py      # 配置管理
│   │   └── logger.py      # 日志记录
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── configs/               # 配置文件
│   └── default.yaml       # 默认配置
├── experiments/           # 实验记录
│   └── example_training/  # 示例训练结果
├── requirements.txt       # 依赖列表
├── README.md             # 项目说明
└── .gitignore           # Git忽略文件
```

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/your-username/pytorch-basic-framework.git
cd pytorch-basic-framework

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖

```txt
torch>=1.9.0
torchvision>=0.10.0
tqdm>=4.60.0
PyYAML>=5.4.0
tensorboard>=2.5.0
```

## 🛠 核心组件

### 1. 数据加载模块

```python
# src/data/dataset.py
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
```

### 2. 模型定义

```python
# src/models/basic_cnn.py
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 16 * 16, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### 3. 训练循环

```python
# src/train.py
import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='训练中')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            '损失': f'{running_loss/(batch_idx+1):.3f}',
            '准确率': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss/len(dataloader), 100.*correct/total
```

### 4. 配置文件

```yaml
# configs/default.yaml
data:
  dataset: "CIFAR10"
  batch_size: 64
  num_workers: 4

model:
  name: "BasicCNN"
  num_classes: 10

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "Adam"

logging:
  log_dir: "./logs"
  save_interval: 5
```

## 🎯 使用方法

### 开始训练

```python
# 示例训练脚本
from src.data.dataloader import create_dataloader
from src.models.basic_cnn import BasicCNN
from src.train import train_epoch
from src.utils.config import load_config

def main():
    # 加载配置
    config = load_config('configs/default.yaml')
    
    # 准备数据
    train_loader, val_loader = create_dataloader(config)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicCNN(num_classes=config['model']['num_classes']).to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'轮次 {epoch+1}: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%')

if __name__ == '__main__':
    main()
```

## ✨ 核心特性

### 🎨 模块化设计
- **数据模块**: 统一的数据加载和预处理
- **模型模块**: 灵活的模型定义和扩展
- **工具模块**: 配置管理和日志记录
- **训练模块**: 完整的训练流程封装

### 🏆 最佳实践
- ✅ 完整的训练/验证循环
- ✅ 实时进度监控
- ✅ 自动混合精度训练支持
- ✅ 模型保存和恢复
- ✅ TensorBoard 可视化

### 🔧 可扩展性
- 🎯 支持自定义数据集
- 🎯 灵活添加新模型
- 🎯 多种优化器和学习率调度器
- 🎯 配置驱动的训练流程

## 🚀 进阶功能

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

def train_epoch_amp(model, dataloader, criterion, optimizer, device):
    model.train()
    scaler = GradScaler()
    
    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 学习率调度

```python
# 多种学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```

## 📊 性能优化技巧

### 1. 数据加载优化
- 使用多进程数据加载 (`num_workers > 0`)
- 数据预取 (`prefetch_factor`)
- 数据增强和标准化

### 2. 训练加速
- 混合精度训练 (AMP)
- 梯度累积
- 学习率热身 (Warmup)

### 3. 内存优化
- 梯度检查点
- 模型并行
- 批量大小调整

## 🤝 如何贡献

我们欢迎各种形式的贡献！包括但不限于：

1. **报告问题**: 提交 Issue 描述你遇到的问题
2. **功能请求**: 提出新的功能建议
3. **代码贡献**: 提交 Pull Request 改进代码
4. **文档完善**: 帮助改进文档和示例

### 贡献步骤

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

如果这个项目对你有所帮助，请考虑给它一个 ⭐️，这对我是很大的鼓励！

