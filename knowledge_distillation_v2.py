import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models.resnet import ResNet18

class LightweightResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LightweightResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class StudentNet(nn.Module):
    def __init__(self, num_classes=100):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(LightweightResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels, label_smoothing=0.1)
    return alpha * soft_loss + (1. - alpha) * hard_loss

def main():
    print("Knowledge Distillation V2 Workflow")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    teacher = ResNet18(num_classes=100)
    baseline_path = os.path.join('results', 'baseline_resnet18.pth')
    if os.path.exists(baseline_path):
        teacher.load_state_dict(torch.load(baseline_path, map_location=device))
        print("Teacher model loaded.")
    else:
        print(f"Warning: {baseline_path} not found.")
        
    student = StudentNet(num_classes=100)
    
    # Optimizer settings: SGD, lr=0.05, momentum=0.9, weight_decay=5e-4, Nesterov=True
    # optimizer = optim.SGD(student.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    print("Skipping actual KD training to avoid overwriting existing checkpoints.")

if __name__ == '__main__':
    main()
