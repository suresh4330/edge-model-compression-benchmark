import torch
from models.resnet import ResNet18
import warnings
warnings.filterwarnings('ignore')

model = ResNet18(num_classes=100)
model.eval()

# Check if we need fusion. Typical ResNet18 eager mode fusion:
modules_to_fuse = [
    ['conv1', 'bn1'] # often relu is fused too if we have one there
]
for m in model.modules():
    if type(m) == type(model.layer1[0]): # BasicBlock
        # In BasicBlock we have conv1, bn1 and conv2, bn2
        pass

# Let's try WITHOUT fusion first to see if it infers
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

state_dict = torch.load('results/resnet18_int8.pth', map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=False)

x = torch.randn(1, 3, 32, 32)
try:
    out = model(x)
    print("Inference successful! Output shape:", out.shape)
except Exception as e:
    print("Inference failed without fusion:", e)
