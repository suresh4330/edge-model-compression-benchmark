import torch
import torch.nn.quantized as nnq
from models.resnet import ResNet18
import warnings
warnings.filterwarnings('ignore')

torch.backends.quantized.engine = 'fbgemm'

model = ResNet18(num_classes=100)
model.eval()

# Let's see if we actually need to fuse. PyTorch eagerness allows prepare/convert without fusing
# but if the original model fused, we MUST fuse. The missing keys from my previous test were:
# "bn1.weight", "layer1.0.bn1.weight". This means they were FUSED in the saved model, 
# because they are missing from the state_dict! Wait! If they are missing in the state_dict, it means the model was fused!
import torch.quantization

# We must fuse Conv2d and BatchNorm2d manually for ResNet18
modules_to_fuse = [['conv1', 'bn1']]
for name, module in model.named_modules():
    if type(module).__name__ == 'BasicBlock':
        modules_to_fuse.append([f'{name}.conv1', f'{name}.bn1'])
        modules_to_fuse.append([f'{name}.conv2', f'{name}.bn2'])
        if len(module.shortcut) > 0:
            modules_to_fuse.append([f'{name}.shortcut.0', f'{name}.shortcut.1'])

torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

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
    print("Inference failed:", e)
