import os
import sys
import warnings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from models.resnet import ResNet18
warnings.filterwarnings('ignore')

try:
    model = ResNet18(num_classes=100)
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    state_dict = torch.load(os.path.join(ROOT_DIR, 'results', 'resnet18_int8.pth'), map_location='cpu', weights_only=False)
    
    # Try strict=False in case there are minor differences or non-quantized parts
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded INT8 strictly successfully!")
    except Exception as e:
        print("Strict loading failed:", e)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded INT8 with strict=False.")
except Exception as e:
    print("Error:", e)
