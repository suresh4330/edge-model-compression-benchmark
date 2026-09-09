# Edge-Deployment Model-Compression Benchmark

## Project Objective
This is an academic project for benchmarking established deep-learning model compression techniques for edge deployment. We evaluate how different compression methods trade off model size, parameter count, and inference time with accuracy.

## Problem Statement
Deploying deep learning models on edge devices (such as mobile phones, IoT devices, or embedded systems) is challenging due to strict constraints on memory, compute, and battery. Model compression is essential to overcome these barriers.

## Why Model Compression is Useful for Edge Deployment
Compression reduces memory footprint, accelerates inference, and decreases power consumption. This enables complex models to run efficiently on low-resource hardware without requiring cloud offloading, preserving privacy and reducing latency.

## Dataset
We use the **CIFAR-100** dataset, containing 60,000 32x32 color images in 100 classes.

## Methodology & Evaluated Models

We benchmark established compression techniques. This project does not introduce a novel algorithm.

### Baseline ResNet-18
- CIFAR-adapted ResNet-18 (no initial max-pooling, 3x3 conv).
- Channels: 64, 128, 256, 512.
- Trained using SGD, momentum 0.9, weight decay 5e-4, cosine annealing.

### INT8 Quantization
- Post-training static quantization to INT8.
- Reduces precision of weights and activations from 32-bit floating point to 8-bit integer.

### Structured Pruning + Fine-tuning
- Magnitude-based structured pruning (20% pruning ratio) using `torch-pruning`.
- Removes entire convolutional filters, followed by fine-tuning to recover accuracy.

### Knowledge Distillation V2
- Distills knowledge from the Baseline ResNet-18 (Teacher) into a lightweight student network.
- Student architecture: 4 layers with lightweight residual blocks.
- Hyperparameters: Temperature = 4.0, Alpha = 0.7.

## Experimental Methodology
All compression methods were initialized or derived from the same base model (77.31% Baseline). We measure final validation accuracy, model size on disk, number of parameters, and CPU inference time per image.

## Final Results Table

| Model | Accuracy | Size | Parameters | Size Reduction | Accuracy Drop | CPU Inference |
|---|---|---|---|---|---|---|
| Baseline ResNet-18 | 77.31% | 85.70 MB | 11,220,132 | 0% | 0 | 20.99 ms/image |
| INT8 Quantization | 76.95% | 10.83 MB | N/A | 87.36% | 0.36 pp | 11.31 ms/image |
| Pruning + Fine-tuning | 75.82% | 27.39 MB | 7,159,260 | 68.04% | 1.49 pp | 17.62 ms/image |
| Knowledge Distillation V2 | 74.88% | 10.82 MB | 2,820,740 | 87.37% | 2.43 pp | 8.38 ms/image |

## Comparisons
- **Accuracy comparison**: See `accuracy_comparison.png`
- **Model-size comparison**: See `model_size_comparison.png`
- **CPU inference comparison**: See `cpu_inference_comparison.png`

## Discussion
- **INT8 Quantization** provided the best balance, shrinking the model size by over 87% with less than a 0.5% drop in accuracy. 
- **Pruning** effectively reduced parameters and size but suffered a moderate accuracy drop of ~1.5%.
- **Knowledge Distillation** drastically reduced the parameter count (by ~75%) and achieved the fastest inference time (8.38 ms) but experienced the largest accuracy drop (2.43 pp).

## Conclusion
Established compression methods successfully adapt heavy models for edge deployment. Quantization is highly effective for immediate size reduction without retraining. For faster theoretical compute (fewer FLOPs/parameters), Knowledge Distillation and Structured Pruning provide significant speedups at the cost of some accuracy.

## Limitations
- We only evaluated on CIFAR-100.
- CPU inference times can vary based on hardware load and architecture.
- We did not measure power consumption or deploy to a physical microcontroller/mobile device.

## Future Scope
- Combining multiple techniques (e.g., pruning followed by quantization).
- Evaluating on larger datasets (e.g., ImageNet).
- Benchmarking on actual edge hardware (e.g., Raspberry Pi, Jetson Nano).

## Installation Instructions
```bash
pip install -r requirements.txt
```

## Usage Commands
Note: Execution of training, pruning, or quantization requires pre-trained checkpoints in the `results/` folder.

```bash
# Check GPU setup
python check_gpu.py

# Generate benchmark summary and graphs
python benchmark.py

# Optional: Run workflows (Requires existing checkpoints)
python train.py
python quantize_int8.py
python prune.py
python finetune_pruned.py
python finetune_pruned_more.py
python knowledge_distillation_v2.py
```

## Project Structure
```text
edge-model-compression/
├── models/
│   ├── __init__.py
│   └── resnet.py
├── results/
├── train.py
├── quantize_int8.py
├── prune.py
├── finetune_pruned.py
├── finetune_pruned_more.py
├── knowledge_distillation_v2.py
├── benchmark.py
├── check_gpu.py
├── requirements.txt
└── README.md
```
