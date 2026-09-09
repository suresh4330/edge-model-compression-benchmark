# EDGE-BENCH: Model Compression Lab

> A reproducible benchmark and interactive dashboard for evaluating ResNet-18 compression strategies for edge deployment on CIFAR-100.

EDGE-BENCH evaluates the practical trade-offs between accuracy, model footprint, parameter count, and CPU inference latency. It pairs repeatable PyTorch workflows with a React dashboard and FastAPI inference service so results can be explored and validated in one place.

## Highlights

- Benchmarks four deployment options: baseline, INT8 quantization, structured pruning with fine-tuning, and knowledge distillation.
- Reports measured accuracy, disk size, parameters, size reduction, accuracy drop, and CPU latency.
- Includes an interactive dashboard with benchmark charts, deployment guidance, and a live image-inference lab.
- Keeps live request timing separate from the controlled benchmark latency reported in the results table.
- Uses model compression only: it optimizes neural-network models, not uploaded images, PDFs, or other files.

## Benchmark at a Glance

| Model | Accuracy | Model size | Parameters | Size reduction | Accuracy drop | CPU inference |
|---|---:|---:|---:|---:|---:|---:|
| Baseline ResNet-18 | 77.31% | 85.70 MB | 11,220,132 | 0.00% | 0.00 pp | 20.99 ms/image |
| INT8 Quantization | 76.95% | 10.83 MB | N/A | 87.36% | 0.36 pp | 11.31 ms/image |
| Pruning + Fine-tuning | 75.82% | 27.39 MB | 7,159,260 | 68.04% | 1.49 pp | 17.62 ms/image |
| Knowledge Distillation V2 | 74.88% | 10.82 MB | 2,820,740 | 87.37% | 2.43 pp | 8.38 ms/image |

### Deployment recommendations

| Priority | Recommended model | Rationale |
|---|---|---|
| Highest accuracy | Baseline ResNet-18 | Reference result with the strongest measured accuracy. |
| Best overall trade-off | INT8 Quantization | Cuts storage by 87.36% while retaining accuracy within 0.36 percentage points of the baseline. |
| Balanced option | Pruning + Fine-tuning | Reduces both footprint and parameters while retaining a conventional FP32-style workflow. |
| Smallest and fastest benchmark model | Knowledge Distillation V2 | Matches the smallest footprint and records the lowest benchmark CPU latency. |

## Compression Methods

### Baseline ResNet-18

A CIFAR-100-adapted ResNet-18 used as the accuracy and footprint reference. The architecture uses a 3×3 input convolution and omits initial max pooling.

### INT8 Quantization

Post-training static quantization converts model weights and activations from FP32 to INT8. This is the most effective choice here when preserving baseline accuracy is the priority.

### Structured Pruning + Fine-tuning

Magnitude-based structured pruning removes less-important model structure at a 20% pruning ratio. The pruned model is then fine-tuned to recover accuracy.

### Knowledge Distillation V2

A lightweight student network learns from the baseline teacher. The V2 experiment uses temperature `4.0` and alpha `0.7`, producing the smallest and fastest benchmark result.

## Architecture

```text
Browser (React + Vite dashboard)
        │
        ├── GET  /api/benchmark     → recorded benchmark metrics
        └── POST /api/inference     → prediction, confidence, live timing, model metadata
        │
FastAPI service (PyTorch CPU inference)
        │
        └── results/*.pth           → existing trained model checkpoints
```

The dashboard sections are Overview, Benchmark, Models, Performance, Compression, Edge Deployment, and Inference Lab. Sidebar links use synchronized hash navigation and active-section state.

## Quick Start

### Prerequisites

- Python 3.10+ recommended
- Node.js 18+ recommended
- npm

### 1. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Start the API

Run this command from the repository root:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`; interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Start the dashboard

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`).

### 4. Validate the frontend build

```bash
cd frontend
npm run lint
npm run build
```

## Live Inference Lab

The Inference Lab accepts PNG and JPEG images, lets you select any one of the four models, and can compare all models using the same uploaded image. Results are returned by FastAPI and include:

- Predicted CIFAR-100 class
- Prediction confidence
- Live inference time
- Recorded model size and size reduction

For meaningful predictions, use CIFAR-100-style images. The live timing is a request-time measurement and should not be compared as an exact replacement for the controlled CPU benchmark latency above.

## Reproducing Benchmark Artifacts

The recorded benchmark results and chart images are included in the repository. To regenerate the summary charts:

```bash
python benchmark.py
```

Optional experiment scripts are provided for training and compression workflows:

```bash
python train.py
python quantize_int8.py
python prune.py
python finetune_pruned.py
python finetune_pruned_more.py
python knowledge_distillation_v2.py
```

These workflows use the checkpoints in `results/`. They are not required to run the dashboard or API.

## Repository Layout

```text
.
├── backend/                  # FastAPI benchmark and inference service
├── frontend/                 # React/Vite dashboard
├── models/                   # ResNet model definition
├── results/                  # Existing checkpoints and benchmark charts
├── benchmark.py              # Benchmark summary and chart generator
├── train.py                  # Baseline training workflow
├── quantize_int8.py          # INT8 quantization workflow
├── prune.py                  # Structured pruning workflow
├── finetune_pruned*.py       # Pruning fine-tuning workflows
└── knowledge_distillation_v2.py
```

## Limitations

- Results are measured on CIFAR-100 and may not transfer directly to other datasets or production workloads.
- CPU latency depends on hardware, runtime settings, and system load.
- The project does not report power consumption or results from physical edge devices such as Raspberry Pi or Jetson hardware.
- Large checkpoint files are included for reproducibility; Git LFS is recommended for future checkpoint revisions.

## Future Work

- Benchmark on physical edge hardware.
- Measure power and memory use under deployment workloads.
- Evaluate compound compression approaches such as pruning followed by quantization.
- Extend evaluation to larger datasets and deployment targets.

## License

This repository currently does not include a license file. Add one before distributing or reusing the project outside its intended academic scope.
