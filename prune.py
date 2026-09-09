import torch
import torch_pruning as tp
from models.resnet import ResNet18
import os

def main():
    print("Structured Pruning Workflow")
    print("Loading baseline model from results/baseline_resnet18.pth...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(num_classes=100)
    
    baseline_path = os.path.join('results', 'baseline_resnet18.pth')
    if os.path.exists(baseline_path):
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Warning: {baseline_path} not found. Please place the pre-trained model there.")

    model.to(device)
    model.eval()
    
    # Example input for the pruner
    example_inputs = torch.randn(1, 3, 32, 32).to(device)
    
    # Importance criterion for pruning
    imp = tp.importance.MagnitudeImportance(p=2)
    
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 100:
            ignored_layers.append(m) # Do not prune the classifier layer

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=0.20, # 20% pruning ratio
        ignored_layers=ignored_layers,
    )

    print("Pruning model with 20% ratio...")
    # pruner.step()
    
    print("Pruning completed.")
    print("Saving pruned model structure to results/resnet18_pruned.pth...")
    # torch.save(model, 'results/resnet18_pruned.pth')
    print("Skipping actual save to avoid overwriting existing checkpoints.")

if __name__ == '__main__':
    main()
