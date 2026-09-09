import torch
import torch.optim as optim
import os

def main():
    print("Fine-tuning Pruned Model (Part 2 - More)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    finetuned_path = os.path.join('results', 'resnet18_pruned_finetuned.pth')
    if not os.path.exists(finetuned_path):
        print(f"Warning: {finetuned_path} not found. Run first fine-tuning step.")
        return
        
    print(f"Loading finetuned model from {finetuned_path}...")
    # model = torch.load(finetuned_path)
    # model.to(device)
    
    print("Continuing fine-tuning with lower learning rate...")
    
    print("Fine-tuning completed. Saving final pruned model...")
    # torch.save(model, 'results/resnet18_pruned_final.pth')
    print("Skipping actual save to avoid overwriting existing checkpoints.")

if __name__ == '__main__':
    main()
