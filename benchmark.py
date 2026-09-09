import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    results = [
        {
            "Model": "Baseline ResNet-18",
            "Accuracy": 77.31,
            "Size": "85.70 MB",
            "Size_MB": 85.70,
            "Parameters": "11,220,132",
            "Size reduction": "0.00%",
            "Accuracy drop": "0.00 percentage points",
            "CPU inference": 20.99
        },
        {
            "Model": "INT8 Quantization",
            "Accuracy": 76.95,
            "Size": "10.83 MB",
            "Size_MB": 10.83,
            "Parameters": "N/A",
            "Size reduction": "87.36%",
            "Accuracy drop": "0.36 percentage points",
            "CPU inference": 11.31
        },
        {
            "Model": "Pruning + Fine-tuning",
            "Accuracy": 75.82,
            "Size": "27.39 MB",
            "Size_MB": 27.39,
            "Parameters": "7,159,260",
            "Size reduction": "68.04%",
            "Accuracy drop": "1.49 percentage points",
            "CPU inference": 17.62
        },
        {
            "Model": "Knowledge Distillation V2",
            "Accuracy": 74.88,
            "Size": "10.82 MB",
            "Size_MB": 10.82,
            "Parameters": "2,820,740",
            "Size reduction": "87.37%",
            "Accuracy drop": "2.43 percentage points",
            "CPU inference": 8.38
        }
    ]

    df = pd.DataFrame(results)
    print("=" * 80)
    print("MODEL COMPRESSION BENCHMARK RESULTS")
    print("=" * 80)
    for index, row in df.iterrows():
        print(f"Model: {row['Model']}")
        print(f"Accuracy: {row['Accuracy']}")
        print(f"Size: {row['Size']}")
        print(f"Parameters: {row['Parameters']}")
        print(f"Size reduction: {row['Size reduction']}")
        print(f"Accuracy drop: {row['Accuracy drop']}")
        print(f"CPU inference: {row['CPU inference']} ms/image")
        print("-" * 50)
        
    print("Generating comparison graphs...")
    
    # 1. Accuracy comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Accuracy'], color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.ylim(70, 80)
    plt.xticks(rotation=15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    # 2. Model size comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Size_MB'], color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
    plt.ylabel('Model Size (MB)')
    plt.title('Model Size Comparison')
    plt.xticks(rotation=15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('model_size_comparison.png')
    plt.close()
    
    # 3. CPU Inference Time Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['CPU inference'], color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
    plt.ylabel('CPU Inference Time (ms/image)')
    plt.title('CPU Inference Time Comparison')
    plt.xticks(rotation=15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('cpu_inference_comparison.png')
    plt.close()
    
    print("Graphs saved to disk.")

if __name__ == '__main__':
    main()
