import random, torch, numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def log_results(results, filename='results/metrics.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Saved metrics to {filename}")

def plot_metric(results, metric='accuracy'):
    plt.figure(figsize=(6,4))
    for label, vals in results.items():
        plt.plot(vals[metric], label=label)
    plt.title(f'{metric.upper()} Comparison')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()
