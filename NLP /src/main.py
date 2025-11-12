import torch
import torchtext
from preprocess import load_data
from models import SentimentRNN
from train import train_model
from utils import set_seed, log_results

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seq_lengths = [25, 50, 100]
architectures = ['RNN', 'LSTM']
activations = ['relu', 'tanh']
optimizers = ['adam', 'sgd']
grad_clip_opts = [False, True]

results = []
csv_path = "src/IMDB Dataset.csv"
for seq_len in seq_lengths:
    train_loader, test_loader, vocab_size = load_data(csv_path, seq_len=50)
    for arch in architectures:
        for act in activations:
            for opt in optimizers:
                for clip in grad_clip_opts:
                    model = SentimentRNN(vocab_size, model_type=arch, activation=act)
                    metrics = train_model(
    model,
    train_loader,
    test_loader,
    optimizer_name=opt,
    grad_clip=clip,
    device=device,
    config_details={
        'Model': arch,
        'Activation': act,
        'Optimizer': opt,
        'Seq Length': seq_len
    }
)

                    results.append({
                        'Model': arch,
                        'Activation': act,
                        'Optimizer': opt,
                        'Seq Length': seq_len,
                        'Grad Clipping': clip,
                        'Accuracy': max(metrics['accuracy']),
                        'F1': max(metrics['f1']),
                        'Epoch Time (s)': sum(metrics['epoch_time']) / len(metrics['epoch_time'])
                    })

log_results(results)
