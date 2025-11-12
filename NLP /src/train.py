import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
from sklearn.metrics import f1_score, accuracy_score

def train_model(model, train_loader, test_loader, optimizer_name='adam',
                grad_clip=False, num_epochs=10, device='cpu',
                config_details=None):

    criterion = nn.BCEWithLogitsLoss()  
    optimizer_cls = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}[optimizer_name.lower()]
    lr = 0.001 if optimizer_name.lower() == 'adam' else 0.01
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    model.to(device)

    metrics = {'accuracy': [], 'f1': [], 'epoch_time': []}

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        epoch_time = time.time() - start_time
        acc, f1 = evaluate(model, test_loader, device)
        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch+1:02d} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {epoch_time:.1f}s | "
              f"Model: {config_details.get('Model','')} | Act: {config_details.get('Activation','')} | "
              f"Opt: {config_details.get('Optimizer','')} | Seq: {config_details.get('Seq Length','')} | "
              f"Clip: {grad_clip}")
    return metrics


def evaluate(model, loader, device='cpu'):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)          # convert logits to probabilities
            preds.extend((probs.cpu() > 0.5).int().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='macro')
    return acc, f1
