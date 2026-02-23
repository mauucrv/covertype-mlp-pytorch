import torch
import numpy as np

def evaluate(model, data_loader, criterion, device):
    """
    Evalúa el modelo en un conjunto de datos.
    Retorna:
        eval_loss: pérdida promedio
        eval_acc: accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    eval_loss = running_loss / total
    eval_acc = correct / total

    return eval_loss, eval_acc
