import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader

from src.model import MLP
from src.dataset import get_dataloaders
from src.evaluate import evaluate

def create_model(trial, input_size, num_classes, device):
    """Crea un modelo MLP con hiperparámetros sugeridos por Optuna."""
    n_layers = trial.suggest_int('n_layers', 2, 5)
    
    hidden_sizes = []
    first_layer_size = trial.suggest_int('first_layer_size', 256, 512, step=64)

    for i in range(n_layers):
        if i == 0:
            hidden_sizes.append(first_layer_size)
        else:
            prev_size = hidden_sizes[-1]
            layer_size = trial.suggest_int(f'layer_{i}_size', max(64, prev_size // 2), prev_size, step=32)
            hidden_sizes.append(layer_size)

    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)

    return model

def objective(trial, data_info, device):
    """Función objetivo para Optuna."""
    input_size = data_info['input_size']
    num_classes = data_info['num_classes']
    
    # Crear modelo
    model = create_model(trial, input_size, num_classes, device)

    # Hiperparámetros de entrenamiento
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

    train_loader = DataLoader(
        data_info['train_loader'].dataset,
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        data_info['val_loader'].dataset,
        batch_size=batch_size, shuffle=False, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss(weight=data_info['class_weights'].to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_epochs = 20
    best_val_acc = 0

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

    return best_val_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Cargando datos para optimización...')
    data_info = get_dataloaders(batch_size=256)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    print('Iniciando búsqueda de hiperparámetros con Optuna...')
    study.optimize(lambda trial: objective(trial, data_info, device), n_trials=30, show_progress_bar=True)
    
    print(f'\nBúsqueda completada.')
    print(f'Mejor accuracy en validación: {study.best_value*100:.2f}%')
    print('Mejores hiperparámetros encontrados:\n')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()
