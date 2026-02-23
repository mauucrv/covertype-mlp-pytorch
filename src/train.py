import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse

from src.model import MLP
from src.dataset import get_dataloaders
from src.evaluate import evaluate
from src.config import load_config

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Estadísticas
        running_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    return train_loss, train_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience=10):
    """Entrena el modelo completo con early stopping."""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print('Iniciando entrenamiento\n')
    print(f'{"Época":<8} {"Train Loss":<12} {"Train Acc":<12} {"Val Loss":<12} {"Val Acc":<12} {"LR":<12}')

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f'{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {current_lr:<12.6f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'\nEarly stopping en época {epoch+1}. Sin mejora por {patience} épocas.')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nMejor modelo restaurado (val_loss: {best_val_loss:.4f})')

    return history

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento MLP para Cobertura Forestal")
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Ruta al archivo de configuración YAML')
    args = parser.parse_args()

    # Cargar configuración desde el CLI
    config = load_config(args.config)
    print(f'Usando configuración cargada desde: {args.config}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}')
    
    # Cargar datos
    data_info = get_dataloaders(
        batch_size=config['data']['batch_size'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        random_state=config['data']['random_state']
    )
    
    # Crear modelo (arquitectura optimizada final reportada)
    INPUT_SIZE = data_info['input_size']
    NUM_CLASSES = data_info['num_classes']
    HIDDEN_SIZES = config['model']['hidden_sizes']
    DROPOUT_RATE = config['model']['dropout_rate']
    
    model = MLP(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    print(f'Parámetros entrenables: {model.count_parameters():,}')
    
    # Entrenar modelo final sin pesos de clase
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['training']['scheduler_factor'], 
        patience=config['training']['scheduler_patience']
    )
    
    EPOCHS = config['training']['epochs']
    
    print('Entrenando modelo...')
    history = train_model(
        model=model,
        train_loader=data_info['train_loader'],
        val_loader=data_info['val_loader'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        patience=config['training']['patience']
    )
    
    # Evaluación en test
    test_loss, test_acc = evaluate(model, data_info['test_loader'], criterion, device)
    print(f'\nEvaluación final en Test -> Loss: {test_loss:.4f}, Accuracy: {test_acc*100:.2f}%')
    
    # Guardar modelo
    models_dir = Path(config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / config['paths']['best_model_name']
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'input_size': INPUT_SIZE,
            'hidden_sizes': HIDDEN_SIZES,
            'num_classes': NUM_CLASSES,
            'dropout_rate': DROPOUT_RATE
        },
        'test_accuracy': test_acc,
        'scaler_mean': data_info['scaler'].mean_,
        'scaler_scale': data_info['scaler'].scale_
    }, model_path)
    
    print(f'Modelo guardado en: {model_path}')

if __name__ == '__main__':
    main()
