import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CoverTypeDataset(Dataset):
    """
    Dataset personalizado para el problema de clasificación de cobertura forestal.
    Parámetros:
        X: numpy array con las características
        y: numpy array con las etiquetas
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size=256, test_size=0.15, val_size=0.176, random_state=42):
    """
    Carga, preprocesa los datos y retorna los DataLoaders junto con información del dataset.
    """
    print('Cargando dataset...')
    covtype = fetch_covtype(as_frame=True)
    
    # Separar características y etiquetas
    X = covtype.data
    y = covtype.target

    # Convertir etiquetas al rango [0, 6]
    y_adjusted = y - 1

    # División de datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_adjusted, test_size=test_size, random_state=random_state, stratify=y_adjusted
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    feature_names = X.columns.tolist()
    continuous_cols = feature_names[:10]
    binary_cols = feature_names[10:]

    # Escalar solo las características continuas
    scaler = StandardScaler()
    X_train_continuous = scaler.fit_transform(X_train[continuous_cols])
    X_val_continuous = scaler.transform(X_val[continuous_cols])
    X_test_continuous = scaler.transform(X_test[continuous_cols])

    # Características binarias
    X_train_binary = X_train[binary_cols].values
    X_val_binary = X_val[binary_cols].values
    X_test_binary = X_test[binary_cols].values

    # Concatenar
    X_train_processed = np.hstack([X_train_continuous, X_train_binary])
    X_val_processed = np.hstack([X_val_continuous, X_val_binary])
    X_test_processed = np.hstack([X_test_continuous, X_test_binary])

    # Crear los datasets
    train_dataset = CoverTypeDataset(X_train_processed, y_train)
    val_dataset = CoverTypeDataset(X_val_processed, y_val)
    test_dataset = CoverTypeDataset(X_test_processed, y_test)

    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Calcular pesos de clase para desbalance
    class_counts = np.bincount(y_train.values)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'class_weights': class_weights_tensor,
        'input_size': X_train_processed.shape[1],
        'num_classes': len(np.unique(y_adjusted))
    }
