import torch.nn as nn

class MLP(nn.Module):
    """
    Perceptron Multicapa para clasificación de cobertura forestal.
    Parámetros:
        input_size: número de características de entrada
        hidden_sizes: lista con el número de neuronas en cada capa oculta
        num_classes: número de clases de salida
        dropout_rate: probabilidad de dropout
    """
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Construir las capas dinámicamente
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Capa de salida (sin activación, se aplica en la loss function)
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self):
        """Cuenta el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
