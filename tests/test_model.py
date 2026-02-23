import torch
from src.model import MLP

def test_mlp_forward_pass_shape():
    """Prueba que el modelo retorna un tensor de tamaño (batch_size, num_classes)."""
    batch_size = 32
    input_size = 54
    num_classes = 7
    hidden_sizes = [256, 128]

    # Crear modelo
    model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
    
    # Crear tensor dummy
    dummy_input = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = model(dummy_input)
    
    # Validaciones
    assert output.shape == (batch_size, num_classes), f"Forma esperada ({batch_size}, {num_classes}), obtenida {output.shape}"
    assert not torch.isnan(output).any(), "La salida contiene valores NaN."

def test_mlp_parameter_count():
    """Prueba que la función cuenta correctamente los parámetros (no da cero)."""
    model = MLP(input_size=10, hidden_sizes=[20], num_classes=2)
    param_count = model.count_parameters()
    
    # Input a Oculta: 10 * 20 + 20 (bias) = 220
    # BatchNorm: 2 * 20 = 40
    # Oculta a Salida: 20 * 2 + 2 (bias) = 42
    # Total esperado: 302
    assert param_count == 302, f"Se esperaban 302 parámetros, se obtuvieron {param_count}"
