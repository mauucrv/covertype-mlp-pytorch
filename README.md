# Clasificación de Cobertura Forestal con MLP en PyTorch

Proyecto de clasificación multiclase utilizando un Perceptron Multicapa (MLP) entrenado con PyTorch sobre el dataset CoverType.

## Objetivo

Construir, entrenar y evaluar una red neuronal que clasifique el tipo de cobertura forestal alcanzando al menos 93% de accuracy en el conjunto de prueba.

## Resultado

Se alcanzó un 95.43% de accuracy, superando el objetivo por 2.43 puntos porcentuales.

## Dataset

El dataset CoverType del UCI Machine Learning Repository contiene:

- 581,012 muestras
- 54 características (10 continuas + 44 binarias)
- 7 clases de cobertura forestal
- Desbalance significativo entre clases (ratio 103:1)

## Modelo Final

| Característica | Valor                          |
| -------------- | ------------------------------ |
| Arquitectura   | 54 → 512 → 480 → 400 → 360 → 7 |
| Activación     | ReLU + BatchNorm + Dropout     |
| Parámetros     | 617,191                        |
| Optimizador    | Adam (lr=0.00031)              |
| Épocas         | 100                            |

## Métricas Finales

| Métrica           | Valor  |
| ----------------- | ------ |
| Accuracy          | 95.43% |
| Precision (macro) | 92.20% |
| Recall (macro)    | 93.77% |
| F1-Score (macro)  | 92.94% |

## Estructura del Proyecto

- notebooks/covertype_mlp_classification.ipynb: Desarrollo completo del proyecto
- models/covertype_mlp_final.pth: Modelo entrenado
- README.md: Documentación
- requirements.txt: Dependencias
- environment.yml: Configuración del entorno conda

## Requisitos

- Python 3.12
- CUDA 13.0
- NVIDIA GPU con soporte CUDA

## Instalación

1. Clonar el repositorio:
   git clone https://github.com/mauucrv/covertype-mlp-pytorch.git

2. Crear el entorno conda:
   conda create -n covertype-mlp python=3.12 -y
   conda activate covertype-mlp

3. Instalar PyTorch con CUDA:
   pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

4. Instalar dependencias:
   pip install scikit-learn pandas numpy matplotlib seaborn optuna tqdm jupyter

## Uso

Abrir el notebook en VS Code o Jupyter y ejecutar las celdas en orden.

## Metodología

1. Exploración de datos y análisis de distribuciones
2. Preprocesamiento: escalado de características continuas
3. Modelo inicial: MLP de 3 capas (80% accuracy)
4. Optimización con Optuna: búsqueda de hiperparámetros
5. Modelo optimizado con pesos de clase (92.44% accuracy)
6. Modelo final sin pesos de clase (95.43% accuracy)

## Autor

Brandon Mauricio Cervantes Guerrero
