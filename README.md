# Clasificacion de Cobertura Forestal con MLP en PyTorch

Proyecto de clasificacion multiclase utilizando un Perceptron Multicapa (MLP) entrenado con PyTorch sobre el dataset CoverType de sklearn.

## Objetivo

Construir, entrenar y evaluar una red neuronal que clasifique el tipo de cobertura forestal alcanzando al menos 93% de accuracy en el conjunto de prueba.

## Dataset

El dataset CoverType contiene 581,012 muestras con 54 caracteristicas cartograficas para predecir 7 tipos de cobertura forestal en areas silvestres de Roosevelt National Forest, Colorado.

## Estructura del proyecto

- notebooks/: Jupyter notebook con el desarrollo completo del proyecto
- README.md: Documentacion del proyecto
- requirements.txt: Dependencias del proyecto
- environment.yml: Configuracion del entorno conda

## Requisitos

- Python 3.12
- CUDA 13.0
- NVIDIA GPU con soporte CUDA

## Instalacion

1. Clonar el repositorio
2. Crear el entorno conda:
   conda env create -f environment.yml
3. Activar el entorno:
   conda activate covertype-mlp
4. Instalar PyTorch con CUDA:
   pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130

## Uso

Abrir el notebook en VS Code o Jupyter Lab y ejecutar las celdas en orden.

## Autor

Brandon
