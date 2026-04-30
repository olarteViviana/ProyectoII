# UCF Crime Recognition

Pipeline inicial para clasificar imagenes del dataset **UCF Crime Dataset** de Kaggle.

El proyecto reemplaza el notebook anterior por una estructura reproducible:

- `configs/pipeline.toml`: parametros del dataset, preprocesamiento y modelo.
- `src/ucf_crime_recognition/`: codigo fuente del pipeline.
- `scripts/`: entradas simples para ejecutar cada paso.
- `data/`: datos locales ignorados por Git.
- `models/`: modelos entrenados ignorados por Git.
- `reports/`: metricas y reportes ignorados por Git.

## 1. Instalar dependencias

```bash
uv sync
```

Si prefieres `pip`:

```bash
pip install -r requirements.txt
```

## 2. Descargar el dataset

Necesitas tener configuradas tus credenciales de Kaggle.

```bash
uv run python scripts/download_dataset.py
```

Esto descarga:

```python
kagglehub.dataset_download("odins0n/ucf-crime-dataset")
```

Para no duplicar los 12 GB del dataset, el script crea un enlace local en `data/raw/` hacia la cache de Kaggle cuando el sistema lo permite.

## 3. Crear el manifiesto de imagenes

```bash
uv run python scripts/build_manifest.py
```

El manifiesto queda en `data/processed/manifest.csv` con columnas como:

- `path`
- `label`
- `split`

## 4. Entrenar un modelo base

```bash
uv run python scripts/train_model.py
```

El baseline carga las imagenes, las redimensiona, aplana pixeles normalizados y entrena una regresion logistica multiclase.
Por defecto usa una muestra limitada (`max_train_samples` y `max_test_samples`) para poder validar el flujo sin consumir demasiada memoria.

## 5. Predecir una imagen

```bash
uv run python scripts/predict_image.py ruta/a/imagen.png
```
