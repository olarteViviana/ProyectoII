# UCF Crime Recognition

Pipeline inicial para clasificar imagenes del dataset **UCF Crime Dataset** de Kaggle.

El proyecto reemplaza el notebook anterior por una estructura reproducible:

- `configs/pipeline.toml`: parametros del dataset, preprocesamiento y modelo.
- `src/ucf_crime_recognition/config/`: configuracion, rutas y MLflow.
- `src/ucf_crime_recognition/data/`: descarga, manifiesto, loaders y validaciones.
- `src/ucf_crime_recognition/features/`: transformacion de imagenes a features.
- `src/ucf_crime_recognition/models/`: modelos candidatos, entrenamiento y registry.
- `src/ucf_crime_recognition/pipeline.py`: flow de Prefect.
- `scripts/`: entradas simples para ejecutar cada paso.
- `data/`: datos locales ignorados por Git.
- `models/`: modelos entrenados ignorados por Git.
- `reports/`: metricas y reportes ignorados por Git.

La arquitectura sigue el estilo del proyecto `MLOps_UdM/03-Orchestration/Prefect-pipelines`: un `pipeline.py` orquesta y el codigo de negocio vive en paquetes especializados.

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

## 4. Ejecutar pipeline con Prefect + MLflow

El proyecto puede entrenar varios modelos candidatos y escoger el mejor usando MLflow para tracking:

- `logistic_regression`
- `linear_svm`
- `random_forest`

La búsqueda de hiperparámetros se hace con Optuna sobre una validación interna, y el test final se deja solo para la evaluación del modelo seleccionado.

Prefect orquesta los pasos del flujo:

```bash
uv run ucf-flow --rebuild-manifest
```

Si tambien quieres descargar el dataset desde el flow:

```bash
uv run ucf-flow --download --rebuild-manifest
```

El mejor modelo se escoge por `f1_macro`, configurado en `configs/pipeline.toml`.

Para mejorar `f1_macro`, puedes ajustar rapidamente en `configs/pipeline.toml`:

- `preprocessing.image_size`
- `model.max_train_samples`, `model.max_validation_samples`, `model.max_test_samples`
- `model.min_train_samples_per_class`, `model.max_train_samples_per_class`
- `optuna.n_trials`, `optuna.timeout_seconds`

## 5. Ver experimentos en MLflow

El tracking local queda en `mlflow.db`. Para abrir la interfaz:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abre:

```text
http://localhost:5000
```

## 6. Entrenar sin Prefect

```bash
uv run python scripts/train_model.py
```

Este comando tambien registra experimentos en MLflow y compara los modelos candidatos.
Por defecto usa una muestra limitada (`max_train_samples` y `max_test_samples`) para poder validar el flujo sin consumir demasiada memoria.

## 7. Predecir una imagen

```bash
uv run python scripts/predict_image.py ruta/a/imagen.png
```

## 8. Servir un deployment de Prefect

```bash
uv run python deploy.py
```

El deployment queda programado diariamente a las 2:00 AM.

## 9. Abrir la interfaz SaaS

```bash
uv run streamlit run streamlit_app.py
```

La interfaz permite subir una imagen o un video, mostrar el triage del incidente, revisar el resumen del entrenamiento y leer el reporte de clasificación.
