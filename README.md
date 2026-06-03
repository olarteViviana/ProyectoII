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

## 10. Ejecutar pruebas del backend

Las pruebas iniciales cubren validacion de manifiestos, carga y balanceo de datos, creacion del manifiesto y seleccion de modelos sin entrenar redes pesadas.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## 11. Levantar API REST con FastAPI

La API expone el backend de inferencia como endpoints HTTP. Puedes iniciarla con:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn ucf_crime_recognition.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Tambien queda disponible el comando del proyecto:

```bash
uv run ucf-api
```

Documentacion interactiva:

```text
http://127.0.0.1:8000/docs
```

Endpoints principales:

- `GET /api/v1/health`: estado de la API.
- `GET /api/v1/metadata`: clases y extensiones soportadas.
- `POST /api/v1/predict/image`: predice una imagen subida como `multipart/form-data` en el campo `file`.
- `POST /api/v1/predict/video?frame_samples=16&clip_window_seconds=2&motion_priority=true`: predice un video, evalua 16 anclas por defecto y combina cobertura temporal con momentos de mayor movimiento. Para cada ancla reconstruye un clip tecnico de 16 frames repartidos en una ventana temporal de 2 segundos antes de agregar la decision.

Ejemplo para imagen:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict/image" \
  -F "file=@ruta/a/imagen.png"
```

Ejemplo para video:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict/video?frame_samples=16&clip_window_seconds=2&motion_priority=true" \
  -F "file=@ruta/a/video.mp4"
```

### Conexion frontend-backend

La interfaz Streamlit actual puede seguir llamando funciones Python del backend directamente. La API REST agrega una segunda forma de consumo para otros frontends, pruebas con Postman/cURL o integraciones externas.

- Imagen: `_save_upload()` guarda el archivo temporal y luego `predict_image_details()` carga el modelo, extrae features y devuelve `prediction`, `confidence`, `class_scores` y `model_path`.
- Video: `_extract_sampled_frames()` elige anclas repartidas en el video y, si `motion_priority` esta activo, reserva parte de las anclas para picos de movimiento detectados por diferencia visual entre frames. Luego guarda 16 frames por ancla dentro de una ventana temporal configurable, `predict_image_details()` evalua cada clip y `_aggregate_video_predictions()` consolida la decision operativa.

En video, `frame_samples` controla cuantas anclas se evaluan, no el largo interno del clip. Esas anclas mezclan cobertura de inicio, mitad y final con regiones de alto movimiento cuando `motion_priority=true`. El largo interno del clip tecnico es `VIDEO_CLIP_LEN = 16`; `clip_window_seconds` controla cuanto contexto temporal cubren esos 16 frames. En la interfaz, cada ancla tambien puede abrir un segmento reproducible del video original con todos los FPS disponibles entre `clip_start_frame` y `clip_end_frame`. La tabla muestra `motion_score` y `sampling_reason` para revisar si el clip entro por movimiento, cobertura o ambas.

Para mantener el tiempo de respuesta razonable, el modelo se carga una sola vez por ejecucion y se reutiliza para los clips siguientes. La galeria de la interfaz muestra thumbnails rapidos de todos los clips evaluados; el GIF tecnico y el segmento reproducible completo se generan solo para el clip seleccionado.

Las pruebas de contrato frontend-backend estan en `tests/test_frontend_backend_contract.py`.
Las pruebas REST estan en `tests/test_api.py`.

## 12. Usar VideoMAE como extractor de video

El proyecto puede usar un transformer de video como extractor de embeddings. La configuracion activa es:

```toml
[preprocessing]
feature_extractor = "videomae"
video_model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
```

Con este modo, cada clip tecnico de 16 frames se procesa con VideoMAE y produce un embedding de 768 dimensiones. Luego se conserva el clasificador multi-etiqueta, la busqueda de umbrales, la seleccion por `risk_f1_macro`, la agregacion por clips y la interfaz actual.

Despues de cambiar a VideoMAE hay que reentrenar, porque el modelo anterior fue entrenado con otro espacio de features:

```bash
uv run ucf-flow --rebuild-manifest
```

La primera ejecucion descargara el checkpoint de Hugging Face y guardara embeddings en `data/processed/embeddings`, por lo que puede tardar bastante mas que una prediccion normal. Las siguientes ejecuciones reutilizan cache.

## 13. Desplegar con Docker

El proyecto incluye `Dockerfile` y `docker-compose.yml` para levantar la API, la interfaz Streamlit y MLflow con la misma imagen.

Construir la imagen:

```bash
docker compose build
```

Levantar API, dashboard y MLflow:

```bash
docker compose up api dashboard mlflow
```

Servicios disponibles:

- API REST: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`
- MLflow: `http://localhost:5001`

La imagen no empaqueta `data/`, `models/`, `reports/` ni `mlruns/`; Docker Compose los monta desde el proyecto local para reutilizar los artefactos entrenados y evitar imagenes pesadas. Si la API responde que no encuentra el modelo, verifica que exista `models/ucf_crime_baseline.joblib` o ejecuta entrenamiento.

Entrenar desde Docker:

```bash
docker compose --profile train run --rm train
```

Ver logs:

```bash
docker compose logs -f api
```

Detener servicios:

```bash
docker compose down
```
