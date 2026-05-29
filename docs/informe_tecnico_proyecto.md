# Informe tecnico del proyecto UCF Crime Recognition

## 1. Proposito del proyecto

Este proyecto implementa un pipeline de aprendizaje automatico para clasificar eventos de seguridad a partir de imagenes extraidas del dataset UCF Crime. La aplicacion busca apoyar un flujo de triage operativo: dada una imagen o un video de camara de seguridad, el sistema estima si existe un evento como robo, pelea, incendio, accidente, vandalismo u otro tipo de incidente.

El proyecto no es solamente un script de entrenamiento. Esta organizado como una solucion de MLOps local con:

- configuracion centralizada en TOML;
- descarga y preparacion reproducible del dataset;
- creacion de un manifiesto de datos;
- extraccion de caracteristicas visuales;
- entrenamiento de varios modelos candidatos;
- busqueda de hiperparametros con Optuna;
- seguimiento experimental con MLflow;
- orquestacion con Prefect;
- registro del mejor modelo;
- inferencia sobre imagenes;
- analisis de videos por muestreo de frames;
- interfaz Streamlit para demostracion.

La version actual del proyecto fue adaptada para soportar clasificacion multi-etiqueta mediante `MultiOutputClassifier`, porque en un video real pueden aparecer simultaneamente varios tipos de evento, por ejemplo `Robbery` y `Fighting`.

## 2. Problema de machine learning

### 2.1 Entrada

La unidad basica de entrenamiento es una imagen. Cada fila del manifiesto contiene:

- `path`: ruta local de la imagen;
- `label`: etiqueta o etiquetas asociadas;
- `split`: particion `train` o `test`.

En modo multi-etiqueta, la columna `label` puede contener varias clases separadas por caracteres como:

```text
Robbery|Fighting
Robbery,Fighting
Robbery;Fighting
Robbery+Fighting
```

Internamente, esas etiquetas se transforman en una matriz binaria. Por ejemplo, si las clases son `Robbery`, `Fighting` y `NormalVideos`, una muestra `Robbery|Fighting` se representa como:

```text
Robbery = 1
Fighting = 1
NormalVideos = 0
```

### 2.2 Salida

En clasificacion multiclass tradicional, el modelo devuelve una sola clase. En la version actual, cuando `multi_output = true`, el modelo puede devolver varias clases activas:

```python
{
    "prediction": "Robbery|Fighting",
    "predictions": ["Robbery", "Fighting"],
    "confidence": 0.73,
    "class_scores": {
        "Robbery": 0.73,
        "Fighting": 0.61,
        "NormalVideos": 0.18
    },
    "class_thresholds": {
        "Robbery": 0.35,
        "Fighting": 0.30,
        "NormalVideos": 0.55
    }
}
```

### 2.3 Clases consideradas

Las clases base se definen en `src/ucf_crime_recognition/config/constants.py`:

- `Abuse`
- `Arrest`
- `Arson`
- `Assault`
- `Burglary`
- `Explosion`
- `Fighting`
- `NormalVideos`
- `RoadAccidents`
- `Robbery`
- `Shooting`
- `Shoplifting`
- `Stealing`
- `Vandalism`

## 3. Arquitectura general del proyecto

La estructura principal es:

```text
ProyectoII/
├── configs/
│   └── pipeline.toml
├── docs/
│   └── informe_tecnico_proyecto.md
├── scripts/
│   ├── build_manifest.py
│   ├── download_dataset.py
│   ├── predict_image.py
│   └── train_model.py
├── src/
│   └── ucf_crime_recognition/
│       ├── config/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── tools/
│       ├── ui/
│       ├── download.py
│       ├── manifest.py
│       ├── pipeline.py
│       ├── predict.py
│       └── train.py
├── reports/
├── models/
├── pipeline.py
├── streamlit_app.py
├── deploy.py
├── pyproject.toml
└── README.md
```

La arquitectura separa responsabilidades:

- `config/`: rutas, constantes, lectura de configuracion y MLflow.
- `data/`: descarga, manifest, loaders, muestreo, balanceo y validacion.
- `features/`: conversion de imagenes a vectores numericos.
- `models/`: modelos candidatos, entrenamiento, Optuna, metricas y registro.
- `tools/`: analisis de video y validacion de politicas temporales.
- `ui/`: interfaz Streamlit.
- archivos raiz: entrypoints simples para ejecutar el paquete.

## 4. Configuracion central

El archivo `configs/pipeline.toml` controla el comportamiento del pipeline.

### 4.1 Dataset

```toml
[dataset]
kaggle_slug = "odins0n/ucf-crime-dataset"
raw_dir = "data/raw"
manifest_path = "data/processed/manifest.csv"
```

Indica el dataset de Kaggle, la ruta local para datos crudos y la ruta donde se guarda el manifiesto procesado.

### 4.2 Preprocesamiento

```toml
[preprocessing]
image_size = 96
color_mode = "rgb"
feature_extractor = "r2plus1d_18"
embedding_cache_dir = "data/processed/embeddings"
test_size = 0.2
validation_size = 0.25
random_state = 42
```

- `image_size`: tamano usado si se extraen caracteristicas tradicionales.
- `color_mode`: modo de color para la extraccion tradicional.
- `feature_extractor`: extractor preentrenado usado para embeddings. Puede ser `resnet50`, `vgg16`, `r3d_18` o `r2plus1d_18`.
- `embedding_cache_dir`: carpeta donde se guardan embeddings para evitar recalcularlos en cada entrenamiento.
- `test_size`: proporcion usada si el dataset no trae split definido.
- `validation_size`: proporcion del entrenamiento reservada para validacion interna.
- `random_state`: semilla para reproducibilidad.

### 4.3 Modelo

```toml
[model]
output_path = "models/ucf_crime_baseline.joblib"
multi_output = true
max_iter = 300
class_weight = "balanced"
max_train_samples = 10000
max_validation_samples = 1200
max_test_samples = 1200
min_train_samples_per_class = 250
max_train_samples_per_class = 800
candidate_models = [
    "logistic_regression",
    "linear_svm",
    "extra_trees",
    "random_forest",
]
selection_metric = "risk_f1_macro"
```

Puntos importantes:

- `multi_output = true` activa el enfoque multi-etiqueta con `MultiOutputClassifier`.
- `class_weight = "balanced"` ayuda a penalizar errores en clases minoritarias.
- `max_*_samples` limita el tamano para que el pipeline sea ejecutable localmente.
- `min_train_samples_per_class` y `max_train_samples_per_class` controlan rebalanceo.
- `selection_metric = "risk_f1_macro"` selecciona el modelo por desempeno en clases de riesgo, excluyendo `NormalVideos`.

### 4.4 Busqueda de umbrales

```toml
[model.threshold_search]
min = 0.05
max = 0.95
steps = 19
```

En modo multi-etiqueta, el sistema no usa un umbral fijo de 0.5 para todas las clases. Durante validacion busca un threshold por clase que maximiza F1. Esto es importante porque clases raras como `Robbery` o `Fighting` pueden necesitar umbrales menores para mejorar recall.

### 4.5 Reportes, MLflow y registry

```toml
[reports]
classification_report = "reports/classification_report.txt"
confusion_matrix = "reports/confusion_matrix.csv"
experiment_summary = "reports/experiment_summary.csv"

[mlflow]
tracking_uri = "sqlite:///mlflow.db"
experiment_name = "ucf-crime-recognition"

[registry]
enabled = true
model_name = "ucf-crime-image-classifier"
```

El proyecto guarda reportes locales y registra experimentos en MLflow usando una base SQLite local.

## 5. Pipeline completo paso a paso

El pipeline principal esta en `src/ucf_crime_recognition/pipeline.py`. Se ejecuta con:

```bash
uv run ucf-flow --rebuild-manifest
```

O, si tambien se quiere descargar el dataset:

```bash
uv run ucf-flow --download --rebuild-manifest
```

### 5.1 Paso 1: cargar configuracion

El flow llama a `load_config()`, que lee `configs/pipeline.toml`. Si se pasa `--config`, usa ese archivo alternativo.

La funcion `project_path()` convierte rutas relativas en rutas absolutas dentro del proyecto. Esto evita problemas cuando se ejecuta desde Prefect, scripts o consola.

### 5.2 Paso 2: configurar MLflow

`setup_mlflow()`:

1. lee `tracking_uri`;
2. convierte `sqlite:///mlflow.db` a una ruta absoluta;
3. llama `mlflow.set_tracking_uri(...)`;
4. llama `mlflow.set_experiment("ucf-crime-recognition")`.

Desde ese momento, todos los runs quedan asociados al experimento `ucf-crime-recognition`.

### 5.3 Paso 3: descarga opcional

Si se ejecuta con `--download`, Prefect ejecuta la tarea `download_dataset_task()`. Internamente llama:

```python
download_dataset(config_path)
```

Esa funcion usa `kagglehub.dataset_download("odins0n/ucf-crime-dataset")`. Luego intenta crear un symlink en `data/raw/` hacia la cache local de Kaggle. Si el sistema no permite symlinks, copia el directorio.

### 5.4 Paso 4: construccion del manifiesto

Si se pasa `--rebuild-manifest` o no existe `data/processed/manifest.csv`, se ejecuta `build_manifest_task()`.

`build_manifest()`:

1. recorre `data/raw`;
2. busca archivos con extension de imagen: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`;
3. infiere la etiqueta mirando partes de la ruta;
4. detecta si la ruta contiene un split existente como `train`, `training`, `test`, `validation`;
5. si todos los archivos ya tienen split, lo respeta;
6. si no, crea train/test con `train_test_split`;
7. guarda el CSV en `data/processed/manifest.csv`.

El manifiesto permite desacoplar el entrenamiento de la estructura fisica de carpetas.

### 5.5 Paso 5: entrenamiento y seleccion

La tarea `train_task()` llama a `train(config_path)`, definida en `src/ucf_crime_recognition/models/training.py`.

El entrenamiento hace:

1. carga y valida el manifest;
2. imprime distribuciones de clases;
3. separa train/test;
4. aplica muestreo para limitar volumen;
5. separa train/validation;
6. aplica balanceo;
7. extrae features;
8. convierte etiquetas a matriz multilabel si `multi_output = true`;
9. entrena cada modelo candidato con Optuna;
10. selecciona el mejor por `f1_macro`;
11. reentrena el mejor con train + validation;
12. evalua en test;
13. guarda modelo y reportes;
14. registra modelo en MLflow Model Registry.

### 5.6 Paso 6: artefactos de Prefect

Al terminar, el flow crea:

- un artifact de tabla con el mejor modelo, score, run ID y tracking URI;
- un artifact Markdown con resumen del pipeline;
- un archivo local `prefect_run_id.txt` con el run ID ganador.

## 6. Preparacion y validacion de datos

### 6.1 Validacion del manifiesto

`validate_manifest()` exige:

- columnas `path`, `label`, `split`;
- que el manifest no este vacio;
- que existan al menos dos clases.

Esto evita entrenar con datos incompletos o mal estructurados.

### 6.2 Muestreo

`sample_manifest()` reduce el numero de muestras cuando se definen limites como:

```toml
max_train_samples = 10000
max_validation_samples = 1200
max_test_samples = 1200
```

El muestreo intenta preservar proporcion por etiqueta agrupando por `label`.

### 6.3 Balanceo multiclass clasico

En modo tradicional, `rebalance_manifest()` agrupa por `label` y lleva cada clase a un tamano objetivo calculado desde:

- `min_train_samples_per_class`;
- `max_train_samples_per_class`;
- mediana de las clases.

Si una clase tiene menos datos que el objetivo, hace oversampling con reemplazo. Si tiene mas, hace undersampling.

### 6.4 Balanceo multi-etiqueta

En modo `multi_output`, el balanceo usa un `label_parser`. Esto permite contar etiquetas individuales aunque una fila tenga varias.

Ejemplo:

```text
Robbery|Fighting
```

Cuenta como una muestra positiva para `Robbery` y tambien una para `Fighting`.

El balanceo multi-etiqueta:

1. explota la columna de etiquetas en etiquetas individuales;
2. calcula conteos por clase individual;
3. reduce clases sobrerrepresentadas cuando aparecen como etiqueta unica;
4. aumenta clases minoritarias por oversampling de filas que contienen esa etiqueta;
5. conserva filas multi-etiqueta cuando ayudan a varias clases.

Esto es mejor que balancear por el texto completo de `label`, porque `Robbery|Fighting` no deberia tratarse como una clase completamente nueva.

## 7. Extraccion de caracteristicas visuales

El modulo `src/ucf_crime_recognition/features/engineering.py` convierte imagenes en vectores numericos.

### 7.1 Modo principal: embeddings CNN preentrenados

Por defecto `build_feature_matrix(..., use_pretrained=True)` usa un extractor CNN preentrenado. El extractor se configura en `configs/pipeline.toml`:

```toml
[preprocessing]
feature_extractor = "r2plus1d_18"
```

Actualmente el codigo permite comparar:

- `resnet50`: ResNet-50 preentrenada en ImageNet, salida de 2048 dimensiones.
- `vgg16`: VGG16 preentrenada en ImageNet, salida de 4096 dimensiones.
- `r3d_18`: red 3D preentrenada en Kinetics-400, salida de 512 dimensiones.
- `r2plus1d_18`: red R(2+1)D preentrenada en Kinetics-400, salida de 512 dimensiones.

Flujo:

1. carga el extractor seleccionado con `torchvision.models`;
2. elimina la capa final de clasificacion;
3. deja la red como extractor de embeddings;
4. convierte cada imagen a RGB si el extractor es 2D;
5. si el extractor es de video, agrupa frames vecinos del mismo video para formar un clip de 16 frames;
5. redimensiona a 256;
6. recorta centro a 224x224;
7. normaliza con medias y desviaciones de ImageNet;
8. obtiene un vector numerico que resume la imagen.

Justificacion:

- la CNN preentrenada ya aprendio representaciones visuales generales;
- reduce la necesidad de entrenar una CNN completa desde cero;
- permite usar clasificadores tradicionales sobre embeddings;
- es razonable cuando el dataset local es limitado o desbalanceado.

Limitacion:

- ImageNet no esta especializado en vigilancia o eventos criminales;
- ImageNet tampoco esta especializado en relaciones humanas ni interacciones persona-persona;
- una imagen aislada puede no contener suficiente contexto temporal;
- el dominio visual de UCF Crime puede diferir de ImageNet.

### 7.2 Retroalimentacion del profesor: ResNet, VGG16 y COCO

La observacion del profesor es tecnica y valida: ResNet-50 con pesos de ImageNet puede no ser el mejor extractor para este problema porque ImageNet se centra en reconocimiento de objetos y escenas generales, no en relaciones humanas, violencia, robo o interacciones entre personas.

VGG16 tambien suele usarse con pesos de ImageNet. Por eso cambiar ResNet-50 por VGG16 no garantiza resolver el problema de fondo; mas bien sirve como experimento comparativo para saber si otra arquitectura visual produce embeddings mas utiles para los clasificadores del proyecto.

COCO es una linea de mejora distinta. Un modelo entrenado en COCO normalmente se usa para deteccion de objetos, por ejemplo:

- personas;
- carros;
- bolsos;
- mochilas;
- botellas;
- celulares u otros objetos;
- posiciones aproximadas por bounding boxes.

Esto puede ser mas relevante para vigilancia porque permite construir caracteristicas como:

- numero de personas detectadas;
- distancia entre personas;
- presencia de objetos transportables;
- cercania entre persona y mostrador;
- persistencia de personas u objetos a lo largo del video;
- relacion persona-objeto.

Sin embargo, un detector COCO no clasifica directamente `Robbery` o `Fighting`. Proporciona detecciones intermedias. Para aprovecharlo bien, el pipeline deberia extraer features de objetos/personas y luego entrenar otro clasificador sobre esas features, posiblemente combinadas con embeddings CNN.

### 7.3 Modo alternativo: features tradicionales

`load_image_vector()` extrae:

- HOG: orientaciones de gradiente, util para formas y contornos;
- histogramas de color;
- estadisticas de imagen, media y desviacion;
- densidad de bordes con Canny;
- miniatura espacial.

Este modo sirve como alternativa cuando no se quiere usar PyTorch/CNNs preentrenadas, pero usualmente los embeddings profundos deberian capturar informacion mas rica.

### 7.4 Cache de embeddings

Extraer embeddings con ResNet-50 o VGG16 es costoso porque cada imagen debe pasar por una red neuronal profunda. Para poder aumentar datos de entrenamiento y repetir experimentos sin recalcular todo, el proyecto usa cache en:

```toml
embedding_cache_dir = "data/processed/embeddings"
```

La cache guarda un archivo `.npy` por imagen y extractor. El identificador incluye:

- ruta de la imagen;
- extractor usado (`resnet50` o `vgg16`);
- fecha de modificacion del archivo;
- tamano del archivo.

Esto permite que, si se corre de nuevo el pipeline con el mismo extractor y las mismas imagenes, se reutilicen los embeddings ya calculados. Si la imagen cambia, la cache se invalida automaticamente porque cambia su metadata.

## 8. Modelos candidatos y justificacion

Los modelos estan en `src/ucf_crime_recognition/models/candidates.py`.

Todos reciben como entrada vectores numericos, normalmente embeddings CNN preentrenados. Si `feature_extractor = "resnet50"`, la entrada tiene 2048 dimensiones. Si `feature_extractor = "vgg16"`, la entrada tiene 4096 dimensiones.

### 8.1 Logistic Regression

Modelo lineal probabilistico. En el proyecto se usa con:

- regularizacion `C`;
- solver `lbfgs` o `saga`;
- `class_weight = "balanced"`;
- `StandardScaler`.

Por que se usa:

- es una linea base fuerte para embeddings;
- es interpretable comparado con modelos mas complejos;
- produce probabilidades con `predict_proba`;
- permite diagnosticar si los embeddings ya separan clases linealmente.

Riesgos:

- puede quedarse corto si la separacion entre clases no es lineal;
- con muchas clases desbalanceadas puede favorecer clases dominantes.

### 8.2 Linear SVM

Modelo lineal de margen maximo. En el proyecto se usa `LinearSVC` con:

- hiperparametro `C`;
- `class_weight = "balanced"`;
- `StandardScaler`.

Por que se usa:

- funciona bien en espacios de alta dimension;
- suele ser fuerte con embeddings;
- maximiza margen, lo que puede generalizar bien;
- es eficiente frente a SVM con kernel.

Riesgos:

- `LinearSVC` no produce probabilidades directas;
- se aproxima una probabilidad a partir del `decision_function`;
- puede necesitar calibracion adicional si se quiere interpretar scores como probabilidad real.

### 8.3 Random Forest

Ensemble de arboles de decision entrenados con subconjuntos aleatorios.

Por que se usa:

- captura relaciones no lineales;
- es robusto frente a ruido;
- no requiere escalado;
- ayuda a comparar si hay patrones no lineales que los modelos lineales no capturan.

Riesgos:

- puede sobreajustar si hay muchas dimensiones y pocos ejemplos por clase;
- puede ser pesado si aumenta mucho `n_estimators`;
- en embeddings densos a veces no supera a modelos lineales.

### 8.4 Extra Trees

`ExtraTreesClassifier` es similar a Random Forest, pero introduce mas aleatoriedad en los cortes de los arboles.

Por que se usa:

- puede reducir varianza;
- suele entrenar rapido;
- explora fronteras no lineales;
- sirve como alternativa robusta a Random Forest.

Riesgos:

- puede requerir tuning cuidadoso;
- si la senal es debil, la aleatoriedad puede bajar precision;
- al igual que Random Forest, no siempre es ideal para embeddings de alta dimension.

### 8.5 RiskAwareClassifier

Este wrapper se usa en modo multiclass clasico. Su proposito es reducir un problema operativo: que el modelo prediga `NormalVideos` aunque exista una clase riesgosa con probabilidad cercana.

Funcionamiento:

1. calcula scores por clase;
2. identifica la clase con mayor score;
3. si la ganadora es `NormalVideos`, busca la mejor clase de riesgo;
4. si la clase de riesgo esta suficientemente cerca de `NormalVideos`, cambia la decision.

Esto se controla con `normal_switch_ratio`.

En modo multi-output, este wrapper no se usa; se usa `MultiOutputClassifier`.

### 8.6 MultiOutputClassifier

`MultiOutputClassifier` transforma un problema multi-etiqueta en varios clasificadores binarios independientes:

```text
Clasificador 1: Abuse vs no Abuse
Clasificador 2: Arrest vs no Arrest
Clasificador 3: Arson vs no Arson
...
Clasificador N: NormalVideos vs no NormalVideos
```

Por que se usa:

- permite que una muestra tenga varias etiquetas activas;
- es compatible con clasificadores de scikit-learn;
- evita forzar que `Robbery` y `Fighting` sean mutuamente excluyentes;
- se ajusta a la recomendacion del profesor para videos con eventos simultaneos.

Limitacion importante:

- cada salida se aprende de forma independiente;
- no modela correlaciones entre clases;
- por ejemplo, no aprende explicitamente que `Robbery` y `Fighting` pueden coocurrir;
- para modelar dependencia entre etiquetas se podria evaluar `ClassifierChain`.

## 9. Optuna y seleccion de hiperparametros

Cada modelo candidato se optimiza con Optuna. Para cada trial:

1. se sugieren hiperparametros;
2. se construye el modelo;
3. se entrena con train;
4. se evalua con validation;
5. se registran metricas en MLflow;
6. se devuelve la metrica de seleccion.

La metrica usada es:

```toml
selection_metric = "f1_macro"
```

### 9.1 Por que usar F1 macro y risk_f1_macro

F1 macro calcula F1 por clase y luego promedia sin ponderar por soporte. Esto es importante porque el dataset esta desbalanceado. Si se usara accuracy, `NormalVideos` dominaria la evaluacion.

Ejemplo del problema:

```text
NormalVideos: 467 muestras de 800
Robbery:        6 muestras de 800
Abuse:          2 muestras de 800
Fighting:       9 muestras de 800
```

Un modelo podria obtener accuracy aparentemente aceptable prediciendo casi todo como normal, pero seria inutil para detectar clases raras. F1 macro castiga ese comportamiento.

Como mejora posterior, el proyecto tambien calcula `risk_f1_macro`, que es un F1 macro restringido a las clases diferentes de `NormalVideos`. Esta metrica es mas coherente con el objetivo operativo de vigilancia, porque el interes principal no es reconocer normalidad sino detectar eventos de riesgo.

Actualmente el config usa:

```toml
selection_metric = "risk_f1_macro"
```

Esto obliga a Optuna a preferir modelos que funcionen mejor en clases como `Robbery`, `Fighting`, `Burglary`, `Shooting`, etc., aunque `NormalVideos` sea la clase con mas ejemplos.

### 9.2 Busqueda de thresholds por clase

En multi-etiqueta, cada clase tiene un score. La decision binaria depende de un umbral:

```text
score >= threshold => clase activa
score < threshold  => clase inactiva
```

Usar 0.5 para todas las clases suele ser malo en datasets desbalanceados. Por eso el pipeline busca un threshold por clase en validacion.

El rango actual es:

```text
0.05, 0.10, 0.15, ..., 0.95
```

Para cada clase, se elige el threshold que maximiza F1 en validacion.

El modelo final guarda:

- `label_classes_`;
- `label_thresholds_`;
- `multi_output_ = True`.

Asi, la inferencia usa la misma regla que se valido durante entrenamiento.

## 10. Metricas y reportes

El entrenamiento guarda:

- `reports/classification_report.txt`;
- `reports/confusion_matrix.csv`;
- `reports/experiment_summary.csv`;
- modelo local en `models/ucf_crime_baseline.joblib`;
- runs y artefactos en MLflow.

### 10.1 Metricas en modo multi-etiqueta

El pipeline calcula:

- `accuracy`: exact match ratio. En multilabel exige que todas las etiquetas de una muestra coincidan exactamente.
- `f1_macro`: promedio F1 por etiqueta sin ponderar.
- `f1_micro`: F1 global contando verdaderos positivos, falsos positivos y falsos negativos agregados.
- `f1_weighted`: F1 ponderado por soporte.
- `f1_samples`: F1 calculado por muestra y luego promediado.
- `jaccard_samples`: similitud entre conjunto real y conjunto predicho por muestra.
- `hamming_loss`: proporcion de etiquetas individuales mal predichas.

### 10.2 Interpretacion

En multi-etiqueta:

- `accuracy` puede verse baja aunque el modelo acierte algunas etiquetas, porque exige coincidencia exacta.
- `f1_macro` es muy sensible a clases raras.
- `f1_micro` suele ser mas estable si hay muchas etiquetas negativas.
- `hamming_loss` bajo indica pocos errores por etiqueta, pero puede ocultar mala deteccion de clases raras.
- `classification_report` permite ver precision, recall y F1 por etiqueta.

### 10.3 Situacion observada antes de la mejora de balanceo/thresholds

El reporte previo mostraba metricas bajas en clases minoritarias. Por ejemplo:

```text
Robbery        support 6   f1-score 0.00
Fighting       support 9   f1-score 0.00
Abuse          support 2   f1-score 0.00
NormalVideos   support 467 f1-score 0.75
```

Esto indica que el modelo estaba aprendiendo mucho mejor la clase normal que los eventos raros. Por eso se agrego:

- balanceo multi-etiqueta;
- busqueda de umbral por clase;
- reporte de threshold en `confusion_matrix.csv`;
- persistencia de thresholds en el modelo final.

## 11. MLflow

MLflow se usa para tracking experimental y registro del mejor modelo.

Para abrir la UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego entrar a:

```text
http://127.0.0.1:5000
```

### 11.1 Que se registra

Por cada modelo candidato:

- nombre del modelo;
- estrategia de busqueda;
- hiperparametros;
- numero de trials;
- metricas de validacion;
- mejor trial;
- mejores parametros;
- thresholds de validacion si aplica.

Por el modelo final:

- parametros ganadores;
- metricas de validacion;
- metricas de test;
- modelo serializado como artefacto;
- tags de seleccion;
- version en Model Registry si `registry.enabled = true`.

### 11.2 Como explicarlo al profesor

MLflow permite contestar:

- cual modelo gano;
- con que hiperparametros;
- que metrica se optimizo;
- como se comporto cada candidato;
- que run produjo el modelo guardado;
- si el resultado actual es mejor o peor que experimentos anteriores.

## 12. Prefect

Prefect orquesta el pipeline. No reemplaza el entrenamiento; coordina pasos.

El flow principal se llama:

```text
UCF Crime MLflow Prefect Pipeline
```

Tareas:

- `download_dataset`: descarga dataset;
- `build_manifest`: crea CSV de datos;
- `train_and_select_best_model`: entrena, evalua y registra.

Ventajas:

- reproducibilidad;
- logs centralizados;
- retries por tarea;
- artefactos de resumen;
- posibilidad de programar ejecuciones.

El archivo `deploy.py` sirve el flow como deployment diario a las 2:00 AM:

```python
ucf_crime_training_flow.serve(
    name="ucf-crime-training",
    cron="0 2 * * *",
)
```

## 13. Inferencia sobre imagenes

La inferencia esta en `src/ucf_crime_recognition/predict.py`.

Flujo:

1. carga config;
2. resuelve ruta del modelo;
3. carga modelo con `joblib.load`;
4. detecta el numero esperado de features;
5. si el modelo espera 2048 features, usa ResNet-50;
6. si no, usa features tradicionales;
7. llama `model.predict`;
8. si el modelo tiene `label_classes_`, activa logica multi-etiqueta;
9. calcula scores por clase;
10. aplica `label_thresholds_` si existen;
11. devuelve prediccion, confianza y scores.

Comando:

```bash
uv run ucf-predict ruta/a/imagen.png
```

O:

```bash
uv run python scripts/predict_image.py ruta/a/imagen.png
```

## 14. Analisis de video

El modelo se entrena a nivel de imagen, pero el proyecto permite analizar video mediante muestreo de frames.

### 14.1 Extraccion de frames

`_extract_sampled_frames()`:

1. abre el video con OpenCV;
2. obtiene numero total de frames;
3. obtiene FPS;
4. selecciona indices equiespaciados;
5. guarda frames temporales como imagenes PNG;
6. retorna ruta, indice y timestamp de cada frame.

### 14.2 Prediccion por frame

Cada frame se analiza con `predict_image_details()`. El resultado por frame incluye:

- timestamp;
- indice de frame;
- prediccion;
- confianza;
- scores por clase.

### 14.3 Politicas temporales

`tools/analyze_video.py` define varias politicas:

#### Peak

Si cualquier frame tiene una clase no normal con probabilidad superior a `peak_threshold`, se predice esa clase.

Utilidad:

- detectar eventos breves pero fuertes.

Riesgo:

- sensible a falsos positivos aislados.

#### Consecutive

Exige que la misma clase supere `seq_threshold` durante `seq_len` frames consecutivos.

Utilidad:

- reduce falsos positivos aislados;
- busca consistencia temporal.

Riesgo:

- puede perder eventos muy cortos.

#### Aggregate

Promedia probabilidades por clase a lo largo del video.

Utilidad:

- estable frente a ruido frame a frame.

Riesgo:

- puede diluir eventos breves.

#### Risk override

Busca evidencia repetida de clases de riesgo aunque `NormalVideos` domine. Si una clase no normal supera `risk_max` en varios frames (`risk_hits`), puede tomar la decision.

Utilidad:

- evita que eventos de riesgo queden aplastados por una mayoria de frames normales.

Riesgo:

- los parametros necesitan calibracion empirica.

## 15. Validacion de politicas temporales

`tools/validate_temporal_policies.py` permite evaluar las politicas temporales sobre un directorio etiquetado.

Entrada esperada:

```text
dataset_root/
├── Robbery/
│   ├── video1.mp4
│   └── video2.mp4
├── Fighting/
│   └── video3.mp4
└── NormalVideos/
    └── video4.mp4
```

Tambien soporta secuencias de imagenes agrupadas por prefijo.

Salida:

- `temporal_policy_predictions.csv`;
- `temporal_policy_summary.csv`;
- `classification_report_peak.txt`;
- `classification_report_consecutive.txt`;
- `classification_report_aggregate.txt`;
- `classification_report_risk_override.txt`;
- matrices de confusion por politica.

Este modulo permite comparar cual politica temporal funciona mejor para videos.

## 16. Interfaz Streamlit

La UI esta en `src/ucf_crime_recognition/ui/dashboard.py` y se abre con:

```bash
uv run streamlit run streamlit_app.py
```

La aplicacion se llama `Sentinel Review`.

### 16.1 Funciones de la interfaz

- permite subir imagen o video;
- muestra la prediccion;
- muestra confianza;
- asigna nivel operativo;
- propone una accion;
- para video, muestra timeline de frames;
- agrega scores por clase;
- muestra resumen del ultimo entrenamiento;
- permite leer el classification report.

### 16.2 Perfil de riesgo

Las clases se agrupan en:

Alta criticidad:

- `Abuse`
- `Arson`
- `Assault`
- `Explosion`
- `Fighting`
- `Robbery`
- `Shooting`

Riesgo medio:

- `Burglary`
- `RoadAccidents`
- `Shoplifting`
- `Stealing`
- `Vandalism`

Normal:

- `NormalVideos`

La UI convierte la clase detectada en una recomendacion operativa. Por ejemplo, `Robbery` sugiere restringir acceso, revisar rutas de salida y compartir evidencia.

## 17. Scripts y entrypoints

### 17.1 `scripts/download_dataset.py`

Wrapper simple que llama:

```python
ucf_crime_recognition.download.main()
```

Uso:

```bash
uv run python scripts/download_dataset.py
```

### 17.2 `scripts/build_manifest.py`

Wrapper para crear el manifest.

Uso:

```bash
uv run python scripts/build_manifest.py
```

Equivalente:

```bash
uv run ucf-manifest
```

### 17.3 `scripts/train_model.py`

Wrapper para entrenar sin Prefect.

Uso:

```bash
uv run python scripts/train_model.py
```

Equivalente:

```bash
uv run ucf-train --config configs/pipeline.toml
```

### 17.4 `scripts/predict_image.py`

Wrapper para inferencia de imagen.

Uso:

```bash
uv run python scripts/predict_image.py path/a/imagen.png
```

Equivalente:

```bash
uv run ucf-predict path/a/imagen.png
```

### 17.5 `pipeline.py` raiz

Expone el flow principal desde la raiz del proyecto.

Uso:

```bash
uv run python pipeline.py --rebuild-manifest
```

Equivalente:

```bash
uv run ucf-flow --rebuild-manifest
```

### 17.6 `streamlit_app.py`

Entry point de Streamlit:

```bash
uv run streamlit run streamlit_app.py
```

### 17.7 `deploy.py`

Crea un deployment de Prefect programado diariamente.

Uso:

```bash
uv run python deploy.py
```

## 18. Dependencias principales

Definidas en `pyproject.toml`:

- `pandas`: manejo de manifests y reportes.
- `numpy`: arreglos numericos.
- `pillow`: lectura de imagenes.
- `opencv-python-headless`: procesamiento de imagenes y video.
- `torch` y `torchvision`: extractores CNN preentrenados como ResNet-50 y VGG16.
- `scikit-learn`: modelos, metricas, pipelines y `MultiOutputClassifier`.
- `optuna`: busqueda de hiperparametros.
- `mlflow`: tracking y registry.
- `prefect`: orquestacion.
- `streamlit`: interfaz de usuario.
- `joblib`: serializacion local del modelo.
- `kagglehub`: descarga del dataset.

## 19. Estado tecnico actual

### 19.1 Fortalezas

- Arquitectura modular y entendible.
- Pipeline reproducible con configuracion centralizada.
- Integracion con Prefect y MLflow.
- Comparacion automatica de modelos candidatos.
- Uso de embeddings CNN preentrenados en vez de pixeles crudos.
- Soporte multi-etiqueta con `MultiOutputClassifier`.
- Balanceo multilabel-aware.
- Thresholds por clase optimizados en validacion.
- Interfaz funcional para imagenes y videos.
- Herramientas para evaluar politicas temporales.

### 19.2 Limitaciones

- El modelo se entrena con imagenes, no con secuencias temporales completas.
- La inferencia de video depende de reglas sobre frames, no de un modelo temporal profundo.
- `MultiOutputClassifier` aprende cada etiqueta de forma independiente.
- Si el manifest no contiene filas realmente multi-etiqueta, el modo multi-output no aprovecha todo su potencial.
- Clases raras tienen muy poco soporte, lo que reduce F1 macro.
- Los extractores preentrenados en ImageNet, como ResNet-50 o VGG16, pueden no capturar bien relaciones humanas ni eventos de vigilancia.
- El split train/test puede tener distribuciones desbalanceadas.
- La validacion temporal depende de thresholds manuales en las politicas de video.

## 20. Recomendaciones tecnicas para la siguiente asesoria

### 20.1 Preguntas para el profesor

1. Confirmar si el objetivo debe ser clasificacion por frame, por video o ambas.
2. Preguntar si se espera anotacion multi-etiqueta real en el dataset o si debe generarse a partir de segmentos de video.
3. Validar si `MultiOutputClassifier` es suficiente o conviene probar `ClassifierChain`.
4. Preguntar si el profesor prefiere mejorar recall de eventos raros aunque aumenten falsos positivos.
5. Preguntar si se debe usar un modelo temporal como LSTM, GRU, 3D CNN, TimeSformer o algun agregador sobre embeddings por frame.
6. Preguntar si `NormalVideos` debe modelarse como una etiqueta mas o como ausencia de eventos.
7. Preguntar si conviene incorporar un detector entrenado en COCO para extraer personas, objetos y relaciones espaciales.

### 20.2 Mejoras recomendadas

1. Crear un manifest a nivel de video, no solo de imagen.
2. Guardar `video_id` y `timestamp` en el manifest.
3. Separar train/test por video, no por frame, para evitar fuga de datos.
4. Evaluar `ClassifierChain` para capturar dependencias entre etiquetas.
5. Calibrar probabilidades con `CalibratedClassifierCV`, especialmente para SVM.
6. Usar metricas orientadas a riesgo, como recall macro excluyendo `NormalVideos`.
7. Evaluar PR-AUC por clase.
8. Ajustar thresholds segun costo operativo de falso positivo vs falso negativo.
9. Recolectar o sintetizar mas ejemplos de clases raras.
10. Comparar `feature_extractor = "resnet50"` contra `feature_extractor = "vgg16"` usando exactamente el mismo pipeline.
11. Probar features basadas en COCO: conteo de personas, objetos, cajas delimitadoras y relaciones persona-objeto.
12. Probar embeddings de modelos visuales mas modernos si el entorno lo permite.

## 21. Como explicar el proyecto oralmente

Una forma clara de presentarlo:

> El proyecto toma el dataset UCF Crime, construye un manifiesto reproducible de imagenes, extrae embeddings visuales con un extractor CNN preentrenado configurable, por ejemplo ResNet-50 o VGG16, y entrena varios clasificadores tradicionales sobre esos embeddings. El entrenamiento esta orquestado con Prefect y cada experimento queda registrado en MLflow. Como los videos pueden contener multiples eventos simultaneos, el sistema fue adaptado a clasificacion multi-etiqueta usando `MultiOutputClassifier`. Para combatir el desbalance, se aplican `class_weight`, rebalanceo por etiqueta y thresholds optimizados por clase. La retroalimentacion actual es evaluar si ImageNet es suficiente o si conviene incorporar features de personas/objetos con modelos entrenados en COCO. Finalmente, el modelo se puede usar en imagenes o en videos mediante muestreo de frames y politicas temporales de agregacion.

## 22. Comandos utiles

Instalar dependencias:

```bash
uv sync
```

Descargar dataset:

```bash
uv run ucf-download
```

Crear manifest:

```bash
uv run ucf-manifest
```

Ejecutar pipeline completo:

```bash
uv run ucf-flow --download --rebuild-manifest
```

Ejecutar pipeline sin descargar:

```bash
uv run ucf-flow --rebuild-manifest
```

Entrenar sin Prefect:

```bash
uv run ucf-train --config configs/pipeline.toml
```

Abrir MLflow:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Predecir imagen:

```bash
uv run ucf-predict path/a/imagen.png
```

Abrir dashboard:

```bash
uv run streamlit run streamlit_app.py
```

Validar politicas temporales:

```bash
uv run python -m ucf_crime_recognition.tools.validate_temporal_policies data/a/evaluar --output reports/temporal_policy_validation
```

## 23. Cambios recientes realizados al proyecto

En la ultima iteracion del proyecto se realizaron varios ajustes importantes con dos objetivos principales: mejorar la coherencia metodologica del pipeline y preparar una explicacion mas solida para la asesoria con el profesor.

### 23.1. Clasificacion multi-etiqueta con `MultiOutputClassifier`

Inicialmente el pipeline estaba mas orientado a clasificacion multiclase, es decir, a escoger una sola clase dominante por imagen o video. Sin embargo, en el contexto de vigilancia un mismo video puede contener mas de un evento relevante. Por ejemplo, una secuencia podria mostrar simultaneamente `Robbery` y `Fighting`, o un evento de `Burglary` con presencia de `Assault`.

Por esa razon se adapto el entrenamiento para trabajar como problema multi-etiqueta cuando `multi_output = true` en `configs/pipeline.toml`. En este modo, las etiquetas se transforman con `MultiLabelBinarizer` y cada clase se representa como una salida binaria independiente. Luego, los clasificadores base se envuelven con `MultiOutputClassifier`, permitiendo que el modelo aprenda una decision separada por cada categoria.

Este cambio es metodologicamente importante porque evita forzar al modelo a elegir una sola clase cuando el fenomeno real puede contener varias. Tambien permite aplicar umbrales independientes por clase, lo cual es util en datasets desbalanceados donde algunas categorias tienen muy pocos ejemplos.

### 23.2. Metricas orientadas a clases de riesgo

Se agregaron metricas especificas para evaluar mejor las clases anormales o de riesgo. Antes, una metrica global podia verse artificialmente favorecida por `NormalVideos`, porque esta clase suele tener muchos mas ejemplos que las demas. Esto es peligroso en un sistema de vigilancia: un modelo que predice casi todo como normal puede obtener ciertos numeros aceptables, pero fallar justamente en las clases importantes.

Por eso se incorporaron metricas como:

- `risk_f1_macro`: calcula F1 macro excluyendo `NormalVideos`.
- `risk_recall_macro`: mide que tanto se estan recuperando las clases de riesgo.
- `risk_f1_micro`: resume el desempeno global sobre las etiquetas de riesgo.

La configuracion actual selecciona modelos usando:

```toml
selection_metric = "risk_f1_macro"
```

Esta decision hace que Optuna y la seleccion final favorezcan modelos que funcionen mejor en eventos anormales, no solamente en la clase normal.

### 23.3. Aumento del volumen de datos usados en entrenamiento

Tambien se ajustaron los limites de muestreo del dataset para que el pipeline use mas informacion durante el entrenamiento, validacion y prueba. Esto ayuda especialmente porque las clases de crimen tienen menos ejemplos que `NormalVideos`, y el modelo necesita mas variabilidad visual para aprender patrones utiles.

En `configs/pipeline.toml` se aumento la cantidad maxima de muestras y se definieron limites por clase:

```toml
max_train_samples = 10000
max_validation_samples = 1200
max_test_samples = 1200
min_train_samples_per_class = 250
max_train_samples_per_class = 800
```

La intencion no es inflar artificialmente el resultado, sino permitir que el modelo vea mas ejemplos reales y reducir el sesgo hacia las clases dominantes.

### 23.4. Cache de embeddings visuales

Como la extraccion de caracteristicas con CNN preentrenadas puede ser lenta, se agrego un sistema de cache para guardar los embeddings ya calculados. Esto evita recalcular la representacion visual de la misma imagen en ejecuciones posteriores.

La configuracion queda definida asi:

```toml
embedding_cache_dir = "data/processed/embeddings"
```

Este cambio no modifica directamente la calidad predictiva del modelo, pero mejora mucho la eficiencia experimental. Permite probar configuraciones, modelos y metricas con menor tiempo de espera, especialmente cuando se trabaja con miles de imagenes.

### 23.5. Extractor visual configurable: ResNet-50, VGG16 y modelos de accion

Se modifico la extraccion de caracteristicas para que el extractor visual sea configurable. El proyecto ahora puede usar extractores 2D de imagen y extractores 3D de video desde `configs/pipeline.toml`.

Ejemplo:

```toml
feature_extractor = "r2plus1d_18"
```

Alternativas disponibles:

- `resnet50`: extractor 2D preentrenado en ImageNet.
- `vgg16`: extractor 2D preentrenado en ImageNet.
- `r3d_18`: extractor temporal preentrenado en Kinetics-400.
- `r2plus1d_18`: extractor temporal preentrenado en Kinetics-400.

La razon de este cambio fue responder a la observacion del profesor sobre si ResNet-50, entrenado con ImageNet, era la mejor opcion para un dataset de vigilancia. La conclusion tecnica es que VGG16 sirve como comparacion, pero no resuelve por si solo el problema de relaciones humanas, porque sus pesos tambien provienen de ImageNet. Es decir, tanto ResNet-50 como VGG16 aprenden principalmente objetos, texturas y escenas, no interacciones humanas complejas.

Por esa razon se agrego soporte para `r3d_18` y `r2plus1d_18`, modelos de reconocimiento de accion preentrenados en Kinetics-400. A diferencia de ResNet-50, estos modelos reciben clips de varios frames y aprenden patrones espaciotemporales. En el manifest actual, cada fila sigue siendo un frame, pero el extractor busca frames vecinos del mismo video para construir un clip de 16 frames centrado en la imagen de referencia.

```toml
feature_extractor = "r2plus1d_18"
```

Esto acerca el pipeline al problema real, porque crimen y anomalia en vigilancia no dependen solo de objetos presentes en una imagen, sino de movimiento, interaccion y contexto temporal.

### 23.6. Mejora de la inferencia en videos

La logica de prediccion se ajusto para conservar la informacion multi-etiqueta. En lugar de devolver solamente una clase, el sistema puede reportar:

- clases activas por frame,
- puntajes por clase,
- umbrales por clase,
- clase dominante,
- evidencia acumulada por video.

Esto es importante porque la interfaz y las politicas temporales necesitan distinguir entre el maximo bruto de probabilidad y la decision operativa final. Un video puede tener una probabilidad alta de `NormalVideos`, pero tambien tener evidencia puntual de una clase de riesgo. Por eso se agrego una logica que combina promedio, pico, persistencia y peso operativo de cada clase.

### 23.7. Mejora de la interfaz grafica

La interfaz de Streamlit se ajusto para reducir confusiones en la interpretacion de resultados. Antes podia aparecer una clase de riesgo en la parte superior y `NormalVideos` como probabilidad maxima en la tabla, lo que parecia contradictorio.

La interfaz ahora separa mejor estos conceptos:

- `Decision operativa`: clase seleccionada segun la politica del sistema.
- `Evidencia del evento`: score asociado a la decision operativa.
- `Mayor probabilidad`: clase con mayor probabilidad bruta.
- `Pico mayor`: valor maximo observado.
- Tabla agregada por clase con probabilidad media, probabilidad maxima, frames con senal y score operativo.
- Timeline multi-etiqueta por frame.
- Grafica de score operativo por clase.

Con esta separacion se puede explicar que el sistema no toma decisiones solo por la probabilidad maxima, sino por una combinacion mas robusta para videos: evidencia temporal, persistencia y severidad de la clase.

### 23.8. Interpretacion de los cambios

Los cambios recientes no deben presentarse como una garantia de alto desempeno, sino como una mejora metodologica del pipeline. El proyecto ahora esta mejor preparado para analizar el problema real, porque:

- reconoce que un video puede tener multiples etiquetas,
- evalua mejor las clases de riesgo,
- reduce el sesgo hacia `NormalVideos`,
- permite comparar extractores visuales,
- acelera la experimentacion mediante cache,
- muestra resultados mas interpretables en la interfaz.

El resultado observado despues de probar VGG16 fue util porque evidencio que no basta con cambiar de CNN. La mejora mas prometedora no seria simplemente pasar de ResNet-50 a VGG16, sino incorporar informacion mas relacionada con personas, objetos, acciones y temporalidad.

## 24. Conclusiones

El proyecto esta construido como una solucion MLOps local completa: no solo entrena un modelo, sino que gestiona configuracion, datos, features, experimentos, seleccion de modelo, registro, inferencia y visualizacion.

La decision de usar modelos tradicionales sobre embeddings CNN preentrenados es una estrategia razonable para un proyecto academico porque reduce el costo de entrenamiento y permite comparar modelos con Optuna. La adicion de `MultiOutputClassifier` responde al problema real de que un video puede contener multiples eventos. Sin embargo, el desempeno depende fuertemente de la calidad del manifest, el balance de clases, la existencia de etiquetas multi-etiqueta reales y la pertinencia del extractor visual frente al dominio de vigilancia.

La principal conversacion tecnica pendiente con el profesor deberia enfocarse en la unidad correcta de aprendizaje: si el problema se debe resolver por frame, por video o por secuencia temporal. Si el objetivo final es video, el pipeline actual es una buena base, pero el siguiente salto metodologico seria incorporar informacion temporal de forma mas directa.
