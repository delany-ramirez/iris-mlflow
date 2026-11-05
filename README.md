# Random Forest + MLflow (Iris)

Repositorio de ejemplo para **documentar experimentos de Machine Learning con MLflow y/o Databricks**.  
Entrena un modelo *Random Forest* con el dataset Iris, registra hiperparámetros, métricas y artefactos, y muestra cómo comparar runs.

---

## 1 · Requisitos

| Herramienta | Versión mínima |
|-------------|----------------|
| Python      | 3.12 |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | 0.8 |
| Git         | 2.25 |
| (Opcional) Databricks Workspace | --- |

> Las dependencias Python están en `pyproject.toml` y sus versiones exactas en `uv.lock`.

---

## 2 · Instalación

```bash
# 1. Clona el repo
git clone https://github.com/delany-ramirez/iris-mlflow.git
cd iris-mlflow

# 2. Crea el entorno virtual e instala las dependencias
uv sync
```

---

## 3 · Tracking local vs Databricks

Desde MLflow 3.7 el backend por defecto es una base de datos SQLite (`mlflow.db`). Los artefactos, como el modelo, se siguen guardando en `./mlruns`, relativo a la carpeta desde donde se ejecuta el script.

| Opción | Tracking URI | Cómo lanzar |
|--------|--------------|-------------|
| **Local (por defecto)** | `sqlite:///mlflow.db` | `uv run train_rf_mlflow.py --run_name rf_local` |
| **Ruta local personalizada** | `sqlite:///C:/_mlflow_runs/mlflow.db` | 1) `export MLFLOW_TRACKING_URI=sqlite:///C:/_mlflow_runs/mlflow.db`<br>2) Ejecuta el script |
| **Databricks** | `databricks` | Ver el bloque siguiente |

```bash
export MLFLOW_TRACKING_URI=databricks
export DATABRICKS_HOST=https://<workspace>.databricks.com
export DATABRICKS_TOKEN=<PAT>
# En Databricks el experimento debe ser una ruta absoluta del workspace
uv run train_rf_mlflow.py --experiment_name /Users/<usuario>/Iris_RF --n_estimators 150 --max_depth 6 --run_name rf_dbx
```

### 3.1 Si ya tienes una carpeta `mlruns/` de versiones anteriores

Si `./mlruns` contiene runs creados con el backend de archivos, MLflow intenta seguir usándolo y el script termina con el error *"The filesystem tracking backend (...) is in maintenance mode"*. Deja la carpeta donde está, porque los runs antiguos guardan rutas absolutas a sus modelos, e indica SQLite de forma explícita en cada sesión de terminal:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
uv run train_rf_mlflow.py --run_name rf_sqlite
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Para consultar los runs antiguos, abre la UI sobre el backend de archivos:

```bash
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri ./mlruns
```

Puede aparecer un aviso sobre un `meta.yaml` inexistente, causado por las carpetas de artefactos de los runs nuevos. Los runs antiguos se siguen mostrando con normalidad.

> **Nota:** `mlflow migrate-filestore` (probado con MLflow 3.18) falla con los runs creados por este script (`UNIQUE constraint failed: metrics...`), porque el backend de archivos guarda cada métrica dos veces. Por eso los runs antiguos no pasan a SQLite y se consultan por separado.

---

## 4 · Ejecutar experimentos

```bash
# Ejemplo rápido (tracking local)
uv run train_rf_mlflow.py --n_estimators 200 --max_depth 4 --run_name rf_200_4
uv run train_rf_mlflow.py --n_estimators 80  --max_depth 3 --run_name rf_80_3
```

*Verás en consola el `Run ID`, `Experiment ID` y las métricas registradas.*

El modelo se guarda en formato [skops](https://skops.readthedocs.io/), el predeterminado desde MLflow 3.14. El script autoriza de forma explícita el tipo `sklearn.tree._tree.Tree`, que Random Forest necesita y skops no acepta por defecto. Esa autorización solo es adecuada para modelos que entrenaste tú o de una fuente de confianza.

---

## 5 · Visualizar la UI de MLflow

### 5.1 Local

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# Luego abre http://127.0.0.1:5000
```

### 5.2 Databricks

La interfaz está integrada en *Workspace » Experiments*.

---

## 6 · Estructura del proyecto

```
.
├─ train_rf_mlflow.py          # Script principal de entrenamiento + logging
├─ pyproject.toml              # Dependencias del proyecto
├─ uv.lock                     # Versiones exactas de las dependencias
├─ delete_desktop.py           # Limpieza opcional de desktop.ini
├─ mlflow.db                   # (Autogenerado) runs, parámetros y métricas
├─ mlruns/                     # (Autogenerado) artefactos locales
└─ README.md
```

---

## 7 · Prevención de `desktop.ini`

Con SQLite, las métricas ya no se guardan como archivos, así que este problema solo afecta a carpetas `mlruns/` antiguas del backend de archivos (sección 3.1).

En Windows, el Explorador puede crear archivos `desktop.ini` dentro de `mlruns/**/metrics/`, y MLflow puede interpretarlos como métricas. Para evitarlo, no abras esas carpetas con el Explorador. Si ya existen esos archivos, `uv run delete_desktop.py` los borra de `./mlruns`; ejecútalo antes de abrir la UI.

---

## 8 · Licencia

[MIT](LICENSE)

---

## 9 · Autor

Délany Ramírez — *Universidad Tecnológica de Pereira*  
¿Sugerencias o errores? Abre un *issue* o un *pull request*.
