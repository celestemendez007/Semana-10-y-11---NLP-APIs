# Extractive QA as a Service — ModernBERT-base (SQuAD 2.0)

> **Ingeniería de la IA — Semanas 10 y 11: Servicios NLP expuestos por medio de API**  
> Repositorio oficial del proyecto de Question Answering Extractivo con detección de preguntas no contestables (estilo SQuAD 2.0) desplegado como microservicio web protegido por API Token.

---

## 🏆 Modelo Ganador: ModernBERT-base

Tras un riguroso proceso de fine-tuning y benchmarking empírico sobre el conjunto completo de validación de SQuAD 2.0 (11,873 preguntas), **`answerdotai/ModernBERT-base`** fue seleccionado como el **modelo ganador del equipo** con un **F1-Score global de 81.04%** y un **Exact Match de 77.67%**, superando ampliamente la cota de referencia del 70% fijada para la evaluación en vivo.

### 📊 Tabla Comparativa de los 4 Modelos Evaluados

| Métrica / Parámetro | **ModernBERT-base (Ganador)** | **RoBERTa-base** | **BERT-base** | **DistilBERT-base** |
|---|:---:|:---:|:---:|:---:|
| **Checkpoint Base HF** | `answerdotai/ModernBERT-base` | `FacebookAI/roberta-base` | `bert-base-uncased` | `distilbert-base-uncased` |
| **Familia** | Modern Encoder (SOTA) | Classic Encoder | Classic Encoder | Classic Encoder |
| **F1-Score Global** | **81.04%** | 78.31% | 70.20% | 49.25% |
| **Exact Match (EM) Global** | **77.67%** | 74.93% | 67.80% | 48.50% |
| **EM Contestables (`HasAns`)** | 72.39% | **73.06%** | — | 11.36% |
| **F1 Contestables (`HasAns`)** | 79.14% | **79.84%** | — | 12.88% |
| **EM / F1 Incontestables (`NoAns`)** | **82.94%** | 76.79% | 75.00% | 84.62%* |
| **Dataset Validación** | 11,873 (completo) | 11,873 (completo) | 500 (muestra) | 11,873 (completo) |
| **Total de Parámetros** | 149.6 M | 124.1 M | 108.9 M | 66.4 M |
| **Tamaño de Pesos en Disco** | 572 MB | 473.3 MB | 436 MB | 253.9 MB |
| **Latencia Media Inferencia CPU** | ~1,215 – 1,450 ms | 401.6 ms | 340 ms | 22.9 ms |
| **Ventana Nativa / Positional** | Hasta 8,192 (RoPE) | 512 (Absoluto) | 512 (Absoluto) | 512 (Absoluto) |
| **Evidencias en este Repo** | [`pipeline/`](./pipeline/) y [`logs/`](./logs/) | [`experiments/roberta/`](./experiments/roberta/) | [`experiments/bert/`](./experiments/bert/) | [`experiments/distilbert/`](./experiments/distilbert/) |

*\*DistilBERT obtuvo alta abstención en preguntas no contestables únicamente porque sufrió colapso severo al predecir respuestas en preguntas contestables (11.36% EM).*

---

## 📁 Estructura del Repositorio

El repositorio sigue un diseño modular y desacoplado: la raíz y las carpetas principales (`pipeline/`, `logs/`, `app.py`) pertenecen de forma exclusiva al **modelo ganador (ModernBERT)**, mientras que las investigaciones y artefactos de los otros 3 modelos se conservan organizados en `experiments/` como evidencia de justificación técnica:

```text
Semana-10-y-11---NLP-APIs/
├── app.py                      # Servicio FastAPI del modelo ganador (cumple contrato OpenAPI)
├── train_modernbert.py         # Script autónomo de fine-tuning de ModernBERT-base
├── test_service_contract.py    # Suite de pruebas automatizadas contra public_sample.json
├── qa_api_contract.yaml        # Especificación oficial de la API OpenAPI
├── public_sample.json          # Set público de 5 casos de prueba distribuido con la guía
├── requirements.txt            # Dependencias de producción y entrenamiento del modelo ganador
├── REPORT.md                   # Reporte técnico oficial completo (según REPORT_TEMPLATE.md)
├── README.md                   # Esta documentación
│
├── pipeline/                   # Pipeline reproducible de ModernBERT-base
│   ├── 01_eda.py               # Análisis exploratorio del dataset SQuAD 2.0
│   ├── 02_train.py             # Preprocesamiento, tokenización con sliding window y entrenamiento
│   └── 03_error_analysis.py    # Calibración de umbral tau y análisis de errores
│
├── logs/                       # Evidencias y artefactos del modelo ganador
│   ├── loss.png                # Curva oficial de pérdida (train vs val)
│   ├── threshold_search.png    # Gráfica de búsqueda y calibración del umbral óptimo (tau = 0.0)
│   ├── eval_results.json       # Resultados oficiales sobre las 11,873 preguntas de validación
│   ├── train_results.json      # Metadatos del entrenamiento (GPU, hiperparámetros, tiempo)
│   ├── error_analysis.json     # Desglose de errores (falsos positivos, falsos negativos, wrong span)
│   ├── eda_squad2.png          # Visualización de la distribución del dataset
│   └── eda_stats.json          # Estadísticas calculadas del dataset SQuAD 2.0
│
└── experiments/                # Evidencias y pipelines de los otros 3 modelos comparados
    ├── roberta/                # Experimento RoBERTa-base (notebooks Colab, métricas, logs)
    │   ├── pipeline/           # 00_pipeline_completo, 01_eda, 02_train, 03_error_analysis
    │   ├── logs/               # loss.png, threshold_search.png, eval_results.json
    │   ├── ROBERTA_COMPARISON.md
    │   └── requirements_roberta.txt
    │
    ├── distilbert/             # Experimento DistilBERT-base (pipeline scripts, logs)
    │   ├── pipeline/           # 01_eda.py, 02_train.py, 03_error_analysis.py, train_distilbert.py
    │   ├── logs/               # loss.png, threshold_search.png, eval_results.json
    │   ├── DISTILBERT_COMPARISON.md
    │   └── requirements_distilbert.txt
    │
    └── bert/                   # Experimento BERT-base-uncased
        ├── pipeline/           # train_bert.py, evaluar_modelo_bert.py
        ├── logs/               # resultados_modelo.html (reporte interactivo)
        ├── BERT_COMPARISON.md
        └── requirements_bert.txt
```

> **Nota sobre los pesos del modelo:** Por las reglas de la guía (Sección V y XI), los pesos entrenados del modelo (`modernbert-base-squad2-final/`, ~572 MB) **no se incluyen en el control de versiones de Git** y están debidamente excluidos en `.gitignore`.

---

## 🚀 Puesta en Marcha del Servicio (Inferencia)

### 1. Requisitos Previos e Instalación

Requiere Python 3.10 o superior (recomendado Python 3.11/3.12):

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno (Linux/macOS)
source .venv/bin/activate
# O en Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Instalar dependencias del modelo ganador
pip install -r requirements.txt
```

### 2. Configuración y Ejecución del Servidor

El servicio lee el token de autenticación obligatorio y la ruta a los pesos fine-tuned desde variables de entorno:

```bash
# Configurar variables de entorno
export QA_API_TOKEN="tu_token_secreto_aqui"
export MODEL_DIR="modernbert-base-squad2-final"
export TEAM_NAME="team-03"

# En Windows PowerShell:
# $env:QA_API_TOKEN = "tu_token_secreto_aqui"
# $env:MODEL_DIR = "modernbert-base-squad2-final"
# $env:TEAM_NAME = "team-03"

# Iniciar servidor FastAPI con Uvicorn en el puerto 8000
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Exposición Pública mediante Túnel (Requisito de Evaluación)

El evaluador se comunicará con el servicio mediante un túnel público HTTPS:

```bash
# Opción A: Cloudflare Quick Tunnel (sin registro, recomendado)
cloudflared tunnel --url http://localhost:8000

# Opción B: ngrok (de respaldo)
ngrok http 8000
```

---

## 🧪 Verificación del Contrato de la API

El repositorio incluye un script de prueba que valida los 3 endpoints contra el archivo `public_sample.json` y el contrato OpenAPI:

```bash
python test_service_contract.py
```

### Pruebas con `curl`

```bash
# 1. Health check (público) -> 200 {"status": "ok"}
curl.exe -i http://localhost:8000/health

# 2. Info sin token -> 401 {"error": "unauthorized"}
curl.exe -i http://localhost:8000/info

# 3. Info con token -> 200 {"team": "...", "model": "answerdotai/ModernBERT-base", "params": 149606402, ...}
curl.exe -i -H "Authorization: Bearer $env:QA_API_TOKEN" http://localhost:8000/info

# 4. Predict con token -> 200 {"id": "sample-01", "answer": "16", "start": 383, "score": 0.98}
curl.exe -i -X POST http://localhost:8000/predict `
  -H "Authorization: Bearer $env:QA_API_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{"id": "sample-01", "context": "BERT has 12 layers and 16 attention heads.", "question": "How many attention heads does BERT have?"}'
```

---

## 🛠️ Reproducción del Entrenamiento (ModernBERT)

Para re-entrenar ModernBERT-base desde cero utilizando el script integral o por etapas:

```bash
# Opción 1: Ejecución completa modular
python pipeline/01_eda.py
python pipeline/02_train.py
python pipeline/03_error_analysis.py

# Opción 2: Script directo
python train_modernbert.py
```
