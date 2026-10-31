# pipeline_roberta — RoBERTa-base para SQuAD v2

Fine-tuning de [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base)
(checkpoint pretrained original, sin fine-tuning previo de QA) para **Extractive Question Answering**
en **SQuAD v2**, incluyendo preguntas no contestables.

Sigue la misma organización que `pipeline_modernBert/` + `logs_modernBert/`, pero en notebooks.

| Notebook | Qué hace | Necesita GPU | Genera en `logs_roberta/` |
|---|---|---|---|
| `00_pipeline_completo_roberta.ipynb` | **Todo de una sola vez** (01 + 02 + 03 unidos) | **Sí** | Los 9 archivos |
| `01_eda_roberta.ipynb` | Análisis exploratorio del dataset | No | `eda_squad2.png`, `eda_stats.json` |
| `02_train_roberta.ipynb` | Tokenización, fine-tuning, evaluación, latencia CPU | **Sí** | `loss.png`, `train_results.json`, `eval_results.json` |
| `03_error_analysis_roberta.ipynb` | Threshold, análisis de errores, README y comparación | No | `threshold_search.png`, `error_analysis.json`, `README.md`, `ROBERTA_COMPARISON.md` |

Los resultados están en [`../logs_roberta/`](../logs_roberta/) una vez ejecutados los notebooks.

## Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Epochs | 3 |
| Learning rate | 3e-5, scheduler lineal, warmup 10% |
| Weight decay | 0.01 |
| max_seq_length / doc_stride | 384 / 128 |
| Batch efectivo | 32 (16 × 2 en T4/L4; 32 × 1 en A100) |
| Precisión | fp16 (T4) o bf16 (A100/L4), detectada automáticamente |
| Train | SQuAD v2 train completo (configurable con `max_train_samples`) |
| Validation | SQuAD v2 validation completo (11,873 preguntas) |

La configuración que **realmente** se usó queda registrada en `logs_roberta/train_results.json`.

## Cómo ejecutar en Google Colab

**Opción rápida (todo de una sola vez):** sube `00_pipeline_completo_roberta.ipynb`, activa GPU y **Ejecutar todas**.

**Opción por etapas:**

1. Abre [colab.research.google.com](https://colab.research.google.com) → **Archivo → Subir notebook** → elige `01_eda_roberta.ipynb`.
2. **Entorno de ejecución → Ejecutar todas**. Acepta el permiso de Google Drive cuando lo pida.
3. Sube y abre `02_train_roberta.ipynb`. Antes de ejecutarlo:
   **Entorno de ejecución → Cambiar tipo de entorno → GPU**. Luego **Ejecutar todas**.
   - Si Colab se desconecta, vuelve a **Ejecutar todas**: continúa desde el último checkpoint guardado en Drive.
   - Si aparece `CUDA out of memory`, cambia `"per_device_train_batch_size": None` por `8` en la celda `CONFIG`.
   - Para repetir el experimento desde cero, borra `MyDrive/roberta_squad2/checkpoints/`.
4. Sube y abre `03_error_analysis_roberta.ipynb` (puede ser sin GPU) → **Ejecutar todas**.
   La última celda descarga `logs_roberta.zip` a tu computadora.

Todo queda en Google Drive, en `MyDrive/roberta_squad2/`:

```
roberta_squad2/
├── logs_roberta/                 ← esto va a GitHub
├── roberta-base-squad2-final/    ← modelo final (~500 MB, NO va a GitHub)
├── checkpoints/                  ← checkpoints intermedios (puedes borrarlos al terminar)
└── predictions_val.json          ← predicciones de validation (las usa el notebook 03)
```

## Usar el modelo con la API (`app.py`)

`app.py` lee el modelo y el threshold desde variables de entorno:

```bash
MODEL_DIR=roberta-base-squad2-final NULL_THRESHOLD=<valor de error_analysis.json> uvicorn app:app
```

Copia la carpeta `roberta-base-squad2-final/` desde Drive a tu computadora. El mejor threshold está en
`logs_roberta/error_analysis.json → threshold_search.best_threshold`.
