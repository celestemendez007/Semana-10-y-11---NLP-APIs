# Ficha Técnica Comparativa — BERT-base (`bert-base-uncased`)

Documento técnico del experimento de fine-tuning sobre SQuAD 2.0 utilizando el checkpoint clásico `bert-base-uncased`.

---

## 1. Configuración de Entrenamiento

| Parámetro | Valor Exacto | Notas |
|---|---|---|
| **Modelo base** | `bert-base-uncased` | Checkpoint base original de Google en HuggingFace |
| **Dataset** | SQuAD 2.0 (`rajpurkar/squad_v2`) | Preguntas contestables y no contestables |
| **Muestras de entrenamiento** | Subconjunto balanceado | Fine-tuning supervisado |
| **Épocas** | 2 | Convergencia |
| **Learning Rate** | `3e-5` | Linear schedule |
| **Batch Size** | 16 | Por dispositivo |
| **`max_seq_length`** | 384 | Ventana de entrada de tokens |
| **`doc_stride`** | 128 | Solapamiento de ventanas deslizantes |
| **Hardware** | GPU | Script `train_bert.py` |

---

## 2. Métricas de Evaluación (Validación SQuAD 2.0 — 500 ejemplos)

| Métrica | Valor | Descripción |
|---|---|---|
| **Exact Match (EM) Total** | **67.80%** | Coincidencia exacta global |
| **F1-Score Total** | **70.20%** | F1 armónico de tokens (apenas alcanza el ancla de referencia del 70%) |
| **No-Answer EM (`NoAns`)** | **75.00%** | Detección de preguntas trampa (unanswerable) |
| **Umbral no-answer ($\tau$)** | Calibrado | `no_answer_threshold.json` |

---

## 3. Eficiencia y Despliegue en CPU

| Métrica | Valor |
|---|---|
| **Número de parámetros** | **108,893,186** (~108.9 M) |
| **Tamaño en disco** | **436 MB** (`model.safetensors`) |
| **Latencia media CPU** | **0.34 s** (340 ms) |

---

## 4. Evidencias en el Repositorio

* **Script de entrenamiento:** [`pipeline/train_bert.py`](./pipeline/train_bert.py)
* **Script de evaluación:** [`pipeline/evaluar_modelo_bert.py`](./pipeline/evaluar_modelo_bert.py)
* **Reporte interactivo HTML:** [`logs/resultados_modelo.html`](./logs/resultados_modelo.html)
