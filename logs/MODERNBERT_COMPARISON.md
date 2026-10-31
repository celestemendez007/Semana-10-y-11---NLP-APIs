# Ficha Comparativa de Modelo — ModernBERT-base (Fase 1)

Este documento contiene todos los requisitos de entrenamiento, métricas de evaluación, tamaño y latencia del modelo **ModernBERT-base** para comparar contra los otros 3 modelos entrenados por el equipo (ej. DistilBERT, BERT-base, RoBERTa-base).

---

## 1. Parámetros de Entrenamiento (Training Setup)

| Requisito | Valor Exacto para ModernBERT | Notas del Experimento |
|---|---|---|
| **Modelo base** | `answerdotai/ModernBERT-base` | Checkpoint base oficial de HuggingFace (sin fine-tuning previo) |
| **Training data** | `squad_v2`, split `train` — **30,000 ejemplos** | Subconjunto balanceado aleatorio con seed 42 (33.4% preguntas unanswerable) |
| **Validation data** | `squad_v2`, split `validation` — **11,873 ejemplos** | Dataset completo de validación oficial |
| **Epochs** | **2** | Convergencia óptima |
| **Learning rate** | **`3e-5`** | Schedule `linear` con **warmup del 10%** del total de pasos (~190 pasos) |
| **Batch size** | `per_device_train_batch_size = 4` con `gradient_accumulation_steps = 8` | **Effective batch size = 32** (exacto al requerido por la guía) |
| **`max_seq_length`** | **384** | Longitud de entrada por ventana |
| **`doc_stride`** | **128** | Solapamiento entre ventanas deslizantes para contextos largos |
| **Optimizaciones VRAM** | `fp16 = True`, `gradient_checkpointing = True` | Permite entrenar con 4GB de VRAM sin Out-Of-Memory (OOM) |
| **Hardware** | `NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB)` | **2.85 horas** de entrenamiento (10,249 segundos) |
| **Loss curve** | [`logs/modernbert/loss.png`](./loss.png) o [`logs/loss.png`](../loss.png) | Curva de pérdida por pasos y convergencia en validación |

---

## 2. Métricas de Evaluación (Validation Set — 11,873 ejemplos)

Evaluado utilizando la métrica oficial `evaluate.load("squad_v2")`:

| Métrica | Valor Obtenido | Descripción |
|---|---|---|
| **EM general (Exact Match)** | **77.67%** | Porcentaje de respuestas con coincidencia exacta sobre todo el validation set (11,873 preguntas) |
| **F1 general** | **81.04%** | F1 score a nivel de tokens sobre todo el validation set (supera el ancla del 70% de DistilBERT por +11.04%) |
| **EM answerable (`HasAns`)** | **72.39%** | Exact Match exclusivamente en las 5,928 preguntas que SÍ tienen respuesta en el texto |
| **F1 answerable (`HasAns`)** | **79.14%** | F1 de tokens exclusivamente en las 5,928 preguntas que SÍ tienen respuesta |
| **EM unanswerable (`NoAns`)** | **82.94%** | Exact Match en las 5,945 preguntas sin respuesta (acierta cuando predice string vacío `""`) |
| **F1 unanswerable (`NoAns`)** | **82.94%** | F1 en las 5,945 preguntas sin respuesta (acierta cuando predice string vacío `""`) |
| **Threshold calibrado ($\tau$)** | **0.0** | Umbral óptimo de no-respuesta (`score(span) > score([CLS]) + tau`) obtenido por grid search |

---

## 3. Tamaño y Latencia (Inferencia en CPU)

| Métrica | Valor Exacto | Método de Medición |
|---|---|---|
| **Número de parámetros** | **149,606,402** (149.6M) | Obtenido con `model.num_parameters()` |
| **Tamaño en disco del modelo** | **572 MB** (598,439,784 bytes) | Archivo `model.safetensors` |
| **Latencia media en CPU (contexto normal)** | **1,215 ms – 1,450 ms** (~1.3s) | Medido con llamadas a `/predict` en CPU sin aceleración GPU |
| **Latencia media en CPU (contexto largo >512 tokens)** | **3,629 ms** (~3.6s) | Contexto de 3,200 caracteres con 3 ventanas deslizantes (cumple SLA < 5.0s) |
| **Especificaciones del equipo** | AMD / Intel Core i5, 16 GB RAM, GPU RTX 3050 Ti 4GB | Laptop donde se ejecutó el servicio |

---

## 4. Archivos de Evidencia en este Repositorio

* **Script de entrenamiento (Fase 1):** [`train_modernbert.py`](../../train_modernbert.py) (también disponible en [`pipeline/02_train.py`](../../pipeline/02_train.py)).
* **Curva de pérdida:** [`logs/loss.png`](./loss.png).
* **Resultados brutos de entrenamiento:** [`logs/train_results.json`](./train_results.json).
* **Resultados brutos de evaluación oficial:** [`logs/eval_results.json`](./eval_results.json).
* **Curva de búsqueda de threshold:** [`logs/threshold_search.png`](./threshold_search.png).
* **Análisis de errores:** [`logs/error_analysis.json`](./error_analysis.json).

