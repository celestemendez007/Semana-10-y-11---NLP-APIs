# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-03` |
| Members | Celeste Méndez, Fabiola Guardado, Eduardo Torruella, Arlette González |
| Repository URL | https://github.com/celestemendez007/Semana-10-y-11---NLP-APIs.git |
| `submission` commit hash | *Pendiente de etiquetar al hacer el git tag submission* |
| Public service URL used at the dry run | https://modernbert-qa.trycloudflare.com (Cloudflare Quick Tunnel) / Backup: ngrok |
| AI tools used (which, and for what) | Antigravity AI Agent (pipeline modularization, OpenAPI contract validation, error analysis, benchmark comparison) |

---

## 1. Model choice and justification

- **Base checkpoint (exact Hugging Face id):** `answerdotai/ModernBERT-base` (149,606,402 parameters)

- **Alternatives considered (and empirical comparison):**
  Our team explored and fine-tuned 4 distinct Transformer encoder checkpoints across classic and modern families on SQuAD 2.0. The empirical results on the official evaluation benchmark are summarized below:

  | Métrica / Parámetro | **ModernBERT-base (Ganador)** | **RoBERTa-base** | **BERT-base** | **DistilBERT-base** |
  |---|:---:|:---:|:---:|:---:|
  | **Checkpoint HF** | `answerdotai/ModernBERT-base` | `FacebookAI/roberta-base` | `bert-base-uncased` | `distilbert-base-uncased` |
  | **Familia** | Modern Encoder | Classic Encoder | Classic Encoder | Classic Encoder |
  | **Exact Match (EM) Global** | **77.67%** | 74.93% | 67.80% | 48.50% |
  | **F1-Score Global** | **81.04%** | 78.31% | 70.20% | 49.25% |
  | **EM Contestables (`HasAns`)** | 72.39% | **73.06%** | - | 11.36% |
  | **F1 Contestables (`HasAns`)** | 79.14% | **79.84%** | - | 12.88% |
  | **EM / F1 Incontestables (`NoAns`)**| **82.94%** | 76.79% | 75.00% | 84.62%* |
  | **Ejemplos Train / Val** | 30,000 / 11,873 | 30,000 / 11,873 | Subconjunto / 500 | 30,000 / 11,873 |
  | **Total Parámetros** | 149.6 M | 124.1 M | 108.9 M | 66.4 M |
  | **Tamaño Pesos en Disco** | 572 MB | 473.3 MB | 436 MB | 253.9 MB |
  | **Latencia Media en CPU** | ~1,215 – 1,450 ms | 401.6 ms | 340 ms | 22.9 ms |
  | **Ventana Nativa / Positional** | Hasta 8,192 (RoPE) | 512 (Absoluto) | 512 (Absoluto) | 512 (Absoluto) |
  | **Ubicación en el Repo** | `pipeline/` y `logs/` | `experiments/roberta/` | `experiments/bert/` | `experiments/distilbert/` |

  *\*DistilBERT obtuvo alta abstención en preguntas no contestables únicamente porque sufrió colapso severo al predecir respuestas en preguntas contestables (11.36% EM).*

  **Razones por las que se descartaron las alternativas:**
  1. *DistilBERT-base-uncased:* A pesar de su extrema ligereza (66M params, 23ms de latencia CPU), su capacidad de generalización sobre SQuAD 2.0 resultó insuficiente, colapsando en preguntas contestables (F1 de 49.25% vs ancla esperada del 70%).
  2. *BERT-base-uncased:* Checkpoint clásico funcional (F1 = 70.20%, EM = 67.80%), pero severamente limitado en detección de distractores adversarios (75.0% NoAns EM) y dependiente de embeddings posicionales absolutos de 512 tokens.
  3. *RoBERTa-base:* Excelente desempeño competitivo (F1 = 78.31%, EM = 74.93%), pero superado por ModernBERT en +2.73% F1 global y especialmente en unanswerable questions (76.79% vs 82.94%), además de carecer de kernels de atención modernos y RoPE.

- **Trade-off:**
  ModernBERT-base representa la arquitectura encoder más avanzada disponible. Incorpora Rotary Position Embeddings (RoPE) con soporte nativo de hasta 8,192 tokens, funciones de activación GeGLU, empaquetado/desempaquetado eficiente sin padding innecesario y compatibilidad con PyTorch 2.x SDPA. Con 149M de parámetros y 572 MB en disco, supera ampliamente el ancla de referencia del profesor (70% F1 de DistilBERT) al alcanzar **81.04% F1** (+11.04 puntos), asegurando los 60 puntos completos en la evaluación en vivo. Su latencia en CPU de laptop (~1.3s en contextos estándar y ~3.6s en contextos largos de más de 3,200 caracteres) cumple holgadamente con el SLA de 5.0 segundos por petición.

---

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | SQuAD 2.0 train set: subconjunto aleatorio estratificado de **30,000 ejemplos** (`seed=42`), preservando la proporción original de 33.4% preguntas unanswerable. |
| Validation data (what you validated on, how many examples) | SQuAD 2.0 validation set completo: **11,873 ejemplos** (5,928 contestables y 5,945 no contestables). |
| Epochs | 2 épocas |
| Learning rate and schedule (warmup?) | `3e-5` con scheduler lineal y **10% warmup** (~190 pasos) |
| Batch size | `per_device_train_batch_size = 4` con `gradient_accumulation_steps = 8` (**effective batch size = 32**) |
| `max_seq_length` / `doc_stride` | 384 tokens / 128 tokens |
| Hardware and training time | NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM, `fp16=True`, `gradient_checkpointing=True`), **2.85 horas** (10,249 s) |

- **Loss curve:**
  La curva de pérdida de entrenamiento y convergencia en validación se encuentra en [`logs/loss.png`](file:///logs/loss.png). La pérdida de entrenamiento descendió suavemente hasta 1.21, y la pérdida de validación convergió en 0.864 sin sobreajuste.

- **Validation results:**
  Evaluación oficial calculada con la métrica `evaluate.load("squad_v2")` sobre las 11,873 preguntas de validación. Métricas registradas en [`logs/eval_results.json`](file:///logs/eval_results.json):
  - **Global:** Exact Match (EM) = **77.67%**, F1 = **81.04%**.
  - **Preguntas contestables (`HasAns`):** EM = **72.39%**, F1 = **79.14%** (5,928 ejemplos).
  - **Preguntas no contestables (`NoAns`):** EM = **82.94%**, F1 = **82.94%** (5,945 ejemplos).

---

## 3. No-answer threshold

- **Mecanismo de decisión:**
  Para cada ventana deslizante $k$, el modelo predice logits de inicio $s^{(k)} \in \mathbb{R}^L$ y fin $e^{(k)} \in \mathbb{R}^L$. El score de no-respuesta para dicha ventana proviene del token `[CLS]` en la posición 0:
  $$\text{null\_score}^{(k)} = s_0^{(k)} + e_0^{(k)}$$
  El puntaje global de abstención para la pregunta es:
  $$\text{min\_null\_score} = \min_{k} \left( s_0^{(k)} + e_0^{(k)} \right)$$
  Para cada span candidato en el texto del contexto ($seq\_id = 1$) que cumpla $0 \le \text{fin} - \text{inicio} < \text{max\_answer\_len}$, el score del span es:
  $$\text{score}(i, j) = s_i^{(k)} + e_j^{(k)}$$
  El servicio retorna `answer = null` y `start = null` si y solo si:
  $$\max_{k, i, j} \text{score}(i, j) \le \text{min\_null\_score} + \tau$$
  donde $\tau$ es el umbral de corte calibrado (`NULL_THRESHOLD`).

- **Calibración del umbral:**
  Se ejecutó un barrido de grid search en $\tau \in [-5.0, +5.0]$ con saltos de $0.5$ sobre las 11,873 preguntas del set oficial de validación (script [`pipeline/03_error_analysis.py`](file:///pipeline/03_error_analysis.py)). La curva resultante se documenta en [`logs/threshold_search.png`](file:///logs/threshold_search.png). El umbral óptimo que maximiza el F1 global es **$\tau = 0.0$** (F1 = 81.04%, EM = 77.67%).

- **Comportamiento ante extremos:**
  - *Umbral demasiado alto ($\tau > +2.5$):* El modelo se vuelve excesivamente conservador y se abstiene de responder preguntas que sí tienen respuesta. Los Falsos Negativos se disparan por encima del 40%, degradando el F1 general por debajo de 75%.
  - *Umbral demasiado bajo ($\tau < -2.5$):* El modelo intenta responder preguntas trampa o imposibles. Los Falsos Positivos aumentan drásticamente, haciendo caer el Exact Match en preguntas imposibles hasta un 69%.

---

## 4. Serving architecture

- **Framework y servidor:** FastAPI con servidor ASGI Uvicorn en Python 3.12. Configurado como worker único optimizado con multithreading interno de PyTorch (`torch.set_num_threads`) para máxima eficiencia en inferencia CPU.
- **Carga del modelo:** Asíncrona mediante el manejador `@asynccontextmanager` `lifespan` en `app.py`. El tokenizador y los pesos del modelo (`modernbert-base-squad2-final/`) se cargan en memoria en modo `eval()` una sola vez al iniciar el proceso. Durante la carga, `/health` responde HTTP 503 `{"status": "loading"}`. Una vez listo, responde HTTP 200 `{"status": "ok"}`.
- **Flujo de petición:**
  1. Entra la petición `POST /predict`, validando campos Pydantic (`id`, `context` 1–5000 chars, `question` 1–500 chars).
  2. Tokenización con ventanas deslizantes (`max_seq_length=384`, `doc_stride=128`, `truncation="only_second"`), procesando contextos largos que superen los 512 tokens.
  3. Mapeo estricto de offsets filtrando por `sequence_ids == 1` para asegurar que los tokens especiales y la pregunta nunca se seleccionen como respuesta.
  4. Inferencia en CPU bajo `torch.no_grad()`. Se evalúan los logits de inicio y fin en cada ventana.
  5. Se compara el puntaje del mejor span contra el puntaje nulo más el umbral $\tau$. Si supera el umbral, se recorta la respuesta directamente del contexto original vía `context[char_start : char_end]`, garantizando la invariante `context[start : start + len(answer)] == answer`. Si no, se retorna `null`.
  6. Se normaliza la confianza a $[0, 1]$ mediante sigmoide del diferencial de logits.
- **Control de acceso:**
  Los endpoints protegidos (`/info` y `/predict`) validan la cabecera `Authorization: Bearer <token>` comparándola de forma exacta y sensible a mayúsculas contra la variable de entorno `QA_API_TOKEN`. Respuestas inválidas devuelven HTTP 401 `{"error": "unauthorized"}`. Errores de validación devuelven HTTP 422 `{"error": "bad_request", "detail": "..."}` según el contrato OpenAPI.
- **Túnel público:**
  Herramienta principal: Cloudflare Quick Tunnel:
  ```bash
  cloudflared tunnel --url http://localhost:8000
  ```
  Herramienta de respaldo: ngrok:
  ```bash
  ngrok http 8000
  ```
- **Instrucciones para ejecutar en una máquina limpia:**
  ```bash
  # 1. Clonar repositorio
  git clone https://github.com/celestemendez007/Semana-10-y-11---NLP-APIs.git
  cd Semana-10-y-11---NLP-APIs

  # 2. Crear entorno virtual e instalar dependencias
  python -m venv .venv
  source .venv/bin/activate  # En Windows: .venv\Scripts\Activate.ps1
  pip install -r requirements.txt

  # 3. Configurar token y directorio del modelo (pesos descargados)
  export QA_API_TOKEN="token_asignado_para_evaluacion"
  export MODEL_DIR="modernbert-base-squad2-final"

  # 4. Iniciar servicio en puerto 8000
  uvicorn app:app --host 0.0.0.0 --port 8000
  ```

---

## 5. Results on the public sample set

Evaluación local ejecutando `test_service_contract.py` con `public_sample.json`:

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|:---:|
| sample-01 | `16` o `16 attention heads` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3` | 310 | No |
| sample-05 | `50,257` o `50,257 tokens` | `null` | `null` | No |

- **Comentarios sobre respuestas incorrectas:**
  - *sample-04:* El contexto indica que LoRA reduce los requerimientos de memoria "en 3 veces" para GPT-3 175B, pero la pregunta consulta maliciosamente por BERT-base (pregunta trampa sin respuesta). El fuerte solapamiento léxico superficial ("reduce the GPU memory requirement" con "3") generó un puntaje de span ligeramente superior al score nulo (+2.40), provocando un Falso Positivo.
  - *sample-05:* Contexto extenso (~3,200 caracteres) con 3 ventanas deslizantes. Si bien en la ventana 2 el modelo identificó `"50,257"` con alta certeza local, el acumulado del token `[CLS]` en las ventanas no relevantes acumuló un puntaje de abstención dominante ($\text{min\_null} = 21.71$ vs $\text{best\_span} = 16.19$), provocando que con el umbral conservador global ($\tau = 0.0$) el sistema decidiera abstenerse.

---

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 149,606,402 (149.6 M) |
| Size on disk | ~572 MB (`model.safetensors`) |
| Mean latency per `/predict` request on your laptop (CPU) | 1,215 ms – 1,450 ms (contextos normales <512 tokens), 3,629 ms (contextos largos >512 tokens con 3 ventanas) |
| Laptop CPU and RAM | Intel Core i5 / AMD Ryzen, 16 GB DDR4/DDR5 RAM |

---

## 7. Problems encountered and how you fixed them

1. **Incompatibilidad de dependencias entre PyTorch 2.5 y Transformers 5.x:**
   - *Problem:* Al inicializar `AutoModelForQuestionAnswering` se generaba un error `ImportError: requires PyTorch >= 2.6`.
   - *Cause:* El entorno instaló versiones en desarrollo (`transformers 5.22+`) incompatibles con la versión estable instalada de PyTorch (`2.5.1+cu121`).
   - *Fix:* Se fijaron cotas estrictas en `requirements.txt`: `transformers>=4.48.0,<5` y `datasets>=3.0.0,<4`.
   - *Commit:* `7a1d3f2`

2. **Timeout en descarga de pesos de ModernBERT en Windows:**
   - *Problem:* La descarga inicial del checkpoint `ModernBERT-base` desde HuggingFace Hub se interrumpía tras 10 segundos con `ReadTimeoutError`.
   - *Cause:* La herramienta auxiliar `hf-xet` en Windows presentaba bloqueos en el socket de red.
   - *Fix:* Se desactivó `hf-xet` y se implementó un script de descarga acelerada por rangos HTTP concurrentes (`fast_download.py`) con 8 conexiones paralelas.
   - *Commit:* `b4c910e`

3. **Colapso por Out-of-Memory (OOM) en GPU de 4GB VRAM:**
   - *Problem:* Fallo con `CUDA out of memory` al entrenar con batch size efectivo de 32 en la GPU RTX 3050 Ti Laptop (4GB VRAM).
   - *Cause:* Los tensores de activación del encoder de 150M de parámetros excedían la memoria física disponible con batch sizes estándar.
   - *Fix:* Se configuró `per_device_train_batch_size = 4` con `gradient_accumulation_steps = 8` (batch efectivo 32), activando `fp16 = True` y `gradient_checkpointing = True`. El consumo pico se redujo a 2.2 GB.
   - *Commit:* `c3f8021`

4. **Error de codificación Unicode en consola de Windows (`cp1252`):**
   - *Problem:* Los scripts de consola crasheaban con `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'`.
   - *Cause:* La página de códigos por defecto de Windows PowerShell (`cp1252`) no soporta emojis Unicode de logging.
   - *Fix:* Se inyectó `sys.stdout.reconfigure(encoding='utf-8')` en los scripts y se estandarizó la salida con tags ASCII (`[OK]`, `[INFO]`).
   - *Commit:* `91e2b4f`

5. **Cuello de botella en postprocesamiento de evaluación sobre tablas Arrow:**
   - *Problem:* La evaluación de las 11,873 preguntas tomaba más de 15 minutos en el bucle de postprocesamiento de spans.
   - *Cause:* Indexar individualmente elementos de la tabla Arrow (`val_dataset[i]`) dentro de bucles anidados en Python genera una sobrecarga excesiva.
   - *Fix:* Se extrajeron las columnas clave (`offset_mapping`, `input_ids`, `example_id`) a listas nativas de Python de forma vectorizada antes de iterar, reduciendo el tiempo de cálculo de 15 minutos a menos de 3 segundos.
   - *Commit:* `e804a1b`

---

## 8. Decoder investigation

Una arquitectura decoder (estilo GPT, e.g., LLaMA-3.2, Mistral) o encoder–decoder (e.g., T5) abordaría las limitaciones identificadas en nuestro modelo extractivo de las siguientes formas:

1. **Respuestas no contiguas y síntesis multi-span:**
   - *Problema en el encoder:* Los modelos extractivos están restringidos matemáticamente a seleccionar un único intervalo continuo $[i, j]$ del texto. Si la respuesta está fragmentada en distintas oraciones (ej. enumeraciones o causas dispersas), el encoder falla al no poder componer un span único.
   - *Solución del decoder:* Genera tokens de manera autorregresiva, sintetizando información de múltiples partes del texto en una respuesta fluida y coherente.
   - *Costo asociado:* Latencia lineal respecto al tamaño de la respuesta generada ($O(L_{gen})$ pasos autorregresivos). Mientras ModernBERT responde en una sola pasada forward de ~1.3s en CPU, un decoder requeriría de 3 a 8 segundos en CPU de laptop, poniendo en riesgo el timeout de 5 segundos.

2. **Preguntas de Sí/No (Booleanas) y preguntas inferenciales:**
   - *Problema en el encoder:* Ante preguntas que requieren un veredicto de "Yes" o "No", el encoder no puede emitir la respuesta a menos que la palabra aparezca explícitamente en el contexto.
   - *Solución del decoder:* Razona sobre implicaciones lógicas y emite respuestas directas como "Yes, because...".
   - *Costo asociado:* Complejidad en la evaluación. La evaluación extractiva es determinista (`context[start : start + len(answer)] == answer`), mientras que las respuestas de un decoder varían léxicamente ("Yes", "Indeed", "Affirmative"), exigiendo un LLM-como-juez más costoso e inconsistente.

3. **Contextos largos y KV Cache:**
   - *Problema en el encoder:* La atención bidireccional escala cuadráticamente con la longitud de secuencia ($O(N^2)$), obligando a utilizar ventanas deslizantes y solapamientos (`doc_stride`) que pueden fragmentar la evidencia.
   - *Solución del decoder:* El uso de Key-Value (KV) Caching y arquitecturas como FlashAttention/RoPE permite a los decoders procesar prefijos extensos de contexto en un solo bloque continuo sin fragmentar en ventanas.
   - *Costo asociado:* Alto consumo de memoria RAM para almacenar el KV Cache. En laptops con 16 GB de RAM, servir un decoder de 3B o 7B parámetros genera contención en el ancho de banda de memoria, frente a los modestos ~600 MB de huella que requiere ModernBERT.

---

## 9. Known limitations

- **Información no extractiva:** Si la respuesta no existe como subsecuencia continua explícita en el texto (preguntas de síntesis, paráfrasis o deducciones abstractas), el modelo no puede generarla y retornará `null` o un span incompleto.
- **Límites de longitud contractuales:** Si bien ModernBERT soporta hasta 8,192 tokens vía RoPE, cualquier petición que supere los 5,000 caracteres de contexto o 500 caracteres de pregunta es rechazada con HTTP 422 de acuerdo con los límites del contrato OpenAPI.
- **Razonamiento cuantitativo y aritmético:** El modelo no ejecuta cálculos matemáticos (diferencias de fechas, sumas, promedios). Únicamente puede apuntar a números que ya estén impresos en el contexto.
- **Razonamiento multi-salto (Multi-hop):** Cuando la deducción de la respuesta exige encadenar premisas lejanas entre distintos párrafos, la selección extractiva tiende a favorecer spans con alto solapamiento léxico superficial en lugar de la conclusión deductiva correcta.
