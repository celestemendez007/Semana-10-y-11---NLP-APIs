# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `team-03` |
| Members | Celeste Méndez, Fabiola Guardado, Eduardo Torruella, Arlette González |
| Repository URL | https://github.com/celestemendez007/Semana-10-y-11---NLP-APIs.git |
| `submission` commit hash | *Tagged at `submission`* |
| Public service URL used at the dry run | https://win-grab-florence-hearings.trycloudflare.com (Cloudflare Quick Tunnel) |
| AI tools used (which, and for what) | Antigravity AI Agent (pipeline architecture, memory optimization, contract validation, benchmark comparison) |

---

## 1. Model choice and justification

- **Base checkpoint (exact Hugging Face id):** `answerdotai/ModernBERT-base` (149,606,402 parameters)

- **Alternatives considered (and empirical benchmark):**
  Our team explored, trained, and benchmarked 4 distinct Transformer encoder checkpoints across classic and modern families on SQuAD 2.0. The empirical results on the official evaluation benchmark (11,873 questions) are summarized below:

  | Metric / Parameter | **ModernBERT-base (Winner)** | **RoBERTa-base** | **BERT-base** | **DistilBERT-base** |
  |---|:---:|:---:|:---:|:---:|
  | **Hugging Face Checkpoint** | `answerdotai/ModernBERT-base` | `FacebookAI/roberta-base` | `bert-base-uncased` | `distilbert-base-uncased` |
  | **Architecture Family** | Modern Encoder (SOTA) | Classic Encoder | Classic Encoder | Classic Encoder |
  | **Overall Exact Match (EM)** | **77.67%** | 74.93% | 67.80% | 48.50% |
  | **Overall F1-Score** | **81.04%** | 78.31% | 70.20% | 49.25% |
  | **Answerable EM (`HasAns`)** | 72.39% | **73.06%** | — | 11.36% |
  | **Answerable F1 (`HasAns`)** | 79.14% | **79.84%** | — | 12.88% |
  | **Unanswerable EM/F1 (`NoAns`)**| **82.94%** | 76.79% | 75.00% | 84.62%* |
  | **Train / Validation Samples** | 30,000 / 11,873 | 30,000 / 11,873 | Subsample / 500 | 30,000 / 11,873 |
  | **Total Parameters** | 149.6 M | 124.1 M | 108.9 M | 66.4 M |
  | **Model Size on Disk** | 572 MB | 473.3 MB | 436 MB | 253.9 MB |
  | **Mean CPU Inference Latency** | ~1,215 – 1,450 ms | 401.6 ms | 340 ms | 22.9 ms |
  | **Native Context / Positional** | Up to 8,192 (RoPE) | 512 (Absolute) | 512 (Absolute) | 512 (Absolute) |
  | **Location in Repository** | `pipeline/` and `logs/` | `experiments/roberta/` | `experiments/bert/` | `experiments/distilbert/` |

  *\*DistilBERT achieved high unanswerable accuracy solely due to an over-abstention failure mode (collapsing on answerable questions with only 11.36% EM).*

  **Why alternatives were discarded:**
  1. *DistilBERT-base-uncased:* While highly efficient on CPU (66M params, ~23 ms latency), its generalization capacity on SQuAD 2.0 fell significantly short of the 70% instructor reference anchor, suffering severe collapse on answerable questions (F1 = 49.25%).
  2. *BERT-base-uncased:* Provided a stable baseline (F1 = 70.20%, EM = 67.80%), but struggled on unanswerable distractor detection (75.0% NoAns EM) and is constrained by legacy absolute positional embeddings limited to 512 tokens.
  3. *RoBERTa-base:* Delivered strong performance (F1 = 78.31%, EM = 74.93%), but was outperformed by ModernBERT across overall F1 (+2.73%) and especially on unanswerable questions (76.79% vs 82.94%), while lacking modern architectural kernels like FlashAttention/SDPA and RoPE.

- **Trade-off:**
  ModernBERT-base represents the state-of-the-art evolution of the Transformer encoder architecture. It incorporates Rotary Position Embeddings (RoPE) extending native sequence lengths up to 8,192 tokens, GeGLU activation functions, unpadded sequence tensor manipulation, and PyTorch 2.x Scaled Dot-Product Attention (SDPA) compatibility. At 149M parameters and 572 MB on disk, it surpasses the instructor's 70% DistilBERT reference anchor by **+11.04 percentage points (81.04% F1)**, securing the full 60 points on live evaluation. For laptop CPU serving under strict 5-second per-request SLAs, ModernBERT achieves the optimal balance of answer quality, unanswerable robustness, and latency (~1.3s single-window, ~3.6s long 3-window contexts).

---

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | SQuAD 2.0 train set: stratified random subsample of **30,000 examples** (`seed=42`), preserving the original ~33.4% unanswerable distribution. |
| Validation data (what you validated on, how many examples) | Official SQuAD 2.0 validation set: **11,873 examples** (5,928 answerable, 5,945 unanswerable). |
| Epochs | 2 epochs |
| Learning rate and schedule (warmup?) | `3e-5` with linear decay and **10% warmup ratio** (~190 steps) |
| Batch size | `per_device_train_batch_size = 4` with `gradient_accumulation_steps = 8` (**effective batch size = 32**) |
| `max_seq_length` / `doc_stride` | 384 tokens / 128 tokens |
| Hardware and training time | NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM, `fp16=True`, `gradient_checkpointing=True`), **2.85 hours** (10,249s) |

- **Loss curve:**
  The training and evaluation loss trajectory is recorded and plotted in [`logs/loss.png`](file:///logs/loss.png). The training loss decreased smoothly to 1.21, and validation loss converged at 0.864 without signs of severe overfitting.

- **Validation results:**
  Evaluated using official `evaluate.load("squad_v2")` on all 11,873 validation questions. Detailed metrics are stored in [`logs/eval_results.json`](file:///logs/eval_results.json):
  - **Overall:** Exact Match (EM) = **77.67%**, F1 = **81.04%** (surpassing the 70% reference anchor by +11.04 percentage points).
  - **Answerable questions (`HasAns`):** EM = **72.39%**, F1 = **79.14%** (5,928 examples).
  - **Unanswerable questions (`NoAns`):** EM = **82.94%**, F1 = **82.94%** (5,945 examples).

---

## 3. No-answer threshold

- **Decision Mechanism:**
  For each sliding window chunk $k$, the model produces start logits $s^{(k)} \in \mathbb{R}^L$ and end logits $e^{(k)} \in \mathbb{R}^L$. The no-answer score for that window is computed from the classification token (`[CLS]` at index 0):
  $$\text{null\_score}^{(k)} = s_0^{(k)} + e_0^{(k)}$$
  The global null score for an example across all chunks is:
  $$\text{min\_null\_score} = \min_{k} \left( s_0^{(k)} + e_0^{(k)} \right)$$
  For candidate answer spans within context tokens ($seq\_id = 1$) with $0 \le \text{end} - \text{start} < \text{max\_answer\_len}$, the span score is:
  $$\text{score}(i, j) = s_i^{(k)} + e_j^{(k)}$$
  The best span is selected as $\text{best\_span\_score} = \max_{k, i, j} \text{score}(i, j)$.
  The service predicts `null` (`answer = None`, `start = None`) if and only if:
  $$\text{best\_span\_score} \le \text{min\_null\_score} + \tau$$
  where $\tau$ is the calibrated decision threshold (`NULL_THRESHOLD`).

- **Threshold Calibration:**
  A systematic grid search over $\tau \in [-5.0, +5.0]$ in increments of $0.5$ was performed on the official SQuAD 2.0 validation dataset of 11,873 examples ([`pipeline/03_error_analysis.py`](file:///pipeline/03_error_analysis.py)). The resulting F1 and EM curves across thresholds are visualized in [`logs/threshold_search.png`](file:///logs/threshold_search.png). The optimal threshold maximizing overall F1 is **$\tau = 0.0$** (F1 = **81.04%**, EM = **77.67%**).

- **Impact of Threshold Extremes:**
  - *Threshold too high ($\tau > +2.5$):* The model becomes overly conservative and abstains from answering valid questions. False Negatives surge above 40%, dropping overall F1 below 75%.
  - *Threshold too low ($\tau < -2.5$):* The model hallucinates spans on unanswerable questions. False Positives surge, degrading Exact Match on impossible questions down to 69%.

---

## 4. Serving architecture

- **Framework and Server:** FastAPI with Uvicorn ASGI server on Python 3.12. Configured as a single worker process utilizing intra-op PyTorch multi-threading (`torch.set_num_threads`) for optimal CPU latency.
- **Model Loading:** Managed asynchronously via FastAPI's `@asynccontextmanager` `lifespan` handler in `app.py`. The tokenizer and fine-tuned weights (`modernbert-base-squad2-final/`) are loaded into memory and set to `eval()` mode once upon application startup. During initialization, `/health` serves HTTP 503 `{"status": "loading"}`. Once ready, `/health` serves HTTP 200 `{"status": "ok"}`.
- **Request Flow:**
  1. Incoming `POST /predict` request is parsed and validated against the Pydantic `PredictRequest` schema (`id`, `context` 1–5000 chars, `question` 1–500 chars).
  2. Input pair is tokenized with sliding-window chunking (`max_length=384`, `stride=128`, `truncation="only_second"`), handling contexts longer than 512 tokens.
  3. Token offsets are strictly masked by sequence ID (`sequence_ids == 1`) ensuring special tokens and question tokens are never extracted as answers.
  4. Forward pass executes on CPU under `torch.no_grad()`. Start and end logits are extracted across all chunks.
  5. The global minimum null score and maximum valid span score are compared against threshold $\tau$. If the span score exceeds the null score plus $\tau$, the exact character slice is copied via `context[char_start : char_end]`, guaranteeing `context[start : start + len(answer)] == answer`. Otherwise, `answer = null` and `start = null` are returned.
  6. Confidence score is mapped to $[0, 1]$ via sigmoid of the logit difference.
- **Access Control:**
  Protected endpoints (`/info` and `/predict`) enforce authentication via an `HTTPBearer` dependency. The incoming `Authorization: Bearer <token>` is matched exactly and case-sensitively against the `QA_API_TOKEN` environment variable. Unauthorized requests return HTTP 401 `{"error": "unauthorized"}`. Validation errors return HTTP 422 `{"error": "bad_request", "detail": "..."}` per `qa_api_contract.yaml`.
- **Tunnel:**
  Exposed publicly via Cloudflare Quick Tunnel:
  ```bash
  cloudflared tunnel --url http://localhost:8000
  ```
  Backup tunnel via ngrok:
  ```bash
  ngrok http 8000
  ```
- **How to Run from a Clean Machine:**
  ```bash
  # 1. Clone repository
  git clone https://github.com/celestemendez007/Semana-10-y-11---NLP-APIs.git
  cd Semana-10-y-11---NLP-APIs

  # 2. Set up virtual environment and install dependencies
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
  pip install -r requirements.txt

  # 3. Configure environment variable and launch service
  export QA_API_TOKEN="your_assigned_token_here"
  export MODEL_DIR="modernbert-base-squad2-final"
  uvicorn app:app --host 0.0.0.0 --port 8000
  ```

---

## 5. Results on the public sample set

Evaluated against `public_sample.json`:

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|:---:|
| sample-01 | `16` or `16 attention heads` | `16` | 383 | Yes |
| sample-02 | `3.5 days` | `3.5 days` | 437 | Yes |
| sample-03 | `null` | `null` | `null` | Yes |
| sample-04 | `null` | `3` | 310 | No |
| sample-05 | `50,257` or `50,257 tokens` | `null` | `null` | No |

- **Commentary on failed items:**
  - *sample-04:* The context states that LoRA reduces GPU memory requirement "by 3 times" when fine-tuning GPT-3 175B, but the question asks about BERT-base (an impossible question). The strong lexical overlap ("reduce the GPU memory requirement" and "3") produced a logit difference of $+2.40$, exceeding the null threshold and triggering a False Positive.
  - *sample-05:* This example features a long context (~3,200 characters) split across 3 sliding windows. While the model correctly identified `"50,257"` in Window 2, the `[CLS]` token in non-target windows accumulated a high no-answer score ($\text{min\_null} = 21.71$ vs $\text{best\_span} = 16.19$), causing the model to conservatively return `null` under the global threshold ($\tau = 0.0$).

---

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | 149,606,402 (149.6 M) |
| Size on disk | ~572 MB (`model.safetensors`) |
| Mean latency per `/predict` request on laptop CPU | 1,215 ms – 1,450 ms (single window <512 tokens), 3,629 ms (long 3-window context) |
| Laptop CPU and RAM | Intel Core i5 / AMD Ryzen, 16 GB DDR4/DDR5 RAM |

---

## 7. Problems encountered and how you fixed them

1. **PyTorch 2.5 CUDA vs. Transformers 5.x Incompatibility:**
   - *Problem:* Model instantiation failed with `ImportError: AutoModelForQuestionAnswering requires PyTorch >= 2.6`.
   - *Cause:* The virtual environment installed pre-release `transformers 5.22.1` which required an unreleased PyTorch 2.6, while the stable CUDA runtime was `torch 2.5.1+cu121`.
   - *Fix:* Constrained dependency bounds in `requirements.txt` to `transformers>=4.48.0,<5` and `datasets>=3.0.0,<4`.
   - *Commit:* `7a1d3f2`

2. **HuggingFace Hub Weights Download Timeout on Windows:**
   - *Problem:* Downloading `ModernBERT-base` stalled with `ReadTimeoutError` after 10 seconds.
   - *Cause:* The default HuggingFace download timeout was short and the optional `hf-xet` daemon experienced socket deadlocks on Windows.
   - *Fix:* Disabled `hf-xet` and implemented an accelerated multi-threaded range chunk downloader (`fast_download.py`) using 8 parallel TCP connections to fetch the 598MB weights in 90 seconds.
   - *Commit:* `b4c910e`

3. **CUDA Out-Of-Memory (OOM) on 4GB Laptop GPU:**
   - *Problem:* Training with standard batch size 32 crashed immediately with `CUDA out of memory` on the laptop's RTX 3050 Ti GPU (4096 MiB VRAM).
   - *Cause:* Activation tensors and AdamW optimizer states for 150M parameters exceeded 4GB under large batch dimensions.
   - *Fix:* Configured `per_device_train_batch_size = 4` with `gradient_accumulation_steps = 8` (effective batch size 32), combined with `fp16 = True` and `gradient_checkpointing = True`, keeping peak VRAM stable at ~2.2 GiB.
   - *Commit:* `c3f8021`

4. **Windows Console Unicode Crash (`cp1252`):**
   - *Problem:* Training scripts crashed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'`.
   - *Cause:* Default Windows command line code page (`cp1252`) does not support emoji Unicode glyphs.
   - *Fix:* Injected `sys.stdout.reconfigure(encoding='utf-8')` into script entry points and standardized CLI output formatting on standard ASCII tags (`[OK]`, `[INFO]`).
   - *Commit:* `91e2b4f`

5. **Arrow Dataset Row Indexing Bottleneck in QA Postprocessing:**
   - *Problem:* SQuAD 2.0 validation span extraction loop took over 15 minutes to evaluate candidate spans across the dataset.
   - *Cause:* Calling `val_dataset[i]` on an Arrow table inside nested Python loops incurs high overhead per row access.
   - *Fix:* Extracted column arrays (`offset_mapping`, `input_ids`, `example_id`) into native Python lists upfront before iterating, reducing postprocessing time from 15 minutes to under 3 seconds.
   - *Commit:* `e804a1b`

---

## 8. Decoder investigation

Modern generative decoders (GPT-style, e.g., LLaMA-3.2, Mistral) and encoder–decoder models (e.g., T5) offer distinct capabilities and trade-offs compared to extractive encoders like ModernBERT:

1. **Non-Contiguous Answers and Multi-Span Synthesis:**
   - *Problem in Encoder:* Extractive encoders are constrained to selecting a single contiguous character slice $[i, j]$. If an answer requires synthesizing disjoint facts (e.g., "in 2018 and later in 2022"), the encoder must choose one or fail.
   - *Decoder Solution:* Decoders generate tokens autoregressively, freely combining disparate clauses or summarizing complex passages into coherent synthetic answers.
   - *Cost & Trade-offs:* Autoregressive generation scales linearly with response length ($O(L_{gen})$ steps). While ModernBERT produces answers in a single forward pass (~1.3s on CPU), a decoder requires multiple sequential passes taking 3–8 seconds on CPU, easily threatening the 5s SLA.

2. **Boolean (Yes/No) and Inferential Questions:**
   - *Problem in Encoder:* When a question requires a "Yes" or "No" judgment, an extractive model cannot emit "Yes" unless the exact word appears in the context.
   - *Decoder Solution:* Generative models reason over logical entailment and emit direct boolean answers ("Yes, because...").
   - *Cost & Trade-offs:* Evaluation becomes non-deterministic. Extractive encoders produce verifiable character offsets (`context[start : start + len(answer)] == answer`), whereas decoder phrasing varies ("Yes", "Indeed", "Correct"), requiring expensive LLM-as-a-judge pipelines.

3. **Long Contexts and KV Caching:**
   - *Problem in Encoder:* Encoders process bidirectional attention ($O(N^2)$), requiring sliding windows (`doc_stride=128`) when context exceeds `max_seq_length`.
   - *Decoder Solution:* Decoders with Key-Value (KV) caching and RoPE can ingest long prompt prefixes without sliding-window re-tokenization.
   - *Cost & Trade-offs:* High memory footprint. Storing KV caches for 3B or 7B parameter models on a 16GB RAM laptop causes severe memory bus contention, compared to ModernBERT's lightweight ~600MB memory footprint.

---

## 9. Known limitations

- **Non-Extractive Information:** If an answer is not present verbatim as a continuous substring (summaries, paraphrased insights, or implicit conclusions), the system cannot generate it and will either return `null` or an incomplete span.
- **Contract Boundary Constraints:** While ModernBERT supports long sequences via sliding windows, requests with contexts exceeding 5,000 characters or questions exceeding 500 characters are rejected with HTTP 422 according to contract boundaries.
- **Quantitative & Arithmetic Reasoning:** The model does not perform mathematical calculations (date differences, sums, percentages). It can only point to explicit numbers written in the text.
- **Multi-Hop Deductive Reasoning:** When an answer requires chaining multiple distant premises across different paragraphs, extractive span scoring tends to favor spans with high lexical overlap rather than valid multi-hop deductive conclusions.
