"""
pipeline/02_train.py
---------------------
Pasos 5–9: Preprocessing → Training → Evaluation
Fine-tunes answerdotai/ModernBERT-base on SQuAD 2.0

Genera automáticamente (evidencias obligatorias):
  logs/loss.png
  logs/train_results.json
  logs/eval_results.json
  modernbert-base-squad2-final/  (modelo listo para serving)
"""

import os, json, collections, shutil
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN  (respeta los parámetros de la guía)
# ──────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "answerdotai/ModernBERT-base"
MAX_LENGTH        = 384          # max_seq_length
DOC_STRIDE        = 128          # doc_stride
BATCH_SIZE        = 8            # RTX 3050 Ti 4GB con fp16
GRAD_ACCUM        = 4            # effective batch = 32
EPOCHS            = 3
LR                = 3e-5
OUTPUT_DIR        = "modernbert-base-squad2"
FINAL_DIR         = "modernbert-base-squad2-final"
LOGS_DIR          = "logs"
SEED              = 42

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# PASO 2: Data Collection (automático vía HuggingFace)
# ──────────────────────────────────────────────────────────────
print("📦 [Paso 2] Cargando SQuAD 2.0 desde HuggingFace...")
raw_datasets = load_dataset("squad_v2")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

print(f"   Train:      {len(raw_datasets['train']):,} ejemplos")
print(f"   Validation: {len(raw_datasets['validation']):,} ejemplos")

# ──────────────────────────────────────────────────────────────
# PASO 3: Data Cleaning (mínimo — SQuAD ya está limpio)
# ──────────────────────────────────────────────────────────────
# Solo normalización de espacios en preguntas (ya incluida en preprocessing)

# ──────────────────────────────────────────────────────────────
# PASO 5: Train/Val Split — ya viene en SQuAD 2.0
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# PASO 6: Preprocessing — Tokenización con sliding window
# ──────────────────────────────────────────────────────────────
print("\n🔧 [Paso 6] Preprocesando train set...")

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    offset_mapping  = inputs.pop("offset_mapping")
    sample_map      = inputs.pop("overflow_to_sample_mapping")
    answers         = examples["answers"]
    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer     = answers[sample_idx]
        input_ids  = inputs["input_ids"][i]
        cls_index  = input_ids.index(tokenizer.cls_token_id)
        seq_ids    = inputs.sequence_ids(i)

        if len(answer["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answer["answer_start"][0]
            end_char   = start_char + len(answer["text"][0])
            idx = 0
            while seq_ids[idx] != 1: idx += 1
            context_start = idx
            while idx < len(seq_ids) and seq_ids[idx] == 1: idx += 1
            context_end = idx - 1

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char: idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char: idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"]   = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    sample_map  = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        seq_ids = inputs.sequence_ids(i)
        inputs["offset_mapping"][i] = [
            o if seq_ids[k] == 1 else None
            for k, o in enumerate(inputs["offset_mapping"][i])
        ]
    inputs["example_id"] = example_ids
    return inputs

train_dataset = raw_datasets["train"].map(
    preprocess_training_examples, batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
print(f"   Train tokenizado: {len(train_dataset):,} chunks (sliding window)")

print("🔧 [Paso 6] Preprocesando validation set...")
val_dataset_raw = raw_datasets["validation"]
val_dataset = val_dataset_raw.map(
    preprocess_validation_examples, batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
print(f"   Val tokenizado:   {len(val_dataset):,} chunks")

# ──────────────────────────────────────────────────────────────
# PASO 9: Función de evaluación EM + F1
# ──────────────────────────────────────────────────────────────
squad_metric = evaluate.load("squad_v2")

def postprocess_qa_predictions(examples, features, raw_predictions,
                                n_best=20, max_ans_len=30, null_thresh=0.0):
    start_logits, end_logits = raw_predictions
    id_to_idx = {k: i for i, k in enumerate(examples["id"])}
    feats_per_ex = collections.defaultdict(list)
    for i, feat in enumerate(features):
        feats_per_ex[id_to_idx[feat["example_id"]]].append(i)

    predictions = {}
    for ex_idx, example in enumerate(examples):
        feat_idxs      = feats_per_ex[ex_idx]
        min_null_score = None
        valid_answers  = []
        context        = example["context"]

        for fi in feat_idxs:
            sl     = start_logits[fi]
            el     = end_logits[fi]
            offs   = features[fi]["offset_mapping"]
            ids    = features[fi]["input_ids"]
            cls_i  = ids.index(tokenizer.cls_token_id)
            ns     = sl[cls_i] + el[cls_i]
            if min_null_score is None or ns < min_null_score:
                min_null_score = ns

            for si in np.argsort(sl)[-n_best:].tolist():
                for ei in np.argsort(el)[-n_best:].tolist():
                    if offs[si] is None or offs[ei] is None: continue
                    if ei < si or ei - si + 1 > max_ans_len: continue
                    valid_answers.append({
                        "score": sl[si] + el[ei],
                        "text":  context[offs[si][0]:offs[ei][1]],
                    })

        best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:1]
        if best and best[0]["score"] > min_null_score + null_thresh:
            predictions[example["id"]] = best[0]["text"]
        else:
            predictions[example["id"]] = ""
    return predictions

def compute_metrics(p):
    preds = postprocess_qa_predictions(
        val_dataset_raw, val_dataset,
        (p.predictions[0], p.predictions[1]),
    )
    formatted = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                 for k, v in preds.items()]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_dataset_raw]
    return squad_metric.compute(predictions=formatted, references=refs)

# ──────────────────────────────────────────────────────────────
# PASO 7: Model Training
# ──────────────────────────────────────────────────────────────
print(f"\n🤖 [Paso 7] Cargando modelo: {MODEL_CHECKPOINT}")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)
print(f"   Parámetros: {model.num_parameters():,}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,                       # 3e-5 (guía)
    per_device_train_batch_size=BATCH_SIZE,  # 8 (4GB VRAM)
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM, # effective batch = 32
    num_train_epochs=EPOCHS,                # 3 (guía)
    weight_decay=0.01,
    warmup_ratio=0.1,                       # 10% warmup lineal (guía)
    lr_scheduler_type="linear",
    fp16=True,                              # obligatorio RTX 3050 Ti 4GB
    gradient_checkpointing=True,            # obligatorio RTX 3050 Ti 4GB
    logging_dir=LOGS_DIR,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    seed=SEED,
    report_to="none",
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# ──────────────────────────────────────────────────────────────
# ¡ENTRENAR!
# ──────────────────────────────────────────────────────────────
print("\n🚀 Iniciando entrenamiento...")
print(f"   Epochs: {EPOCHS} | LR: {LR} | Batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
train_result = trainer.train()

# Guardar modelo final
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"\n✅ Modelo guardado en: {FINAL_DIR}/")

# ── Evidencias obligatorias ───────────────────────────────────
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

print("\n📊 [Paso 9] Evaluación final...")
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# Copiar JSONs a logs/
for fname in ["train_results.json", "eval_results.json"]:
    src = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(LOGS_DIR, fname))

if os.path.exists(os.path.join(OUTPUT_DIR, "trainer_state.json")):
    shutil.copy(
        os.path.join(OUTPUT_DIR, "trainer_state.json"),
        os.path.join(FINAL_DIR, "trainer_state.json"),
    )

# ── Loss curve (evidencia obligatoria) ───────────────────────
log_history  = trainer.state.log_history
train_steps  = [x["step"] for x in log_history if "loss" in x and "eval_loss" not in x]
train_losses = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]
eval_steps   = [x["step"] for x in log_history if "eval_loss" in x]
eval_losses  = [x["eval_loss"] for x in log_history if "eval_loss" in x]

plt.figure(figsize=(12, 5))
plt.plot(train_steps, train_losses, label="Train Loss", color="royalblue", alpha=0.8)
if eval_losses:
    plt.plot(eval_steps, eval_losses, label="Eval Loss",
             color="tomato", marker="o", linestyle="--", linewidth=2)
plt.xlabel("Step"); plt.ylabel("Loss")
plt.title(f"Training Loss — ModernBERT-base / SQuAD 2.0\n"
          f"LR={LR}, Epochs={EPOCHS}, Effective Batch={BATCH_SIZE*GRAD_ACCUM}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(LOGS_DIR, "loss.png"), dpi=150)
plt.close()
print("📈 Loss curve guardada en logs/loss.png")

# ── Resumen para el reporte ───────────────────────────────────
print("\n" + "="*55)
print("📋 RESUMEN PARA REPORT.md — Secciones 2 y 6")
print("="*55)
print(f"Base checkpoint:    {MODEL_CHECKPOINT}")
print(f"Parámetros:         {model.num_parameters():,}")
print(f"Training data:      squad_v2 train — {len(raw_datasets['train']):,} ejemplos")
print(f"Validation data:    squad_v2 val   — {len(raw_datasets['validation']):,} ejemplos")
print(f"Epochs:             {EPOCHS}")
print(f"Learning rate:      {LR} (linear, 10% warmup)")
print(f"Batch size:         {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} (effective)")
print(f"max_seq_length:     {MAX_LENGTH}")
print(f"doc_stride:         {DOC_STRIDE}")
print(f"Hardware:           NVIDIA RTX 3050 Ti 4GB (fp16)")
print(f"Runtime:            {train_result.metrics.get('train_runtime', 0)/3600:.2f} horas")
print(f"EM  (all):          {eval_metrics.get('eval_exact', eval_metrics.get('eval_exact_match', 'N/A'))}")
print(f"F1  (all):          {eval_metrics.get('eval_f1', 'N/A')}")
print("="*55)
print("\nArchivos generados:")
print(f"  logs/loss.png")
print(f"  logs/train_results.json")
print(f"  logs/eval_results.json")
print(f"  {FINAL_DIR}/  (modelo completo)")
