"""
train_modernbert.py
-------------------
Fine-tunes answerdotai/ModernBERT-base on SQuAD 2.0 for Extractive QA.

Hardware objetivo: NVIDIA RTX 3050 Ti Laptop (4 GB VRAM)
Genera automáticamente:
  - logs/loss.png              (evidencia obligatoria)
  - logs/train_results.json    (evidencia obligatoria)
  - logs/eval_results.json     (evidencia obligatoria)
  - modernbert-base-squad2-final/  (modelo listo para serving)

Uso:
  pip install -r requirements.txt
  python train_modernbert.py
"""

import os
import json
import collections
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
# CONFIGURACIÓN — ajusta aquí si tu VRAM no alcanza
# ──────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "answerdotai/ModernBERT-base"
MAX_LENGTH        = 384    # max_seq_length
DOC_STRIDE        = 128    # doc_stride
BATCH_SIZE        = 8      # por device (RTX 3050 Ti 4GB con fp16)
GRAD_ACCUM        = 4      # effective batch = 8 × 4 = 32
EPOCHS            = 3
LR                = 3e-5
OUTPUT_DIR        = "modernbert-base-squad2"
FINAL_DIR         = "modernbert-base-squad2-final"
LOGS_DIR          = "logs"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. Dataset — SQuAD 2.0
# ──────────────────────────────────────────────────────────────
print("Cargando SQuAD 2.0...")
raw_datasets = load_dataset("squad_v2")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# ──────────────────────────────────────────────────────────────
# 2. Preprocesamiento — TRAIN
# ──────────────────────────────────────────────────────────────
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping  = inputs.pop("offset_mapping")
    sample_map      = inputs.pop("overflow_to_sample_mapping")
    answers         = examples["answers"]
    start_positions = []
    end_positions   = []

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
            while seq_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(seq_ids) and seq_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"]   = end_positions
    return inputs

# ──────────────────────────────────────────────────────────────
# 3. Preprocesamiento — VALIDATION
# ──────────────────────────────────────────────────────────────
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
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

print("Preprocesando train...")
train_dataset = raw_datasets["train"].map(
    preprocess_training_examples, batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

print("Preprocesando validation...")
val_dataset_raw = raw_datasets["validation"]
val_dataset = val_dataset_raw.map(
    preprocess_validation_examples, batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

# ──────────────────────────────────────────────────────────────
# 4. Evaluación EM + F1 separado por tipo (SQuAD 2.0)
# ──────────────────────────────────────────────────────────────
squad_metric = evaluate.load("squad_v2")

def postprocess_qa_predictions(examples, features, raw_predictions,
                                n_best_size=20, max_answer_length=30,
                                null_score_diff_threshold=0.0):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        features_per_example[example_id_to_index[feat["example_id"]]].append(i)

    predictions = {}
    for example_idx, example in enumerate(examples):
        feat_indices   = features_per_example[example_idx]
        min_null_score = None
        valid_answers  = []
        context        = example["context"]

        for feat_idx in feat_indices:
            start_logits    = all_start_logits[feat_idx]
            end_logits      = all_end_logits[feat_idx]
            offset_mapping  = features[feat_idx]["offset_mapping"]
            input_ids       = features[feat_idx]["input_ids"]
            cls_index       = input_ids.index(tokenizer.cls_token_id)
            null_score      = start_logits[cls_index] + end_logits[cls_index]

            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score

            start_indexes = np.argsort(start_logits)[-n_best_size:].tolist()
            end_indexes   = np.argsort(end_logits)[-n_best_size:].tolist()
            for si in start_indexes:
                for ei in end_indexes:
                    if (offset_mapping[si] is None or offset_mapping[ei] is None
                            or ei < si or ei - si + 1 > max_answer_length):
                        continue
                    valid_answers.append({
                        "score": start_logits[si] + end_logits[ei],
                        "text":  context[offset_mapping[si][0]:offset_mapping[ei][1]],
                    })

        best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:1]
        if best and best[0]["score"] > min_null_score + null_score_diff_threshold:
            predictions[example["id"]] = best[0]["text"]
        else:
            predictions[example["id"]] = ""

    return predictions

def compute_metrics(p):
    predictions = postprocess_qa_predictions(
        val_dataset_raw, val_dataset,
        (p.predictions[0], p.predictions[1]),
    )
    formatted = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in predictions.items()
    ]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_dataset_raw]
    return squad_metric.compute(predictions=formatted, references=refs)

# ──────────────────────────────────────────────────────────────
# 5. Modelo + TrainingArguments
# ──────────────────────────────────────────────────────────────
print(f"Cargando modelo: {MODEL_CHECKPOINT}")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

print(f"Parámetros del modelo: {model.num_parameters():,}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    fp16=True,                       # Obligatorio con 4 GB VRAM
    gradient_checkpointing=True,     # Obligatorio con 4 GB VRAM
    logging_dir=LOGS_DIR,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    seed=42,
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
# 6. Entrenamiento
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Iniciando entrenamiento...")
    train_result = trainer.train()

    # Guardar modelo final
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"\n✅ Modelo guardado en: {FINAL_DIR}/")

    # Guardar métricas de train — evidencia obligatoria
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # Copiar a logs/
    import shutil
    if os.path.exists(f"{OUTPUT_DIR}/train_results.json"):
        shutil.copy(f"{OUTPUT_DIR}/train_results.json", f"{LOGS_DIR}/train_results.json")

    # Evaluación final
    print("\n📊 Evaluando en validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if os.path.exists(f"{OUTPUT_DIR}/eval_results.json"):
        shutil.copy(f"{OUTPUT_DIR}/eval_results.json", f"{LOGS_DIR}/eval_results.json")

    # Guardar trainer_state.json en el modelo final
    if os.path.exists(f"{OUTPUT_DIR}/trainer_state.json"):
        shutil.copy(f"{OUTPUT_DIR}/trainer_state.json", f"{FINAL_DIR}/trainer_state.json")

    # ──────────────────────────────────────────────────────────
    # 7. Generar loss.png — evidencia obligatoria
    # ──────────────────────────────────────────────────────────
    log_history = trainer.state.log_history

    train_steps  = [x["step"] for x in log_history if "loss" in x and "eval_loss" not in x]
    train_losses = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]
    eval_steps   = [x["step"] for x in log_history if "eval_loss" in x]
    eval_losses  = [x["eval_loss"] for x in log_history if "eval_loss" in x]

    plt.figure(figsize=(12, 5))
    plt.plot(train_steps, train_losses, label="Train Loss", color="royalblue", alpha=0.8)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Eval Loss",
                 color="tomato", marker="o", linestyle="--", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss — {MODEL_CHECKPOINT} / SQuAD 2.0\n"
              f"LR={LR}, Epochs={EPOCHS}, Batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(LOGS_DIR, "loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"📈 Loss curve guardada en: {loss_path}")

    # ──────────────────────────────────────────────────────────
    # 8. Resumen final para el REPORT.md
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("📋 RESUMEN PARA EL REPORTE (Sección 2 y 6)")
    print("="*50)
    print(f"Modelo base:        {MODEL_CHECKPOINT}")
    print(f"Parámetros:         {model.num_parameters():,}")
    print(f"max_seq_length:     {MAX_LENGTH}")
    print(f"doc_stride:         {DOC_STRIDE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Learning rate:      {LR}")
    print(f"Batch size:         {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} (effective)")
    print(f"EM (all):           {eval_metrics.get('eval_exact', eval_metrics.get('eval_exact_match', 'N/A'))}")
    print(f"F1 (all):           {eval_metrics.get('eval_f1', 'N/A')}")
    print(f"Runtime:            {train_result.metrics.get('train_runtime', 0)/3600:.2f} horas")
    print("="*50)
    print(f"\nEvidencias generadas en: {LOGS_DIR}/")
    print(f"Modelo final en:         {FINAL_DIR}/")
