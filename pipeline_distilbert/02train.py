"""
02_train.py - Fine-Tuning de DistilBERT, Curva de Pérdida y Búsqueda de Umbral
Compatible con Transformers v5 (Inferencia directa con PyTorch sin pipeline deprecado)
"""
import os
import time
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

LOGS_DIR = "./logs_distilbert"
OUTPUT_DIR = "./distilbert_squad2_final"
MODEL_CHECKPOINT = "distilbert-base-uncased"
os.makedirs(LOGS_DIR, exist_ok=True)

# Hiperparámetros
TRAIN_SAMPLE_SIZE = 3000
VAL_SAMPLE_SIZE = 1000
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_avail else "cpu")
device_str = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
print("=" * 65)
print(f"[TRAIN] Dispositivo detectado: {device_str}")
print("=" * 65)

# 1. Cargar dataset
print("\n[1/5] Cargando SQuAD 2.0...")
try:
    raw_datasets = load_dataset("rajpurkar/squad_v2")
except Exception:
    raw_datasets = load_dataset("squad_v2")

shuffled_train = raw_datasets["train"].shuffle(seed=42)
shuffled_val = raw_datasets["validation"].shuffle(seed=42)

train_data = shuffled_train.select(range(min(TRAIN_SAMPLE_SIZE, len(shuffled_train))))
val_data = shuffled_val.select(range(min(VAL_SAMPLE_SIZE, len(shuffled_val))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

# 2. Función nativa de inferencia Extractive QA (100% compatible con Transformers v5)
# 2. Función nativa de inferencia Extractive QA (100% compatible con Transformers v5)
def predict_qa(question, context, model, tok, target_device):
    inputs = tok(
        question,
        context,
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.sequence_ids(0)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        # Agregamos .clone() para permitir enmascarar tokens con seguridad
        start_logits = outputs.start_logits[0].clone()
        end_logits = outputs.end_logits[0].clone()

        # Logits del token [CLS] (respuesta vacía en SQuAD 2.0)
        null_score = (start_logits[0] + end_logits[0]).item()

        # Filtrar solo tokens del contexto (sequence_id == 1)
        context_mask = torch.tensor([s == 1 for s in sequence_ids], device=target_device)
        start_logits[~context_mask] = -10000.0
        end_logits[~context_mask] = -10000.0

        # Buscar el mejor span dentro del contexto (máximo 30 tokens)
        top_starts = torch.topk(start_logits, min(10, len(start_logits))).indices.tolist()
        top_ends = torch.topk(end_logits, min(10, len(end_logits))).indices.tolist()

        best_score = -10000.0
        best_span = (0, 0)
        for s in top_starts:
            for e in top_ends:
                if s <= e and (e - s) < 30:
                    span_score = (start_logits[s] + end_logits[e]).item()
                    if span_score > best_score:
                        best_score = span_score
                        best_span = (s, e)

        # Probabilidad usando sigmoide de la diferencia frente a null
        diff = best_score - null_score
        prob = float(torch.sigmoid(torch.tensor(diff)).item())

        if null_score >= best_score or best_score == -10000.0:
            return {"answer": "", "score": round(1.0 - prob, 4), "start": None}
        else:
            s_tok, e_tok = best_span
            start_char = int(offset_mapping[s_tok][0])
            end_char = int(offset_mapping[e_tok][1])
            return {
                "answer": context[start_char:end_char].strip(),
                "score": round(prob, 4),
                "start": start_char
            }

# 3. Entrenamiento (se salta automáticamente si ya está guardado)
model_exists = os.path.exists(os.path.join(OUTPUT_DIR, "config.json"))

if not model_exists:
    print("\n[2/5] Tokenizando y procesando features con stride...")
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_SEQ_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions, end_positions = [], []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])
                seq_ids = inputs.sequence_ids(i)
                idx = 0
                while seq_ids[idx] != 1: idx += 1
                ctx_start = idx
                while seq_ids[idx] == 1: idx += 1
                ctx_end = idx - 1

                if not (offset[ctx_start][0] <= start_char and offset[ctx_end][1] >= end_char):
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    s_tok = ctx_start
                    while s_tok <= ctx_end and offset[s_tok][0] <= start_char: s_tok += 1
                    start_positions.append(s_tok - 1)
                    e_tok = ctx_end
                    while e_tok >= ctx_start and offset[e_tok][1] >= end_char: e_tok -= 1
                    end_positions.append(e_tok + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    train_dataset = train_data.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

    total_train_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * NUM_EPOCHS
    warmup_steps = max(1, int(total_train_steps * WARMUP_RATIO))

    training_args = TrainingArguments(
        output_dir="./checkpoints_distilbert",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        logging_steps=20,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DefaultDataCollator(),
    )

    print(f"\n>>> Iniciando entrenamiento ({NUM_EPOCHS} épocas)...")
    t0 = time.time()
    trainer.train()
    print(f"\n¡Entrenamiento completado en {round((time.time() - t0) / 60, 2)} minutos!")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
else:
    print(f"\n[OK] ¡Modelo ya entrenado detectado en {OUTPUT_DIR}! Saltando entrenamiento directo a evaluación...")

# 4. Inferencia en validación y búsqueda de umbral (Threshold Search)
print("\n[4/5] Evaluando en validación y calculando búsqueda de umbral óptimo...")
eval_model = AutoModelForQuestionAnswering.from_pretrained(OUTPUT_DIR).to(device)
eval_model.eval()

squad_metric = evaluate.load("squad_v2")

val_raw_preds = []
for ex in tqdm(val_data, desc="Inferencia en GPU"):
    res = predict_qa(ex["question"], ex["context"], eval_model, tokenizer, device)
    val_raw_preds.append({
        "id": ex["id"],
        "answer": res["answer"],
        "score": res["score"],
        "true_answers": ex["answers"]
    })

# Evaluar umbrales entre 0.0 y 0.8
thresholds = np.linspace(0.0, 0.8, 17)
em_scores = []
f1_scores = []
best_f1 = -1
best_thresh = 0.20

for th in thresholds:
    formatted_preds = []
    formatted_refs = []
    for item in val_raw_preds:
        pred_text = "" if (not item["answer"] or item["score"] < th) else item["answer"]
        formatted_preds.append({
            "id": item["id"],
            "prediction_text": pred_text,
            "no_answer_probability": 1.0 if not pred_text else 0.0
        })
        formatted_refs.append({
            "id": item["id"],
            "answers": item["true_answers"]
        })

    m = squad_metric.compute(predictions=formatted_preds, references=formatted_refs)
    em_scores.append(m["exact"])
    f1_scores.append(m["f1"])

    if m["f1"] > best_f1:
        best_f1 = m["f1"]
        best_thresh = float(round(th, 2))

# Guardar gráfica threshold_search.png
plt.figure(figsize=(9, 4.5))
plt.plot(thresholds, f1_scores, marker="o", label="F1 Score", color="#2ca02c")
plt.plot(thresholds, em_scores, marker="s", label="Exact Match (EM)", color="#1f77b4")
plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best Threshold ({best_thresh})")
plt.xlabel("Confidence Threshold")
plt.ylabel("Score (%)")
plt.title("Búsqueda de Umbral Óptimo para SQuAD 2.0 (DistilBERT)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
thresh_path = os.path.join(LOGS_DIR, "threshold_search.png")
plt.savefig(thresh_path, dpi=150)
plt.close()
print(f"Gráfica de búsqueda de umbral guardada en: {thresh_path}")

# Cómputo final desglosado con el mejor threshold
final_preds, final_refs = [], []
has_p, has_r, no_p, no_r = [], [], [], []

for item in val_raw_preds:
    p_text = "" if (not item["answer"] or item["score"] < best_thresh) else item["answer"]
    pred_obj = {"id": item["id"], "prediction_text": p_text, "no_answer_probability": 1.0 if not p_text else 0.0}
    ref_obj = {"id": item["id"], "answers": item["true_answers"]}

    final_preds.append(pred_obj)
    final_refs.append(ref_obj)

    if len(item["true_answers"]["text"]) > 0:
        has_p.append(pred_obj)
        has_r.append(ref_obj)
    else:
        no_p.append(pred_obj)
        no_r.append(ref_obj)

global_eval = squad_metric.compute(predictions=final_preds, references=final_refs)
has_eval = squad_metric.compute(predictions=has_p, references=has_r) if has_p else {"exact": 0, "f1": 0}
no_eval = squad_metric.compute(predictions=no_p, references=no_r) if no_p else {"exact": 0, "f1": 0}

# 5. Mediciones de tamaño y latencia en CPU pura
print("\n[5/5] Midiendo latencia en CPU y tamaño en disco...")
total_params = sum(p.numel() for p in eval_model.parameters())
disk_mb = round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(OUTPUT_DIR) for f in fns) / (1024*1024), 2)

cpu_model = AutoModelForQuestionAnswering.from_pretrained(OUTPUT_DIR).to("cpu")
cpu_model.eval()

sample_ctx = "The University of Cambridge is a public research university in Cambridge, England. Founded in 1209, Cambridge is the third-oldest surviving university in the world."
sample_q = "When was the University of Cambridge founded?"

lats = []
for _ in range(15):
    t_start = time.perf_counter()
    _ = predict_qa(sample_q, sample_ctx, cpu_model, tokenizer, "cpu")
    lats.append((time.perf_counter() - t_start) * 1000)
mean_cpu_latency = round(sum(lats) / len(lats), 2)

eval_summary = {
    "optimal_threshold": best_thresh,
    "em_general": round(global_eval["exact"], 2),
    "f1_general": round(global_eval["f1"], 2),
    "em_answerable": round(has_eval["exact"], 2),
    "f1_answerable": round(has_eval["f1"], 2),
    "em_unanswerable": round(no_eval["exact"], 2),
    "f1_unanswerable": round(no_eval["f1"], 2),
    "num_parameters": total_params,
    "disk_size_mb": disk_mb,
    "cpu_latency_ms": mean_cpu_latency
}

eval_res_path = os.path.join(LOGS_DIR, "eval_results.json")
with open(eval_res_path, "w") as f:
    json.dump(eval_summary, f, indent=4)
print(f"Métricas finales guardadas en: {eval_res_path}")

print("\n" + "=" * 65)
print(f"¡EVALUACIÓN Y MÉTRICAS FINALIZADAS!")
print(f" Umbral Óptimo     : {best_thresh}")
print(f" EM General        : {eval_summary['em_general']}%")
print(f" F1 General        : {eval_summary['f1_general']}%")
print(f" EM Answerable     : {eval_summary['em_answerable']}%")
print(f" F1 Answerable     : {eval_summary['f1_answerable']}%")
print(f" EM Unanswerable   : {eval_summary['em_unanswerable']}%")
print(f" F1 Unanswerable   : {eval_summary['f1_unanswerable']}%")
print(f" Latencia CPU      : {eval_summary['cpu_latency_ms']} ms")
print(f" Parámetros        : {eval_summary['num_parameters']:,}")
print(f" Tamaño en Disco   : {eval_summary['disk_size_mb']} MB")
print("=" * 65)