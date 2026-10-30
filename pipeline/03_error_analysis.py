"""
pipeline/03_error_analysis.py
------------------------------
Paso 10: Error Analysis — analiza predicciones incorrectas del modelo
Corre DESPUÉS de 02_train.py, usa el modelo guardado en modernbert-base-squad2-final/

Genera:
  logs/error_analysis.json   — ejemplos donde el modelo falló
  logs/threshold_search.png  — curva F1 vs threshold (Paso 11)
"""

import os, json, collections
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import evaluate
import torch

os.makedirs("logs", exist_ok=True)

MODEL_DIR     = "modernbert-base-squad2-final"
MAX_LENGTH    = 384
DOC_STRIDE    = 128
N_BEST        = 20
MAX_ANS_LEN   = 30

print("Cargando modelo y dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
model.eval()

raw_datasets    = load_dataset("squad_v2")
val_dataset_raw = raw_datasets["validation"]

# Tokenizar validation
def preprocess_validation(examples):
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

val_dataset = val_dataset_raw.map(
    preprocess_validation, batched=True,
    remove_columns=val_dataset_raw.column_names,
)

# Inferencia batch
from torch.utils.data import DataLoader
val_dataset.set_format("torch")
loader = DataLoader(val_dataset.remove_columns(["example_id", "offset_mapping"]),
                    batch_size=16)

all_start, all_end = [], []
with torch.no_grad():
    for batch in loader:
        out = model(**{k: v for k, v in batch.items()
                       if k in ("input_ids", "attention_mask", "token_type_ids") and v is not None})
        all_start.append(out.start_logits.numpy())
        all_end.append(out.end_logits.numpy())

all_start = np.concatenate(all_start)
all_end   = np.concatenate(all_end)

squad_metric = evaluate.load("squad_v2")

# ── Paso 11: Búsqueda del threshold óptimo ───────────────────
print("\n🔍 [Paso 11] Buscando threshold óptimo...")

def get_predictions(null_thresh):
    id_to_idx = {k: i for i, k in enumerate(val_dataset_raw["id"])}
    feats_per_ex = collections.defaultdict(list)
    for i, feat in enumerate(val_dataset):
        feats_per_ex[id_to_idx[feat["example_id"]]].append(i)
    predictions = {}
    null_scores  = {}
    for ex_idx, example in enumerate(val_dataset_raw):
        feat_idxs      = feats_per_ex[ex_idx]
        min_null_score = None
        valid_answers  = []
        context        = example["context"]
        for fi in feat_idxs:
            sl, el = all_start[fi], all_end[fi]
            offs   = val_dataset[fi]["offset_mapping"]
            ids    = val_dataset[fi]["input_ids"].tolist()
            cls_i  = ids.index(tokenizer.cls_token_id)
            ns     = sl[cls_i] + el[cls_i]
            if min_null_score is None or ns < min_null_score:
                min_null_score = ns
            for si in np.argsort(sl)[-N_BEST:].tolist():
                for ei in np.argsort(el)[-N_BEST:].tolist():
                    if offs[si] is None or offs[ei] is None: continue
                    if ei < si or ei - si + 1 > MAX_ANS_LEN: continue
                    valid_answers.append({
                        "score": sl[si] + el[ei],
                        "text":  context[offs[si][0]:offs[ei][1]],
                    })
        best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:1]
        if best and best[0]["score"] > min_null_score + null_thresh:
            predictions[example["id"]] = best[0]["text"]
        else:
            predictions[example["id"]] = ""
        null_scores[example["id"]] = float(min_null_score or 0)
    return predictions, null_scores

thresholds = np.arange(-5.0, 5.5, 0.5).tolist()
f1_scores  = []
em_scores  = []

for thresh in thresholds:
    preds, _ = get_predictions(thresh)
    formatted = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                 for k, v in preds.items()]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_dataset_raw]
    res  = squad_metric.compute(predictions=formatted, references=refs)
    f1_scores.append(res["f1"])
    em_scores.append(res["exact"])
    print(f"  threshold={thresh:+.1f} → F1={res['f1']:.2f}  EM={res['exact']:.2f}")

best_idx   = int(np.argmax(f1_scores))
best_thresh = thresholds[best_idx]
best_f1     = f1_scores[best_idx]
print(f"\n✅ Mejor threshold: {best_thresh:+.1f} → F1={best_f1:.2f}")

# Guardar curva
plt.figure(figsize=(10, 4))
plt.plot(thresholds, f1_scores, label="F1", color="royalblue", marker="o")
plt.plot(thresholds, em_scores, label="EM", color="tomato",   marker="s")
plt.axvline(best_thresh, color="green", linestyle="--", label=f"Mejor threshold={best_thresh}")
plt.xlabel("Null Score Threshold"); plt.ylabel("Score")
plt.title("F1 y EM vs No-Answer Threshold\nModernBERT-base / SQuAD 2.0 Validation")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("logs/threshold_search.png", dpi=150)
plt.close()
print("📈 Curva guardada en logs/threshold_search.png")

# ── Paso 10: Error Analysis con el mejor threshold ────────────
print(f"\n🔍 [Paso 10] Error analysis con threshold={best_thresh}...")
best_preds, _ = get_predictions(best_thresh)

errors = []
for ex in val_dataset_raw:
    pred = best_preds.get(ex["id"], "")
    gold_answers = ex["answers"]["text"]
    is_impossible = len(gold_answers) == 0

    # Determinar si es correcto
    if is_impossible:
        correct = (pred == "")
        error_type = None if correct else "false_positive"
    else:
        correct = any(pred.strip().lower() == g.strip().lower() for g in gold_answers)
        error_type = None if correct else ("false_negative" if pred == "" else "wrong_span")

    if not correct:
        errors.append({
            "id":          ex["id"],
            "question":    ex["question"],
            "context":     ex["context"][:300] + "...",
            "gold":        gold_answers,
            "prediction":  pred,
            "is_impossible": is_impossible,
            "error_type":  error_type,
        })

# Estadísticas de errores
error_types = {}
for e in errors:
    t = e["error_type"]
    error_types[t] = error_types.get(t, 0) + 1

print(f"   Total errores: {len(errors):,} / {len(val_dataset_raw):,}")
for t, c in error_types.items():
    print(f"   {t}: {c:,} ({c/len(errors)*100:.1f}%)")

with open("logs/error_analysis.json", "w", encoding="utf-8") as f:
    json.dump({
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "total_errors": len(errors),
        "error_breakdown": error_types,
        "samples": errors[:50],  # primeros 50 errores
    }, f, indent=2, ensure_ascii=False)

print("✅ Error analysis guardado en logs/error_analysis.json")
print(f"\n💾 Guarda este threshold en app.py: NULL_THRESHOLD = {best_thresh}")
