"""
03_error_analysis.py - Desglose de Errores para SQuAD 2.0 (DistilBERT)
Genera: logs_distilbert/error_analysis.json
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm

LOGS_DIR = "./logs_distilbert"
OUTPUT_DIR = "./distilbert_squad2_final"
EVAL_RESULTS_PATH = os.path.join(LOGS_DIR, "eval_results.json")
MAX_SEQ_LENGTH = 384

# Cargar umbral óptimo encontrado en el paso 02
best_threshold = 0.20
if os.path.exists(EVAL_RESULTS_PATH):
    with open(EVAL_RESULTS_PATH, "r") as f:
        best_threshold = json.load(f).get("optimal_threshold", 0.20)

print(f"[ERROR ANALYSIS] Analizando errores con threshold óptimo: {best_threshold}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(OUTPUT_DIR).to(device)
model.eval()

# Función nativa de inferencia
def predict_qa(question, context):
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.sequence_ids(0)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        start_logits = outputs.start_logits[0].clone()
        end_logits = outputs.end_logits[0].clone()

        null_score = (start_logits[0] + end_logits[0]).item()
        context_mask = torch.tensor([s == 1 for s in sequence_ids], device=device)
        start_logits[~context_mask] = -10000.0
        end_logits[~context_mask] = -10000.0

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

        diff = best_score - null_score
        prob = float(torch.sigmoid(torch.tensor(diff)).item())

        if null_score >= best_score or best_score == -10000.0:
            return {"answer": "", "score": round(1.0 - prob, 4)}
        else:
            s_tok, e_tok = best_span
            start_char = int(offset_mapping[s_tok][0])
            end_char = int(offset_mapping[e_tok][1])
            return {
                "answer": context[start_char:end_char].strip(),
                "score": round(prob, 4)
            }

try:
    ds = load_dataset("rajpurkar/squad_v2")
except Exception:
    ds = load_dataset("squad_v2")

val_sample = ds["validation"].shuffle(seed=42).select(range(500))

false_positives = []  # Pregunta sin respuesta, pero el modelo inventó una
false_negatives = []  # Pregunta con respuesta, pero el modelo predijo vacío
boundary_errors = []  # El modelo encontró parte de la respuesta pero no el span exacto

for ex in tqdm(val_sample, desc="Analizando errores"):
    q = ex["question"]
    ctx = ex["context"]
    true_ans = ex["answers"]["text"]
    is_impossible = (len(true_ans) == 0)

    res = predict_qa(q, ctx)
    raw_pred = res["answer"]
    score = res["score"]

    pred = "" if (not raw_pred or score < best_threshold) else raw_pred

    if is_impossible and pred != "":
        if len(false_positives) < 5:
            false_positives.append({
                "id": ex["id"],
                "question": q,
                "predicted": pred,
                "confidence": score,
                "context_snippet": ctx[:150] + "..."
            })
    elif not is_impossible and pred == "":
        if len(false_negatives) < 5:
            false_negatives.append({
                "id": ex["id"],
                "question": q,
                "expected": true_ans[0],
                "confidence": score,
                "context_snippet": ctx[:150] + "..."
            })
    elif not is_impossible and pred != "":
        if pred not in true_ans and any(w in pred for w in true_ans[0].split()):
            if len(boundary_errors) < 5:
                boundary_errors.append({
                    "id": ex["id"],
                    "question": q,
                    "expected": true_ans[0],
                    "predicted": pred,
                    "context_snippet": ctx[:150] + "..."
                })

error_report = {
    "sample_evaluated": len(val_sample),
    "threshold_used": best_threshold,
    "summary": {
        "false_positives_count": len(false_positives),
        "false_negatives_count": len(false_negatives),
        "boundary_errors_count": len(boundary_errors)
    },
    "examples": {
        "false_positives_unanswerable_hallucinations": false_positives,
        "false_negatives_missed_answers": false_negatives,
        "boundary_overlap_errors": boundary_errors
    }
}

out_file = os.path.join(LOGS_DIR, "error_analysis.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(error_report, f, indent=4)

print(f"\n[OK] Análisis completado con éxito. Archivo guardado en: {out_file}")