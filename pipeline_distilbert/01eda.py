"""
01_eda.py - Análisis Exploratorio de Datos (EDA) para SQuAD 2.0
Genera: logs_distilbert/eda_stats.json y logs_distilbert/eda_squad2.png
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

LOGS_DIR = "./logs_distilbert"
os.makedirs(LOGS_DIR, exist_ok=True)

print("[EDA] Cargando dataset SQuAD 2.0...")
try:
    ds = load_dataset("rajpurkar/squad_v2")
except Exception:
    ds = load_dataset("squad_v2")

train_data = ds["train"]
val_data = ds["validation"]

def analyze_split(split):
    total = len(split)
    has_answer = sum(1 for x in split if len(x["answers"]["text"]) > 0)
    no_answer = total - has_answer
    context_lens = [len(x["context"].split()) for x in split]
    question_lens = [len(x["question"].split()) for x in split]
    return {
        "total_examples": total,
        "answerable": has_answer,
        "unanswerable": no_answer,
        "answerable_ratio": round(has_answer / total, 4),
        "unanswerable_ratio": round(no_answer / total, 4),
        "avg_context_words": round(sum(context_lens) / total, 2),
        "max_context_words": max(context_lens),
        "avg_question_words": round(sum(question_lens) / total, 2),
        "max_question_words": max(question_lens),
    }

print("[EDA] Calculando estadísticas de Train y Validation...")
train_stats = analyze_split(train_data)
val_stats = analyze_split(val_data)

stats = {
    "dataset": "squad_v2",
    "train": train_stats,
    "validation": val_stats
}

# Guardar estadísticas en JSON
stats_path = os.path.join(LOGS_DIR, "eda_stats.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=4)
print(f"[EDA] Estadísticas guardadas con éxito en: {stats_path}")

# Generar gráfica de barras e histograma
print("[EDA] Generando gráfica eda_squad2.png...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Gráfico 1: Balance de clases
splits = ["Train", "Validation"]
ans = [train_stats["answerable"], val_stats["answerable"]]
no_ans = [train_stats["unanswerable"], val_stats["unanswerable"]]

x = range(len(splits))
axes[0].bar(x, ans, width=0.4, label="Answerable", color="#2ca02c")
axes[0].bar([p + 0.4 for p in x], no_ans, width=0.4, label="Unanswerable", color="#d62728")
axes[0].set_xticks([p + 0.2 for p in x])
axes[0].set_xticklabels(splits)
axes[0].set_ylabel("Número de Ejemplos")
axes[0].set_title("Distribución de Clases en SQuAD 2.0")
axes[0].legend()
axes[0].grid(axis="y", linestyle="--", alpha=0.6)

# Gráfico 2: Longitud de contextos
train_ctx_lens = [len(x["context"].split()) for x in train_data.select(range(min(5000, len(train_data))))]
sns.histplot(train_ctx_lens, bins=30, ax=axes[1], color="#1f77b4", kde=True)
axes[1].set_xlabel("Palabras por contexto")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Longitud de Contextos (Muestra Train)")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
eda_png_path = os.path.join(LOGS_DIR, "eda_squad2.png")
plt.savefig(eda_png_path, dpi=150)
plt.close()
print(f"[EDA] Gráfica guardada con éxito en: {eda_png_path}")