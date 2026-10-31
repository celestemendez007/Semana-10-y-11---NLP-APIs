"""
pipeline/01_eda.py
------------------
Paso 4: Exploratory Data Analysis — SQuAD 2.0
Genera estadísticas y guarda graficas en logs/eda_*.png
"""

import os, json, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

LOGS_DIR = "logs_modernBert"
os.makedirs(LOGS_DIR, exist_ok=True)

print("Cargando SQuAD 2.0...")
ds = load_dataset("squad_v2")
train = ds["train"]
val   = ds["validation"]

# ── Estadísticas básicas ─────────────────────────────────────
print(f"\n{'='*50}")
print(f"TRAIN split:      {len(train):,} ejemplos")
print(f"VALIDATION split: {len(val):,} ejemplos")

n_unanswerable_train = sum(1 for x in train if len(x["answers"]["text"]) == 0)
n_unanswerable_val   = sum(1 for x in val   if len(x["answers"]["text"]) == 0)
print(f"\nTrain — unanswerable: {n_unanswerable_train:,} ({n_unanswerable_train/len(train)*100:.1f}%)")
print(f"Val   — unanswerable: {n_unanswerable_val:,}   ({n_unanswerable_val/len(val)*100:.1f}%)")

# Longitud de contextos (chars)
ctx_lens_train = [len(x["context"]) for x in train]
ctx_lens_val   = [len(x["context"]) for x in val]
print(f"\nLongitud de contexto (chars) — TRAIN:")
print(f"  Media: {np.mean(ctx_lens_train):.0f} | Mediana: {np.median(ctx_lens_train):.0f} | Max: {max(ctx_lens_train)}")

# Longitud de respuestas (chars)
ans_lens = [len(a) for x in train for a in x["answers"]["text"]]
print(f"\nLongitud de respuesta (chars) — TRAIN:")
print(f"  Media: {np.mean(ans_lens):.1f} | Mediana: {np.median(ans_lens):.0f} | Max: {max(ans_lens)}")
print(f"{'='*50}\n")

# ── Gráficas ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Distribución de longitud de contexto
axes[0].hist(ctx_lens_train, bins=50, color="royalblue", alpha=0.8)
axes[0].axvline(384*4, color="red", linestyle="--", label="~384 tokens (~1500 chars)")
axes[0].set_title("Longitud de Contexto (chars)\nTrain SQuAD 2.0")
axes[0].set_xlabel("Caracteres"); axes[0].set_ylabel("Frecuencia")
axes[0].legend()

# 2. Distribución de longitud de respuesta
axes[1].hist(ans_lens, bins=40, color="tomato", alpha=0.8)
axes[1].set_title("Longitud de Respuesta (chars)\nTrain SQuAD 2.0")
axes[1].set_xlabel("Caracteres"); axes[1].set_ylabel("Frecuencia")

# 3. Pie: answerable vs unanswerable
labels = ["Answerable", "Unanswerable"]
sizes  = [len(train) - n_unanswerable_train, n_unanswerable_train]
axes[2].pie(sizes, labels=labels, autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"], startangle=90)
axes[2].set_title("Train SQuAD 2.0\nAnswerable vs Unanswerable")

plt.tight_layout()
plt.savefig(f"{LOGS_DIR}/eda_squad2.png", dpi=150)
plt.close()
print(f"✅ EDA guardada en {LOGS_DIR}/eda_squad2.png")

# Guardar stats en JSON
stats = {
    "train_total": len(train),
    "val_total": len(val),
    "train_unanswerable": n_unanswerable_train,
    "val_unanswerable": n_unanswerable_val,
    "context_len_mean": float(np.mean(ctx_lens_train)),
    "context_len_max": int(max(ctx_lens_train)),
    "answer_len_mean": float(np.mean(ans_lens)),
    "answer_len_max": int(max(ans_lens)),
}
with open(f"{LOGS_DIR}/eda_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"✅ Stats guardadas en {LOGS_DIR}/eda_stats.json")
