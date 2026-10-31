#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extractive QA as a Service - Fine-tuning de bert-base-uncased en SQuAD 2.0
==========================================================================

Uso (ejemplos):
    # Prueba rápida de humo (subconjunto pequeño, 1 época)
    python finetune_squad2.py --max_train_samples 2000 --max_eval_samples 500 --epochs 1

    # Entrenamiento completo (GPU recomendada)
    python finetune_squad2.py --epochs 2 --batch_size 16 --lr 3e-5

Salidas:
    logs/loss.png                -> curva de pérdida (train vs. validación)
    logs/log_history.json        -> historial crudo de trainer.state.log_history
    modelo_guardado/             -> modelo + tokenizador (cargables en CPU)
    modelo_guardado/no_answer_threshold.json -> umbral calibrado + metadatos para /info

Reglas del proyecto que este script respeta:
    * Se parte del checkpoint BASE `bert-base-uncased` (NO uno ya afinado en SQuAD).
    * Nada de APIs comerciales: todo corre localmente.
"""

import argparse
import collections
import inspect
import json
import os
import re
import string

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla: funciona en servidores/Colab/SSH
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"   # checkpoint BASE (obligatorio por las reglas)
OUTPUT_MODEL_DIR = "./modelo_guardado"
LOGS_DIR = "logs"


# ==========================================================================
# 1. PREPROCESAMIENTO
# ==========================================================================
def prepare_train_features(examples, tokenizer, max_length=384, stride=128):
    """
    Tokeniza pares (pregunta, contexto) con VENTANAS DESLIZANTES y calcula las
    etiquetas `start_positions` / `end_positions` para cada ventana.

    - Contextos largos se parten en varias ventanas de `max_length` tokens con
      un solape de `stride` tokens (return_overflowing_tokens=True).
      Un ejemplo original puede generar N >= 1 "features".
    - Pregunta CON respuesta cuya respuesta cae dentro de la ventana:
          start/end = índices de los tokens que cubren la respuesta.
    - Pregunta SIN respuesta (SQuAD 2.0) O ventana que no contiene la respuesta:
          start = end = 0  (índice del token [CLS]).
      Esto es lo que enseña al modelo a "decir no sé" y es la base del umbral.
    """
    # BERT rellena a la derecha: orden (pregunta, contexto), truncamos el contexto.
    pad_on_right = tokenizer.padding_side == "right"
    context_seq_id = 1 if pad_on_right else 0

    tokenized = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=stride,                      # solape entre ventanas
        return_overflowing_tokens=True,     # genera varias ventanas por ejemplo
        return_offsets_mapping=True,        # token -> (char_ini, char_fin)
        padding="max_length",               # longitud fija -> batches uniformes
    )

    # Mapea cada ventana a su ejemplo original y quita columnas auxiliares.
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)  # = 0 en BERT

        # sequence_ids: 0 = pregunta, 1 = contexto, None = tokens especiales/padding
        sequence_ids = tokenized.sequence_ids(i)
        answers = examples["answers"][sample_mapping[i]]

        if len(answers["answer_start"]) == 0:
            # Pregunta sin respuesta -> apunta a [CLS]
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        # Posición de la respuesta en caracteres dentro del contexto
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Primer y último token del CONTEXTO dentro de esta ventana
        token_start_index = 0
        while sequence_ids[token_start_index] != context_seq_id:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_seq_id:
            token_end_index -= 1

        # ¿La respuesta completa cae dentro de esta ventana?
        if not (offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char):
            # No cae -> etiqueta [CLS] (en esta ventana "no hay respuesta")
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            # Avanza hasta el token que contiene el carácter inicial
            # (el límite `<= token_end_index` evita desbordarse hacia [SEP]/padding, cuyo
            #  offset es (0, 0), cuando la respuesta es el último token del contexto)
            while (token_start_index <= token_end_index
                   and offsets[token_start_index][0] <= start_char):
                token_start_index += 1
            tokenized["start_positions"].append(token_start_index - 1)
            # Retrocede hasta el token que contiene el carácter final
            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized["end_positions"].append(token_end_index + 1)

    return tokenized


def prepare_validation_features(examples, tokenizer, max_length=384, stride=128):
    """
    Igual que la tokenización de entrenamiento, pero SIN etiquetas. Guarda:
      - `example_id`: a qué ejemplo original pertenece cada ventana.
      - `offset_mapping`: solo para tokens del contexto (los demás -> None),
        así el post-procesado puede recuperar el texto de la respuesta
        y descartar spans que caigan en la pregunta o en el padding.
    Se usa para calibrar el umbral de "sin respuesta".
    """
    pad_on_right = tokenizer.padding_side == "right"
    context_seq_id = 1 if pad_on_right else 0

    tokenized = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        tokenized["example_id"].append(examples["id"][sample_mapping[i]])
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_seq_id else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    return tokenized


# ==========================================================================
# 2. POST-PROCESADO Y MÉTRICAS (para calibrar el umbral)
# ==========================================================================
def normalize_answer(s):
    """Normalización oficial de SQuAD: minúsculas, sin puntuación ni artículos."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction, gold):
    """F1 a nivel de tokens. Si ambos son vacíos (null) -> 1.0; si solo uno -> 0.0."""
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(pred_toks), n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def compute_span_and_null_scores(examples, features, start_logits, end_logits,
                                 n_best=20, max_answer_len=30):
    """
    Para cada ejemplo calcula, combinando todas sus ventanas:
        s_null = min_k ( start_k[CLS] + end_k[CLS] )         (puntaje de "sin respuesta")
        s_best = max_{i<=j} ( start[i] + end[j] )            (mejor span en el contexto)
        diff   = s_null - s_best
    y el texto del mejor span. Devuelve dict example_id -> (diff, texto, start_char).
    """
    example_ids = features["example_id"]
    offsets_all = features["offset_mapping"]
    feats_by_example = collections.defaultdict(list)
    for fi, eid in enumerate(example_ids):
        feats_by_example[eid].append(fi)

    out = {}
    for ex in examples:
        context = ex["context"]
        null_score = None
        best = {"score": -1e9, "text": "", "start_char": None}

        for fi in feats_by_example[ex["id"]]:
            s_log, e_log = start_logits[fi], end_logits[fi]
            offsets = offsets_all[fi]

            # Puntaje "sin respuesta" de la ventana; nos quedamos con el MÍNIMO
            # entre ventanas (la ventana más "convencida" de que sí hay respuesta).
            feat_null = s_log[0] + e_log[0]
            null_score = feat_null if null_score is None else min(null_score, feat_null)

            start_idxs = np.argsort(s_log)[-1:-n_best - 1:-1].tolist()
            end_idxs = np.argsort(e_log)[-1:-n_best - 1:-1].tolist()
            for si in start_idxs:
                for ei in end_idxs:
                    # Descarta tokens fuera del contexto, spans invertidos o muy largos
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    if ei < si or ei - si + 1 > max_answer_len:
                        continue
                    score = s_log[si] + e_log[ei]
                    if score > best["score"]:
                        best = {
                            "score": float(score),
                            "text": context[offsets[si][0]: offsets[ei][1]],
                            "start_char": offsets[si][0],
                        }

        out[ex["id"]] = (float(null_score - best["score"]), best["text"], best["start_char"])
    return out


def find_best_threshold(scores, examples):
    """
    Barre umbrales tau y elige el que maximiza el F1 promedio en validación.
    Regla:  predecir null  si  diff = s_null - s_best  >  tau.
    """
    golds = {ex["id"]: (ex["answers"]["text"] or [""]) for ex in examples}
    diffs = np.array([v[0] for v in scores.values()])
    # Rango del barrido por percentiles: evita que casos extremos (p. ej. sin ningún
    # span válido, diff enorme) vuelvan gigante la rejilla de umbrales.
    lo, hi = np.percentile(diffs, [0.5, 99.5])
    grid = np.arange(np.floor(lo) - 1, np.ceil(hi) + 1.25, 0.25)

    best_tau, best_f1, history = 0.0, -1.0, []
    for tau in grid:
        f1s = []
        for eid, (diff, text, _) in scores.items():
            pred = "" if diff > tau else text
            f1s.append(max(f1_score(pred, g) for g in golds[eid]))
        f1 = float(np.mean(f1s))
        history.append((float(tau), f1))
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    return best_tau, best_f1, history


# ==========================================================================
# 3. GRÁFICA DE PÉRDIDA
# ==========================================================================
def save_loss_plot(log_history, path=os.path.join(LOGS_DIR, "loss.png")):
    """Extrae loss de train y de validación de trainer.state.log_history y las grafica."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    train_steps = [h["step"] for h in log_history if "loss" in h]
    train_loss = [h["loss"] for h in log_history if "loss" in h]
    eval_steps = [h["step"] for h in log_history if "eval_loss" in h]
    eval_loss = [h["eval_loss"] for h in log_history if "eval_loss" in h]

    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, train_loss, label="Pérdida de entrenamiento", alpha=0.8)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, "o-", label="Pérdida de validación")
    plt.xlabel("Paso de entrenamiento")
    plt.ylabel("Loss")
    plt.title(f"Fine-tuning {MODEL_NAME} en SQuAD 2.0")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[OK] Gráfica guardada en {path}")


# ==========================================================================
# 4. VERIFICACIÓN DE COMPATIBILIDAD CON CPU
# ==========================================================================
def verify_cpu_loading(model_dir, tokenizer, max_length):
    """Recarga el modelo guardado forzando CPU y hace una inferencia de prueba."""
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)  # fp32 por defecto
    model.to("cpu").eval()
    assert all(p.device.type == "cpu" for p in model.parameters()), "¡Hay pesos fuera de CPU!"
    enc = tokenizer("Who?", "Alice went home.", return_tensors="pt",
                    truncation=True, max_length=max_length)
    with torch.no_grad():
        out = model(**enc)
    print(f"[OK] Recarga en CPU correcta (logits: {tuple(out.start_logits.shape)})")


# ==========================================================================
# 5. FUNCIÓN DE INFERENCIA (referencia para la API FastAPI)
# ==========================================================================
def predict_with_threshold(question, context, model, tokenizer, tau,
                           max_length=384, stride=128, n_best=20, max_answer_len=30):
    """
    Ejemplo REAL de cómo usar el umbral en inferencia (CPU). Devuelve un dict
    {"answer", "start", "score"} compatible con el contrato de la API.
    La respuesta cumple context[start:start+len(answer)] == answer.
    """
    enc = tokenizer(
        question, context, truncation="only_second", max_length=max_length,
        stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True,
        padding=True, return_tensors="pt",
    )
    offset_mapping = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping")

    with torch.no_grad():
        out = model(**enc)
    start_logits = out.start_logits.cpu().numpy()
    end_logits = out.end_logits.cpu().numpy()

    s_null, best = None, {"score": -1e9, "text": None, "start": None}
    for k in range(start_logits.shape[0]):
        seq_ids = enc.sequence_ids(k)
        offsets = [o if seq_ids[j] == 1 else None for j, o in enumerate(offset_mapping[k].tolist())]
        s_log, e_log = start_logits[k], end_logits[k]
        null_k = s_log[0] + e_log[0]
        s_null = null_k if s_null is None else min(s_null, null_k)
        for si in np.argsort(s_log)[-1:-n_best - 1:-1]:
            for ei in np.argsort(e_log)[-1:-n_best - 1:-1]:
                if offsets[si] is None or offsets[ei] is None:
                    continue
                if ei < si or ei - si + 1 > max_answer_len:
                    continue
                sc = s_log[si] + e_log[ei]
                if sc > best["score"]:
                    best = {"score": float(sc), "start": offsets[si][0],
                            "text": context[offsets[si][0]: offsets[ei][1]]}

    diff = float(s_null - best["score"])
    if best["text"] is None or diff > tau:
        # Confianza en "null": sube cuanto más supera diff al umbral
        return {"answer": None, "start": None, "score": float(1 / (1 + np.exp(-(diff - tau)))),
                "s_null": float(s_null), "s_best": float(best["score"]), "diff": diff}
    return {"answer": best["text"], "start": int(best["start"]),
            "score": float(1 / (1 + np.exp(-(tau - diff)))),
            "s_null": float(s_null), "s_best": float(best["score"]), "diff": diff}


# ==========================================================================
# 6. MAIN
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_train_samples", type=int, default=None, help="submuestreo opcional")
    p.add_argument("--max_eval_samples", type=int, default=2000, help="ejemplos de validación")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # ---- Datos ------------------------------------------------------------
    raw = load_dataset("rajpurkar/squad_v2")
    train_raw, val_raw = raw["train"], raw["validation"]
    if args.max_train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(train_raw))))
    if args.max_eval_samples:
        val_raw = val_raw.shuffle(seed=args.seed).select(
            range(min(args.max_eval_samples, len(val_raw))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Features de entrenamiento y de validación CON etiquetas (para eval_loss)
    kw = dict(tokenizer=tokenizer, max_length=args.max_length, stride=args.doc_stride)
    train_ds = train_raw.map(lambda x: prepare_train_features(x, **kw), batched=True,
                             remove_columns=train_raw.column_names, desc="Tokenizando train")
    val_ds = val_raw.map(lambda x: prepare_train_features(x, **kw), batched=True,
                         remove_columns=val_raw.column_names, desc="Tokenizando val (loss)")

    # ---- Modelo (checkpoint BASE, no afinado en SQuAD) ---------------------
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # ---- Entrenamiento -----------------------------------------------------
    use_cuda = torch.cuda.is_available()
    # Compatibilidad entre versiones de transformers (los nombres cambiaron):
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    compat = {}
    compat["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "steps"
    if "warmup_ratio" in ta_params:
        compat["warmup_ratio"] = 0.1
    else:                              # transformers >= 5: un float < 1 se interpreta como ratio
        compat["warmup_steps"] = 0.1
    training_args = TrainingArguments(
        **compat,
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,              # registra loss de train cada 50 pasos
        eval_steps=500,                # registra eval_loss cada 500 pasos
        save_strategy="no",            # guardamos solo el modelo final (ahorra disco)
        fp16=use_cuda,                 # AMP solo en GPU; los pesos siguen en fp32
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        **{("processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters
            else "tokenizer"): tokenizer},
    )
    trainer.train()
    trainer.evaluate()  # asegura al menos un eval_loss final en el historial

    # ---- Logs y gráfica obligatoria logs/loss.png --------------------------
    history = trainer.state.log_history
    with open(os.path.join(LOGS_DIR, "log_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_loss_plot(history, os.path.join(LOGS_DIR, "loss.png"))

    # ---- Calibración del umbral de "sin respuesta" -------------------------
    val_feats = val_raw.map(lambda x: prepare_validation_features(x, **kw), batched=True,
                            remove_columns=val_raw.column_names, desc="Tokenizando val (umbral)")
    pred = trainer.predict(val_feats.remove_columns(["example_id", "offset_mapping"]))
    start_logits, end_logits = pred.predictions[:2]
    scores = compute_span_and_null_scores(val_raw, val_feats, start_logits, end_logits)
    tau, f1, _ = find_best_threshold(scores, val_raw)
    print(f"[OK] Umbral óptimo tau = {tau:.2f}  (F1 validación = {100 * f1:.2f})")

    # ---- Guardado compatible con CPU --------------------------------------
    trainer.model.to("cpu")            # libera cualquier anclaje a CUDA
    trainer.model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    info = {
        "model": MODEL_NAME,
        "params": int(sum(p.numel() for p in trainer.model.parameters())),
        "max_seq_length": args.max_length,
        "doc_stride": args.doc_stride,
        "no_answer_threshold": tau,
        "val_f1": f1,
    }
    with open(os.path.join(OUTPUT_MODEL_DIR, "no_answer_threshold.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[OK] Modelo y tokenizador guardados en {OUTPUT_MODEL_DIR}")

    verify_cpu_loading(OUTPUT_MODEL_DIR, tokenizer, args.max_length)


if __name__ == "__main__":
    main()


# ==========================================================================
# 7. LÓGICA MATEMÁTICA DEL UMBRAL "NO-ANSWER" EN INFERENCIA
# ==========================================================================
"""
Notación: para una ventana k de la entrada, el modelo produce logits
S_k[t] (inicio) y E_k[t] (fin) para cada token t. El token t=0 es [CLS].

1) Puntaje de "sin respuesta" (apuntar a [CLS]):
        s_null = min_k ( S_k[0] + E_k[0] )
   Se toma el mínimo entre ventanas: si UNA ventana encuentra evidencia clara
   de respuesta, esa ventana tendrá s_null bajo y no debemos decir "null".

2) Puntaje del mejor span de texto (solo tokens del contexto, i <= j,
   longitud <= max_answer_len):
        s_best = max_{k, i<=j} ( S_k[i] + E_k[j] )

3) Diferencia:
        diff = s_null - s_best
   En términos de probabilidad, exp(S+E) es proporcional a la probabilidad
   conjunta (no normalizada) de cada par (inicio, fin). Por tanto
        diff = log( P(CLS,CLS) / P(mejor span) )   (la normalización se cancela)
   diff > 0  -> el modelo cree más probable "sin respuesta" que el mejor span.

4) Regla de decisión con umbral tau (calibrado en validación):
        si diff > tau  ->  answer = null, start = null
        si no          ->  answer = texto del mejor span
   tau = 0 es la regla "argmax" ingenua; en la práctica el tau óptimo suele
   ser distinto de 0 porque el modelo está mal calibrado. Por eso se barre tau
   sobre validación y se elige el valor que maximiza F1 (find_best_threshold).
   tau mayor -> el modelo responde más a menudo (menos nulls);
   tau menor -> el modelo se abstiene más a menudo.

5) Score en [0, 1] para la API (no se califica):
        score_null = sigmoid(diff - tau)     si se devuelve null
        score_span = sigmoid(tau - diff)     si se devuelve un span

Para el servicio FastAPI: cargar modelo y tokenizador con
    AutoModelForQuestionAnswering.from_pretrained("./modelo_guardado")  # CPU por defecto
    tau = json.load(open("modelo_guardado/no_answer_threshold.json"))["no_answer_threshold"]
y llamar a predict_with_threshold(question, context, model, tokenizer, tau).
Recomendación: volver a recalibrar tau con preguntas parecidas al set oculto
(textos que NO son de SQuAD, ~25 % sin respuesta) si dispone de ellas, p. ej. public_sample.json.
"""