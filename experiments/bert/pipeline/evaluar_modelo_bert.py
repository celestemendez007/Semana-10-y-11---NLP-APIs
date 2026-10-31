#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EvalÃºa el modelo guardado: calidad (EM, F1, No-Answer EM), operaciÃ³n (latencia en CPU,
parÃ¡metros, tamaÃ±o) y muestra las variables internas del umbral (s_null, s_best, diff).

Uso:   python evaluar_modelo.py --samples 500
Requiere estar en la misma carpeta que modelo.py y modelo_guardado/.
"""
import argparse, json, os, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from train_bert import normalize_answer, f1_score, predict_with_threshold

MODEL_DIR = "./modelo_guardado"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- Carga forzando CPU (igual que en la API) ---------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).to("cpu").eval()
    info = json.load(open(os.path.join(MODEL_DIR, "no_answer_threshold.json")))
    tau, max_len, stride = info["no_answer_threshold"], info["max_seq_length"], info["doc_stride"]

    # --- Datos de validaciÃ³n -------------------------------------------------
    val = load_dataset("rajpurkar/squad_v2")["validation"].shuffle(seed=args.seed)
    val = val.select(range(min(args.samples, len(val))))

    em, f1, has_em, has_f1, no_em, times, rows = [], [], [], [], [], [], []
    for ex in val:
        golds = ex["answers"]["text"]            # lista vacÃ­a => sin respuesta
        t0 = time.perf_counter()
        out = predict_with_threshold(ex["question"], ex["context"], model, tok, tau, max_len, stride)
        times.append(time.perf_counter() - t0)

        pred = out["answer"] or ""               # null -> cadena vacÃ­a
        if golds:                                 # pregunta CON respuesta
            e = max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds)
            f = max(f1_score(pred, g) for g in golds)
            has_em.append(e); has_f1.append(f)
        else:                                     # pregunta SIN respuesta: acierta si devuelve null
            e = f = float(out["answer"] is None)
            no_em.append(e)
        em.append(e); f1.append(f)
        rows.append((bool(golds), out["diff"]))

    # --- Reporte ---------------------------------------------------------------
    nparams = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(os.path.join(MODEL_DIR, "model.safetensors")) / 1e6
    t = np.array(times)
    print("\n=== CALIDAD (", len(val), "ejemplos ) ===")
    print(f"Exact Match (EM) total : {100*np.mean(em):.2f}")
    print(f"F1 total               : {100*np.mean(f1):.2f}")
    print(f"EM con respuesta       : {100*np.mean(has_em):.2f}   (n={len(has_em)})")
    print(f"No-Answer EM (null ok) : {100*np.mean(no_em):.2f}   (n={len(no_em)})")
    print("\n=== OPERACIÃ“N (CPU) ===")
    print(f"Latencia media / p95 / mÃ¡x : {t.mean():.3f} s / {np.percentile(t,95):.3f} s / {t.max():.3f} s  (lÃ­mite 5 s)")
    print(f"ParÃ¡metros : {nparams:,}   |   TamaÃ±o en disco : {size_mb:.0f} MB")
    print(f"Umbral tau : {tau}")
    print("\n=== VARIABLES DEL UMBRAL (diff = s_null - s_best) ===")
    d_has = [d for h, d in rows if h]; d_no = [d for h, d in rows if not h]
    print(f"diff medio, preguntas CON respuesta : {np.mean(d_has):.2f}")
    print(f"diff medio, preguntas SIN respuesta : {np.mean(d_no):.2f}   (debe ser mayor)")
    print("\nEjemplos (s_null, s_best, diff -> decisiÃ³n):")
    for ex in val.select(range(min(3, len(val)))):
        o = predict_with_threshold(ex["question"], ex["context"], model, tok, tau, max_len, stride)
        dec = "null" if o["answer"] is None else repr(o["answer"])
        print(f"  s_null={o['s_null']:.2f}  s_best={o['s_best']:.2f}  diff={o['diff']:.2f} -> {dec}")


if __name__ == "__main__":
    main()
