# Ficha comparativa — RoBERTa-base vs. ModernBERT-base

Valores de RoBERTa-base: `logs_roberta/*.json`. Valores de ModernBERT-base: `logs_modernBert/`
(documentados por Celeste). Ambos evaluados en el validation completo de SQuAD v2 con la métrica oficial (EM/F1).

| | RoBERTa-base | ModernBERT-base (Celeste) |
|---|---|---|
| Checkpoint | `FacebookAI/roberta-base` | `answerdotai/ModernBERT-base` |
| Ejemplos de train | 30,000 | 30,000 |
| Epochs | 2 | 2 |
| LR / warmup | 3e-05 / 10% | 3e-05 / 10% |
| Batch efectivo | 32 | 32 |
| max_seq_length / stride | 384 / 128 | 384 / 128 |
| GPU | Tesla T4 | RTX 3050 Ti 4GB |
| Tiempo de entrenamiento | 0.37 h | 2.85 h |
| **EM global** | **74.93** | **77.67** |
| **F1 global** | **78.31** | **81.04** |
| EM / F1 contestables | 73.06 / 79.84 | 72.39 / 79.14 |
| EM / F1 no contestables | 76.79 / 76.79 | 82.94 / 82.94 |
| Parámetros | 124,056,578 | 149,606,402 |
| Tamaño de pesos | 473.3 MB | 572 MB |
| Latencia CPU media | 401.6 ms (Google Colab) | ~1,215–1,450 ms (laptop, vía API) |

**Cuidado al comparar:** los modelos no se entrenaron con la misma cantidad de datos ni epochs,
y la latencia se midió en máquinas y formas distintas (medida en Google Colab vs. llamadas a la API en laptop).

## Errores de RoBERTa-base (threshold 0.0)

- Contestables: {'correct': 4331, 'false_negative': 732, 'partial_overlap': 661, 'wrong_span': 204}
- No contestables: {'correct': 4565, 'false_positive': 1380}
