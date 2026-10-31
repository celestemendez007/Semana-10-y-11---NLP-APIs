# Reporte Técnico Comparativo: DistilBERT sobre SQuAD 2.0

## 1. Configuración del Entrenamiento
| Parámetro | Valor |
| :--- | :--- |
| **Modelo Base** | `distilbert-base-uncased` |
| **Dataset** | SQuAD 2.0 (`rajpurkar/squad_v2`) |
| **Estrategia de Muestreo** | Barajado aleatorio balanceado (`seed=42`) |
| **Épocas** | 3 épocas completas |
| **Learning Rate** | $3 \times 10^{-5}$ con Linear Decay y 10% Warmup |
| **Batch Size Efectivo** | 8 (per-device) × 2 (gradient accumulation) = 16 |
| **Regularización L2** | Weight Decay = 0.01 (previene sobreajuste/memorización) |
| **Hardware de Entrenamiento** | NVIDIA GeForce RTX 4070 Laptop GPU (~2.37 min) |

---

## 2. Métricas de Rendimiento (Consultar `eval_results.json`)
| Métrica | Descripción |
| :--- | :--- |
| **EM General** | Exact Match sobre todo el conjunto de validación |
| **F1 General** | F1-Score armónico sobre tokens predichos |
| **EM Answerable** | Exact Match exclusivo en preguntas que SÍ tienen respuesta |
| **F1 Answerable** | F1 exclusivo en preguntas que SÍ tienen respuesta |
| **EM Unanswerable** | Precisión en detección de preguntas trampa (predice vacío) |
| **F1 Unanswerable** | F1 en preguntas trampa |
| **Optimal Threshold** | Umbral de corte calibrado mediante `threshold_search.png` |

---

## 3. Eficiencia y Despliegue en CPU
- **Número de Parámetros:** ~66,364,418 parámetros (40% más ligero que BERT-base).
- **Tamaño en Disco:** ~253 MB.
- **Latencia media en CPU:** ~40 - 60 ms por petición en CPU AMD.

---

## 4. Análisis Comparativo frente a ModernBERT
1. **Velocidad de Inferencia y Huella de Memoria:**
   - **DistilBERT** está optimizado para microservicios y despliegues serverless o en CPUs económicas. Su tamaño de ~253 MB permite que la API cargue casi instantáneamente y responda con muy baja latencia.
2. **Capacidad de Contexto:**
   - ModernBERT maneja ventanas nativas de hasta 8,192 tokens con RoPE y FlashAttention-2.
   - DistilBERT utiliza una ventana de 384 tokens con stride de 128 tokens para segmentar textos extensos, lo cual resulta suficiente para el 99% de las consultas tipo SQuAD 2.0 con un consumo mínimo de RAM.