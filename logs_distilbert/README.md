# Artefactos y Métricas de DistilBERT (SQuAD 2.0)

Este directorio contiene los resultados generados por el pipeline de entrenamiento, evaluación y análisis de errores para el modelo **DistilBERT-base-uncased**.

## Contenido del Directorio

- **`eda_stats.json` & `eda_squad2.png`**: Estadísticas del análisis exploratorio del dataset SQuAD 2.0 (distribución de clases respondibles vs no respondibles y longitud de contextos).
- **`loss.png`**: Curva de convergencia de pérdida de entrenamiento a lo largo de las 3 épocas.
- **`train_results.json`**: Metadatos del entrenamiento (hardware utilizado, tiempo total, pérdida final y tamaño de lote efectivo).
- **`threshold_search.png`**: Curvas de búsqueda de umbral óptimo de confianza contrastando Exact Match (EM) y F1-Score para maximizar el desempeño en preguntas sin respuesta.
- **`eval_results.json`**: Métricas oficiales consolidadas (EM y F1 globales, answerable, unanswerable, latencia en CPU y tamaño del modelo).
- **`error_analysis.json`**: Desglose cualitativo de errores (falsos positivos, falsos negativos y desalineación de límites de caracteres).
- **`DISTILBERT_COMPARISON.md`**: Reporte técnico y comparativo para la entrega académica.