"""
app.py
------
FastAPI service — Extractive QA as a Service
Implementa el contrato definido en qa_api_contract.yaml

Endpoints:
  GET  /health   → liveness check (público)
  GET  /info     → metadata del equipo y modelo (protegido)
  POST /predict  → responde una pregunta sobre un contexto (protegido)

Uso:
  export QA_API_TOKEN=mi_token_secreto
  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

import torch
import numpy as np
import collections
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR        = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "modernbert-base-squad2-final"))
QA_API_TOKEN     = os.getenv("QA_API_TOKEN", "")          # OBLIGATORIO en producción
MAX_SEQ_LENGTH   = int(os.getenv("MAX_SEQ_LENGTH", "384"))
DOC_STRIDE       = int(os.getenv("DOC_STRIDE", "128"))
MAX_ANSWER_LEN   = 30
N_BEST           = 20
NULL_THRESHOLD   = float(os.getenv("NULL_THRESHOLD", "0.0"))  # calibrar en validation set
TEAM_NAME        = os.getenv("TEAM_NAME", "team-XX")           # cambiar al nombre del equipo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Estado global del modelo (cargado una sola vez al arrancar)
# ──────────────────────────────────────────────────────────────
model_state = {"tokenizer": None, "model": None, "params": 0, "ready": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar el servidor."""
    target_dir = MODEL_DIR
    if not os.path.isabs(target_dir):
        target_dir = os.path.join(BASE_DIR, target_dir)
    if not os.path.exists(target_dir) or len(os.listdir(target_dir)) == 0:
        target_dir = "answerdotai/ModernBERT-base"
    logger.info(f"Cargando modelo desde: {target_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_dir)
        model     = AutoModelForQuestionAnswering.from_pretrained(target_dir)
        model.eval()
        # Usar CPU para serving (el contrato no garantiza GPU)
        model_state["tokenizer"] = tokenizer
        model_state["model"]     = model
        model_state["params"]    = model.num_parameters()
        model_state["ready"]     = True
        logger.info(f"Modelo listo — {model_state['params']:,} parámetros")
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
    yield
    logger.info("Apagando servicio...")

# ──────────────────────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Extractive QA Service",
    version="1.0",
    lifespan=lifespan,
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Garantiza que todas las respuestas de error cumplan el esquema de qa_api_contract.yaml."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    error_code = "unauthorized" if exc.status_code == 401 else ("bad_request" if exc.status_code in (400, 422) else "internal_error")
    return JSONResponse(status_code=exc.status_code, content={"error": error_code, "detail": str(exc.detail)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Devuelve error bad_request exacto segun el contrato OpenAPI."""
    return JSONResponse(status_code=422, content={"error": "bad_request", "detail": str(exc)})

security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Valida el Bearer token contra QA_API_TOKEN."""
    if not QA_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "detail": "QA_API_TOKEN not set"},
        )
    if credentials is None or credentials.credentials != QA_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
            detail={"error": "unauthorized"},
        )

# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., description="ok or loading")

class InfoResponse(BaseModel):
    team:           str = Field(..., description="Team name registered with instructor")
    model:          str = Field(..., description="Hugging Face model checkpoint ID")
    params:         int = Field(..., description="Total model parameters")
    max_seq_length: Optional[int] = Field(None, description="Maximum tokens per window")
    doc_stride:     Optional[int] = Field(None, description="Overlap between sliding windows")

class PredictRequest(BaseModel):
    id:       str = Field(..., min_length=1, max_length=64)
    context:  str = Field(..., min_length=1, max_length=5000)
    question: str = Field(..., min_length=1, max_length=500)

class PredictResponse(BaseModel):
    id:     str
    answer: Optional[str]
    start:  Optional[int]
    score:  float

# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["public"])
def health():
    """Liveness check — no requiere token."""
    if model_state["ready"]:
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "loading"})


@app.get("/info", response_model=InfoResponse, tags=["protected"], dependencies=[Depends(verify_token)])
def info():
    """Metadata del equipo y del modelo."""
    return {
        "team":           TEAM_NAME,
        "model":          "answerdotai/ModernBERT-base",
        "params":         model_state["params"],
        "max_seq_length": MAX_SEQ_LENGTH,
        "doc_stride":     DOC_STRIDE,
    }


@app.post("/predict", response_model=PredictResponse,
          tags=["protected"], dependencies=[Depends(verify_token)])
def predict(req: PredictRequest):
    """Responde una pregunta sobre un contexto usando el modelo fine-tuned."""
    if not model_state["ready"]:
        raise HTTPException(status_code=503, detail={"error": "internal_error",
                                                      "detail": "Model not ready"})

    tokenizer = model_state["tokenizer"]
    model     = model_state["model"]

    try:
        # Tokenizar con sliding window para contextos largos
        inputs = tokenizer(
            req.question,
            req.context,
            max_length=MAX_SEQ_LENGTH,
            truncation="only_second",
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map     = inputs.pop("overflow_to_sample_mapping", None)

        with torch.no_grad():
            outputs = model(**{k: v for k, v in inputs.items()
                               if k in ("input_ids", "attention_mask", "token_type_ids")
                               and v is not None})

        start_logits = outputs.start_logits.numpy()
        end_logits   = outputs.end_logits.numpy()

        # Encontrar el mejor span con lógica de no-answer
        best_answer_text  = ""
        best_answer_start = None
        best_answer_score = -float("inf")
        min_null_score    = None

        for chunk_idx in range(start_logits.shape[0]):
            sl = start_logits[chunk_idx]
            el = end_logits[chunk_idx]
            raw_offsets = offset_mapping[chunk_idx].tolist()
            seq_ids     = inputs.sequence_ids(chunk_idx) if hasattr(inputs, "sequence_ids") else None
            input_ids_chunk = inputs["input_ids"][chunk_idx].tolist()

            cls_index  = input_ids_chunk.index(tokenizer.cls_token_id)
            null_score = float(sl[cls_index] + el[cls_index])
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score

            # Mask offsets: solo tokens del contexto (seq_ids == 1) son spans válidos
            offsets = [
                raw_offsets[k] if (seq_ids is None or seq_ids[k] == 1) else None
                for k in range(len(raw_offsets))
            ]

            start_idxs = np.argsort(sl)[-N_BEST:].tolist()
            end_idxs   = np.argsort(el)[-N_BEST:].tolist()

            for si in start_idxs:
                for ei in end_idxs:
                    if ei < si or ei - si + 1 > MAX_ANSWER_LEN:
                        continue
                    if offsets[si] is None or offsets[ei] is None:
                        continue
                    score = sl[si] + el[ei]
                    if score > best_answer_score:
                        char_start = offsets[si][0]
                        char_end   = offsets[ei][1]
                        raw_span   = req.context[char_start:char_end]
                        leading_ws = len(raw_span) - len(raw_span.lstrip())
                        trimmed    = raw_span.strip()
                        if trimmed:
                            best_answer_score = score
                            best_answer_text  = trimmed
                            best_answer_start = char_start + leading_ws

        # No-answer threshold
        if min_null_score is not None and best_answer_score <= min_null_score + NULL_THRESHOLD:
            # Normalizar score a [0, 1]
            confidence = float(1 / (1 + np.exp(-(min_null_score - best_answer_score))))
            return PredictResponse(id=req.id, answer=None, start=None, score=round(confidence, 4))

        # Normalizar score de respuesta a [0, 1]
        confidence = float(1 / (1 + np.exp(-(best_answer_score - (min_null_score or 0)))))
        return PredictResponse(
            id=req.id,
            answer=best_answer_text,
            start=best_answer_start,
            score=round(confidence, 4),
        )

    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "internal_error"})
