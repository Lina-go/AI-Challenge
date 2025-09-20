# -*- coding: utf-8 -*-
"""
test_florence_signatures.py

Detección y extracción de firmantes/cargos en PDFs.
- Soporta CPU por defecto.
- Usa texto del PDF; opcionalmente usa Florence-2 para OCR cuando el PDF es escaneado.
- Incluye fix para evitar AttributeError: '_supports_sdpa' forzando attn_implementation='eager'.
- Guarda resultados en CSV y JSON.

Requisitos:
  pip install torch>=2.2 transformers>=4.44 huggingface_hub>=0.24 pillow>=10.3 pymupdf>=1.24 pandas regex

Uso:
  python -m signature_extractor_v4.test_florence_signatures input.pdf
  # Opciones:
  --model microsoft/Florence-2-base
  --device auto|cpu|cuda
  --out ./salidas
  --ocr-if-needed        # intenta OCR con Florence si no hay texto en la página
  --no-vlm               # desactiva totalmente Florence/VLM
  --dpi 220              # raster para OCR

Autor: tú + ChatGPT 🛠️
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image

# Dependencias VLM (opcionales si --no-vlm)
try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
except Exception:
    torch = None
    AutoProcessor = None
    AutoModelForCausalLM = None
    AutoConfig = None


# --------------------------- Logging ---------------------------------

LOG = logging.getLogger("signatures")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------- Utilidades ---------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def choose_device(user_choice: str = "auto") -> str:
    if user_choice in ("cpu", "cuda"):
        return user_choice
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pil_from_pixmap(pm: fitz.Pixmap) -> Image.Image:
    if pm.n >= 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    img = Image.frombytes(mode, (pm.width, pm.height), pm.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img


# --------------------------- Regex & Heurísticas ---------------------------------

# Palabras clave para contexto de firmas
KW_SIG = [
    r"\bfirma\b", r"\bfirmas\b", r"\bfirmó\b", r"\bfirmante\b",
    r"\bsuscribe\b", r"\bsuscrito\b", r"\bsuscrita\b",
    r"\bsigned\b", r"\bsignature\b", r"\bsignatory\b",
    r"\bfirma del\b", r"\bfirma de\b", r"\bfirmado por\b"
]

# Cargos frecuentes
KW_ROLE_HINTS = [
    r"cargo", r"posición", r"puesto", r"título", r"role", r"position", r"title",
    r"representante legal", r"apoderado", r"gerente", r"director", r"presidente",
    r"secretario", r"tesorero", r"alcalde", r"rector", r"síndico", r"abogado",
    r"jefe", r"coordinador", r"líder", r"manager", r"chief", r"officer"
]

# Nombre propio: secuencia de 2–4 tokens Capitalizados (con tildes) o mayúsculas
NAME_RE = re.compile(
    r"(?<!\w)("
    r"(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]{2,})"           # Juan / JUAN
    r"(?:\s+(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]{2,})){1,3}"  # de 2 a 4 palabras
    r")(?!\w)"
)

ROLE_LINE_RE = re.compile(
    r"(?:cargo|posición|puesto|título|role|position|title)\s*[:\-–]\s*(.+)$",
    flags=re.I
)

# Limpiar líneas ruidosas
def clean_line(s: str) -> str:
    s = re.sub(r"[•·●\u2022]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class SignatureHit:
    page: int
    name: Optional[str]
    role: Optional[str]
    source: str           # "pdf_text" | "vlm_ocr"
    context: str


def window(lines: List[str], idx: int, k: int = 4) -> List[str]:
    a = max(0, idx - k)
    b = min(len(lines), idx + k + 1)
    return lines[a:b]


def extract_signers_from_text(page_text: str, page_num: int, source: str) -> List[SignatureHit]:
    """Heurística basada en texto para encontrar nombres/cargos alrededor de keywords."""
    hits: List[SignatureHit] = []
    lines = [clean_line(x) for x in page_text.splitlines() if clean_line(x)]
    if not lines:
        return hits

    # Índices de líneas “clave”
    kw_re = re.compile("|".join(KW_SIG), flags=re.I)
    role_hints_re = re.compile("|".join(KW_ROLE_HINTS), flags=re.I)

    for i, line in enumerate(lines):
        if kw_re.search(line):
            ctx_lines = window(lines, i, k=5)
            ctx = "\n".join(ctx_lines)

            # Buscar ROLE explícito en ventana
            role = None
            for l in ctx_lines:
                mrole = ROLE_LINE_RE.search(l)
                if mrole:
                    role = clean_line(mrole.group(1))
                    break

            # Buscar NAME cercano (preferimos líneas con varias Capitalizadas)
            name = None
            # 1) en misma línea
            mname = NAME_RE.search(line)
            if mname:
                name = clean_line(mname.group(1))
            else:
                # 2) una o dos líneas alrededor
                for l in ctx_lines:
                    m2 = NAME_RE.search(l)
                    if m2:
                        name = clean_line(m2.group(1))
                        break

            # Filtro simple: evitar líneas que son puro “Firma” sin contenido
            if (name or role) and not line.lower().startswith("firma:"):
                hits.append(SignatureHit(
                    page=page_num,
                    name=name,
                    role=role,
                    source=source,
                    context=ctx
                ))

        # Si no hay keyword pero sí pistas fuertes de cargo + nombre juntos
        elif role_hints_re.search(line):
            mname = NAME_RE.search(line)
            mrole = ROLE_LINE_RE.search(line)
            if mname or mrole:
                hits.append(SignatureHit(
                    page=page_num,
                    name=clean_line(mname.group(1)) if mname else None,
                    role=clean_line(mrole.group(1)) if mrole else None,
                    source=source,
                    context=line
                ))

    # De-duplicar por (name, role) en misma página
    uniq: Dict[Tuple[Optional[str], Optional[str]], SignatureHit] = {}
    for h in hits:
        key = (h.name, h.role)
        if key not in uniq:
            uniq[key] = h
    return list(uniq.values())


# --------------------------- Florence-2 Wrapper ---------------------------------

class FlorenceSignatureDetector:
    """
    Cargador/Wrapper de Florence-2 en CPU/CUDA con fix para SDPA.
    Para OCR simple (cuando no hay texto nativo).
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if torch is None or AutoModelForCausalLM is None:
            raise RuntimeError("Transformers/Torch no están disponibles en este entorno.")

        LOG.info("Inicializando Florence-2 en dispositivo: %s", self.device)
        LOG.info("Cargando modelo: %s", self.model_name)

        # dtype según dispositivo
        dtype = torch.float16 if (self.device == "cuda") else torch.float32

        # Cargar processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Forzar backend de atención “eager” para evitar _supports_sdpa
        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        # Atributos que distintas versiones consultan:
        setattr(config, "attn_implementation", "eager")
        setattr(config, "_attn_implementation", "eager")

        # Cargar modelo intentando usar `dtype`, con fallback a `torch_dtype`
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                dtype=dtype,
                trust_remote_code=True
            )
        except TypeError:
            # Compatibilidad con versiones que aún esperan torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=dtype,
                trust_remote_code=True
            )

        self.model.to(self.device)
        self.model.eval()

        # En CPU, limitar threads para no saturar
        if self.device == "cpu":
            try:
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
            except Exception:
                pass

    @torch.inference_mode()
    def ocr_text(self, image: Image.Image, max_new_tokens: int = 256) -> str:
        """
        OCR simple con Florence-2:
        Florence-2 suele usar prompts/tokens especiales; aquí pedimos OCR/Caption.
        """
        prompt = "<OCR_WITH_REGION>"  # si el repo del modelo no soporta, probamos OCR genérico
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        text = text.replace("\n\n", "\n").strip()
        return text


# --------------------------- Pipeline de PDF ---------------------------------

@dataclass
class PageResult:
    page: int
    raw_text_source: str
    raw_text_len: int
    hits: List[SignatureHit]


def extract_pdf_text(page: fitz.Page) -> str:
    """
    Prioridad:
      1) Texto 'text' completo.
      2) Si está vacío, concatenar por bloques.
    """
    txt = page.get_text("text")
    if txt and txt.strip():
        return txt
    blocks = page.get_text("blocks") or []
    blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))  # y, x
    lines = []
    for b in blocks_sorted:
        if len(b) >= 9:
            lines.append(str(b[4]))
        else:
            lines.append(str(b[4]))
    return "\n".join(lines)


def rasterize_page(page: fitz.Page, dpi: int = 200) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    return pil_from_pixmap(pm)


def process_pdf(
    pdf_path: str,
    use_vlm_ocr_if_needed: bool = False,
    vlm: Optional[FlorenceSignatureDetector] = None,
    dpi: int = 200
) -> List[PageResult]:
    doc = fitz.open(pdf_path)
    LOG.info("PDF abierto: %s | páginas: %d", pdf_path, doc.page_count)

    results: List[PageResult] = []
    for i, page in enumerate(doc, start=1):
        raw_text = extract_pdf_text(page) or ""
        source = "pdf_text"
        if use_vlm_ocr_if_needed and len(raw_text.strip()) < 30:
            # Página sin texto nativo; intentar OCR con VLM si disponible
            if vlm is not None:
                try:
                    img = rasterize_page(page, dpi=dpi)
                    raw_text = vlm.ocr_text(img)
                    source = "vlm_ocr"
                except Exception as e:
                    LOG.warning("OCR VLM falló en página %d: %s", i, e)

        hits = extract_signers_from_text(raw_text, page_num=i, source=source)
        results.append(PageResult(
            page=i,
            raw_text_source=source,
            raw_text_len=len(raw_text),
            hits=hits
        ))
    doc.close()
    return results


# --------------------------- Serialización ---------------------------------

def results_to_rows(results: List[PageResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pr in results:
        if not pr.hits:
            rows.append({
                "page": pr.page,
                "name": None,
                "role": None,
                "source": pr.raw_text_source,
                "raw_text_len": pr.raw_text_len,
                "context": None
            })
        else:
            for h in pr.hits:
                rows.append({
                    "page": h.page,
                    "name": h.name,
                    "role": h.role,
                    "source": h.source,
                    "raw_text_len": pr.raw_text_len,
                    "context": h.context
                })
    return rows


def save_outputs(rows: List[Dict[str, Any]], out_dir: str, base: str) -> Tuple[str, str]:
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"{base}_signatures.csv")
    json_path = os.path.join(out_dir, f"{base}_signatures.json")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


# --------------------------- CLI ---------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extractor de firmas/cargos en PDF")
    ap.add_argument("pdf", help="Ruta al PDF de entrada")
    ap.add_argument("--out", default="outputs", help="Directorio de salida (CSV/JSON)")
    ap.add_argument("--model", default="microsoft/Florence-2-base", help="Nombre del modelo Florence-2")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Dispositivo")
    ap.add_argument("--ocr-if-needed", action="store_true", help="Usar VLM OCR si la página no tiene texto nativo")
    ap.add_argument("--no-vlm", action="store_true", help="Desactivar por completo VLM (ignora --ocr-if-needed)")
    ap.add_argument("--dpi", type=int, default=220, help="DPI para rasterizar página cuando se hace OCR")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        LOG.error("No existe el archivo PDF: %s", pdf_path)
        sys.exit(2)

    device = choose_device(args.device)
    vlm = None
    if not args.no_vlm and args.ocr_if_needed:
        if torch is None:
            LOG.warning("Transformers/Torch no disponibles; no se puede usar OCR VLM.")
        else:
            try:
                vlm = FlorenceSignatureDetector(args.model, device=device)
            except Exception as e:
                LOG.error("Error cargando modelo: %s", e)
                LOG.error("Continuando sin VLM/OCR...")
                vlm = None

    # Procesar PDF
    results = process_pdf(
        pdf_path,
        use_vlm_ocr_if_needed=(args.ocr_if_needed and vlm is not None),
        vlm=vlm,
        dpi=args.dpi
    )

    # Serializar
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    csv_path, json_path = save_outputs(results_to_rows(results), args.out, base)

    # Resumen a consola
    total_hits = sum(len(r.hits) for r in results)
    LOG.info("Páginas procesadas: %d", len(results))
    LOG.info("Firmantes/cargos detectados: %d", total_hits)
    LOG.info("CSV: %s", csv_path)
    LOG.info("JSON: %s", json_path)

    # Impresión compacta de hallazgos
    if total_hits:
        print("\n=== Resumen de firmantes detectados ===")
        for r in results:
            for h in r.hits:
                nm = h.name or "(nombre no detectado)"
                rl = h.role or "(cargo no detectado)"
                print(f"- Página {h.page}: {nm} — {rl}  [{h.source}]")
    else:
        print("\nNo se detectaron firmantes/cargos con las reglas actuales.")


if __name__ == "__main__":
    main()
