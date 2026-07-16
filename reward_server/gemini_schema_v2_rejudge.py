#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rejudge sampled candidates with a stricter Gemini L3 schema.

This script is intentionally standalone and does not modify the training path.
It imports reward_server/test_gemini.py for rubric loading, bottom-up aggregation,
soft penalties, calibration, and grid drawing, but replaces the Gemini L3 output
schema with stable facet slots:

    facet_scores["facet_000"] = {facet_id, score, evidence, limitation}

The goal is to prevent duplicate/missing facet IDs and make score-2 harder by
requiring an empty limitation field for fully satisfied facets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml


def load_base_module(repo_root: Path):
    base_path = repo_root / "reward_server" / "test_gemini.py"
    spec = importlib.util.spec_from_file_location("edit_r1_base_test_gemini_schema_v2", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base scorer from {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def retry_forever_enabled(max_retries: int) -> bool:
    return int(max_retries) <= 0


def retry_attempts(max_retries: int):
    attempt = 1
    while retry_forever_enabled(max_retries) or attempt <= int(max_retries):
        yield attempt
        attempt += 1


def has_retry_left(max_retries: int, attempt: int) -> bool:
    return retry_forever_enabled(max_retries) or attempt < int(max_retries)


def retry_sleep_seconds(attempt: int) -> float:
    try:
        base = float(os.getenv("SINGLE_REWARD_RETRY_SLEEP_BASE", "2"))
    except Exception:
        base = 2.0
    try:
        cap = float(os.getenv("SINGLE_REWARD_RETRY_SLEEP_MAX", "60"))
    except Exception:
        cap = 60.0
    return max(0.0, min(cap, base * max(1, attempt)))


def sleep_before_retry(context: str, attempt: int, max_retries: int, error: Optional[Exception]) -> None:
    delay = retry_sleep_seconds(attempt)
    total = "inf" if retry_forever_enabled(max_retries) else str(max_retries)
    print(
        f"[schema_v2][retry] {context} attempt {attempt}/{total} failed: {error!r}; "
        f"retrying in {delay:.1f}s",
        flush=True,
    )
    time.sleep(delay)


def facet_slot_key(index: int) -> str:
    return f"facet_{index:03d}"


def build_response_schema_v2(active_l3: List[str]) -> Dict[str, Any]:
    # Gemini responseJsonSchema rejects some arbitrary property names used by
    # facet ids, especially ids containing dots. Use stable slot keys but pin
    # each slot's facet_id to exactly one active L3 id. This preserves the core
    # guarantee: every active facet appears exactly once, with no duplicate or
    # unexpected facet ids.
    facet_properties: Dict[str, Any] = {}
    required_slots: List[str] = []
    for idx, facet_id in enumerate(active_l3):
        slot = facet_slot_key(idx)
        required_slots.append(slot)
        facet_properties[slot] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "facet_id": {"type": "string", "enum": [facet_id]},
                "score": {"type": "string", "enum": ["0", "1", "2", "N/A"]},
                "evidence": {"type": "string"},
                "limitation": {"type": "string"},
            },
            "required": ["facet_id", "score", "evidence", "limitation"],
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facet_scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": facet_properties,
                "required": required_slots,
            },
            "failure_tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["facet_scores", "failure_tags"],
    }


def convert_schema_for_gemini(value: Any) -> Any:
    # Gemini responseJsonSchema accepts a subset of JSON Schema. Keep this very
    # close to the existing helper but preserve required/properties/enums.
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if key in {"minItems", "maxItems"}:
                continue
            if key == "type":
                out[key] = item.lower() if isinstance(item, str) else item
            elif key == "properties" and isinstance(item, dict):
                out[key] = {str(k): convert_schema_for_gemini(v) for k, v in item.items()}
            else:
                out[key] = convert_schema_for_gemini(item)
        return out
    if isinstance(value, list):
        return [convert_schema_for_gemini(v) for v in value]
    return value


def gemini_generate_url(model: str, base_url: str) -> str:
    base = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    if base.endswith(":generateContent"):
        return base
    model_part = urllib.parse.quote(model.strip(), safe="")
    if base.endswith("/models"):
        return f"{base}/{model_part}:generateContent"
    return f"{base}/models/{model_part}:generateContent"


def extract_gemini_text(response_json: Dict[str, Any]) -> str:
    texts: List[str] = []
    for cand in response_json.get("candidates", []) or []:
        content = cand.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and part.get("text") is not None:
                texts.append(str(part["text"]))
    if not texts:
        raise RuntimeError("Gemini response has no text parts: " + json.dumps(response_json, ensure_ascii=False)[:1000])
    return "\n".join(texts).strip()


def data_url_to_inline_part(data_url: str) -> Dict[str, Any]:
    header, encoded = data_url.split(",", 1)
    mime = "image/jpeg"
    if header.startswith("data:") and ";base64" in header:
        mime = header[len("data:") : header.index(";base64")]
    return {"inlineData": {"mimeType": mime, "data": encoded}}


def validate_and_normalize_object_scores(output: Dict[str, Any], active_l3: List[str]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("Gemini output is not a JSON object")
    facet_scores = output.get("facet_scores")
    if not isinstance(facet_scores, dict):
        raise ValueError("facet_scores must be an object keyed by stable facet slot")

    expected_slots = {facet_slot_key(i) for i in range(len(active_l3))}
    got_slots = set(str(k) for k in facet_scores.keys())
    missing_slots = sorted(expected_slots - got_slots)
    unexpected_slots = sorted(got_slots - expected_slots)
    if missing_slots or unexpected_slots:
        raise ValueError(f"Invalid facet score slots: missing={missing_slots}, unexpected={unexpected_slots}")

    normalized_items: List[Dict[str, Any]] = []
    returned_ids: List[str] = []
    consistency_errors: List[str] = []
    consistency_warnings: List[str] = []
    for idx, facet_id in enumerate(active_l3):
        slot = facet_slot_key(idx)
        item = facet_scores.get(slot)
        if not isinstance(item, dict):
            raise ValueError(f"facet_scores[{slot!r}] is not an object")
        returned_facet_id = str(item.get("facet_id", "")).strip()
        returned_ids.append(returned_facet_id)
        if returned_facet_id != facet_id:
            consistency_errors.append(f"{slot}: expected facet_id {facet_id!r}, got {returned_facet_id!r}")
        score = str(item.get("score", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        limitation = str(item.get("limitation", "")).strip()
        if score not in {"0", "1", "2", "N/A"}:
            raise ValueError(f"Invalid score for {facet_id}: {score!r}")
        if not evidence:
            consistency_warnings.append(f"{facet_id}: evidence is empty")
        if score == "2" and limitation:
            consistency_errors.append(f"{facet_id}: score 2 has non-empty limitation")
        if score == "1" and not limitation:
            consistency_warnings.append(f"{facet_id}: score 1 did not explain why it is not 2")
        if score == "0" and not limitation:
            consistency_warnings.append(f"{facet_id}: score 0 did not explain the failed central requirement")
        if score == "N/A" and not limitation:
            consistency_warnings.append(f"{facet_id}: N/A did not explain why it is inapplicable or unjudgeable")
        normalized_items.append({
            "facet_id": facet_id,
            "score": score,
            "evidence": evidence,
            "limitation": limitation,
        })

    expected_ids = set(active_l3)
    got_ids = set(returned_ids)
    if got_ids != expected_ids or len(returned_ids) != len(set(returned_ids)):
        consistency_errors.append(
            f"facet_id coverage mismatch: missing={sorted(expected_ids - got_ids)}, "
            f"unexpected={sorted(got_ids - expected_ids)}, returned={returned_ids}"
        )
    if consistency_errors:
        raise ValueError("Facet consistency validation failed: " + "; ".join(consistency_errors[:10]))

    tags = output.get("failure_tags")
    if not isinstance(tags, list):
        raise ValueError("failure_tags must be an array")
    return {
        "facet_scores": normalized_items,
        "failure_tags": [str(x) for x in tags],
        "schema_version": "facet_slot_object_v2_limitation",
        "raw_object_facet_scores": facet_scores,
    }


def call_gemini_l3_judge_v2(
    *,
    base: Any,
    model: str,
    api_key: str,
    base_url: str,
    source_image_path: str,
    edited_image_path: str,
    prompt: str,
    rubric: Any,
    requirement: str,
    max_retries: int,
    request_timeout: float,
    max_image_side: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    active_l3 = rubric.active_l3
    rubric_text = base.build_rubric_text(rubric)
    schema = build_response_schema_v2(active_l3)
    official_schema = convert_schema_for_gemini(schema)
    consistency_metrics = base.image_consistency_metrics(source_image_path, edited_image_path)
    source_url = base.image_to_data_url(source_image_path, max_side=max_image_side, quality=jpeg_quality)
    edited_url = base.image_to_data_url(edited_image_path, max_side=max_image_side, quality=jpeg_quality)

    system_prompt = (
        "You are a strict expert evaluator for fine-grained human image editing. "
        "You compare the source image and the edited image under the user's edit instruction. "
        "Score only the listed active L3 rubric facets. Do not output an overall score, L1 score, or L2 score. "
        "Use only visible evidence. Be strict about identity drift, non-target changes, global color shifts, "
        "texture loss, blur, over-smoothing, leakage, and artifacts."
    )

    scoring_policy = """
Global scoring policy for every facet:

2 = Fully satisfied.
Give 2 only when the requirement is clearly and completely satisfied, with no meaningful visible defect. A result that is merely acceptable, generally correct, or mostly compliant must not receive 2.

1 = Partially satisfied.
Give 1 when the requested property is present but incomplete, inaccurate, weak, excessive, unnatural, inconsistent, or accompanied by a minor visible defect.

0 = Not satisfied.
Give 0 when the requested property is absent, clearly incorrect, severely distorted, contradictory to the instruction, or when a major non-target modification directly violates this facet.

N/A = Not applicable or not judgeable.
Use N/A only when the facet genuinely does not apply to the current instruction, or when the relevant visual region is not visible enough to evaluate. Do not use N/A merely because the judgment is uncertain.

Two-stage decision rule for each facet:
Step 1: Determine whether the central requirement is achieved.
Step 2: Check whether any visible defect prevents full satisfaction.
Step 3: Assign 0 if the central requirement is not achieved; 1 if achieved but any meaningful defect remains; 2 only if achieved completely and no meaningful defect remains; N/A only if genuinely inapplicable or unjudgeable.

Important calibration rule:
Do not interpret 2 as acceptable or mostly correct. A score of 2 means nearly ideal compliance for that specific facet. When deciding between 1 and 2, choose 1 unless the evidence clearly supports full satisfaction. When deciding between 0 and 1, choose 0 if the central requirement of the facet is not achieved.

Evidence and limitation requirements:
- evidence must identify visible observations, not generic statements such as "looks good".
- limitation must be empty only when score is 2.
- limitation must explain the defect for score 1, the failed central requirement for score 0, or the inapplicability/unjudgeability reason for N/A.
""".strip()

    user_text = f"""
# Task
task_key: {rubric.task_key}
rubric_version: {rubric.version}
schema_version: facet_object_v2_limitation

# Edit instruction
{prompt}

# Additional requirement
{requirement or "None"}

# Rubric judge instructions
{judge_instruction_block}

# Objective image consistency hints
These values compare the source image and edited image over the full image. They are not the final score, but they should guide preservation facets. Large absolute changes usually mean global color, brightness, contrast, or quality drift outside the target edit.
{base.format_consistency_metrics(consistency_metrics)}

# Global scoring policy
{scoring_policy}

# Rubric-specific judge instructions
{rubric.judge_instructions or "None"}

# Rubric source audit
rubric_source_path: {rubric.source_path}
rubric_task_key: {rubric.task_key}
rubric_version: {rubric.version}
rubric_checklist_sha256: {hashlib.sha256(rubric_text.encode("utf-8", errors="replace")).hexdigest()}

# Active L3 facet ids
{json.dumps(active_l3, ensure_ascii=False)}

# L1/L2/L3 rubric checklist actually used for this judgment
{rubric_text}

# Exact JSON schema
{json.dumps(schema, ensure_ascii=False)}

Return only JSON matching the schema. The facet_scores object must contain every required facet_### slot exactly once. Each slot's facet_id is fixed by the schema and must match the enum value exactly.
""".strip()

    use_response_schema = os.getenv("GEMINI_SCHEMA_V2_USE_RESPONSE_SCHEMA", "0").strip().lower() in {"1", "true", "yes", "y"}
    generation_config = {
        "temperature": 0,
        "candidateCount": 1,
        "maxOutputTokens": int(os.getenv("GEMINI_NATIVE_MAX_OUTPUT_TOKENS", "8192")),
        "responseMimeType": "application/json",
    }
    if use_response_schema:
        generation_config["responseJsonSchema"] = official_schema

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_text},
                {"text": "Source image before editing:"},
                data_url_to_inline_part(source_url),
                {"text": "Edited image generated by the model:"},
                data_url_to_inline_part(edited_url),
            ],
        }],
        "generationConfig": generation_config,
    }

    url = gemini_generate_url(model, base_url)
    last_err: Optional[Exception] = None
    for attempt in retry_attempts(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8", "x-goog-api-key": api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
            output_text = extract_gemini_text(response_json)
            parsed = base.parse_json_object(output_text)
            normalized = validate_and_normalize_object_scores(parsed, active_l3)
            normalized["raw_model_response_text"] = output_text
            normalized["gemini_response_json"] = response_json
            return normalized
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")[:4000]
            last_err = RuntimeError(f"Gemini HTTP {err.code}: {body}")
            if err.code in {400, 401, 403, 404} and not retry_forever_enabled(max_retries):
                raise last_err
        except Exception as err:
            last_err = err
        if has_retry_left(max_retries, attempt):
            sleep_before_retry("Gemini schema-v2 judge", attempt, max_retries, last_err)
    raise RuntimeError(f"Gemini schema-v2 judge failed after {max_retries} attempts: {last_err!r}")


def font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ])
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int, max_lines: int = 3) -> List[str]:
    words = str(text).replace("\n", " ").split()
    lines: List[str] = []
    cur = ""
    for word in words:
        cand = word if not cur else cur + " " + word
        try:
            w = draw.textbbox((0, 0), cand, font=fnt)[2]
        except Exception:
            w = len(cand) * 7
        if w <= width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines[:max_lines]


def save_schema_v2_grid(source_path: str, rows: List[Dict[str, Any]], prompt: str, output_path: Path) -> None:
    ordered = sorted(rows, key=lambda r: int(r.get("sample_index", 0)))
    ranked = sorted(ordered, key=lambda r: float(r.get("reward", 0.0)), reverse=True)
    rank_by_idx = {int(r.get("sample_index", 0)): i for i, r in enumerate(ranked, start=1)}
    tile_w, image_h, label_h, cols = 420, 360, 210, 3
    cell_h = image_h + label_h
    canvas = Image.new("RGB", (cols * tile_w, 3 * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(18, True)
    text_font = font(12, False)

    def panel(cell_idx: int, image_path: str, title: str, lines: List[str], border: Tuple[int, int, int]) -> None:
        row, col = divmod(cell_idx, cols)
        x, y = col * tile_w, row * cell_h
        draw.rectangle((x, y, x + tile_w - 1, y + cell_h - 1), outline=(215, 215, 215), width=1)
        try:
            im = Image.open(image_path).convert("RGB")
            im = ImageOps.contain(im, (tile_w - 24, image_h - 24), method=Image.Resampling.LANCZOS)
            canvas.paste(im, (x + (tile_w - im.width) // 2, y + (image_h - im.height) // 2))
        except Exception as err:
            draw.text((x + 12, y + 12), f"image error: {err}", fill=(180, 0, 0), font=text_font)
        draw.rectangle((x + 4, y + 4, x + tile_w - 5, y + image_h - 5), outline=border, width=4)
        ty = y + image_h + 8
        draw.text((x + 12, ty), title, fill=(0, 0, 0), font=title_font)
        ty += 24
        for line in lines:
            for part in wrap(draw, line, text_font, tile_w - 24, 2):
                draw.text((x + 12, ty), part, fill=(35, 35, 35), font=text_font)
                ty += 16
                if ty > y + cell_h - 16:
                    return

    panel(0, source_path, "SOURCE", ["schema_v2 object-key rubric", f"prompt: {prompt}"], (65, 115, 210))
    for idx, row in enumerate(ordered, start=1):
        sample_idx = int(row.get("sample_index", idx))
        reward = float(row.get("reward", 0.0))
        rank = rank_by_idx.get(sample_idx, idx)
        border = (40, 150, 75) if rank == 1 else (210, 80, 65) if rank == len(ordered) else (150, 150, 150)
        l1 = row.get("l1_scores") or {}
        l1_line = "L1: " + ", ".join(f"{k}={float(v):.2f}" for k, v in list(l1.items())[:3] if isinstance(v, (int, float)))
        counts = row.get("label_counts", {})
        label_order = [label for label in [str(i) for i in range(10)] + ["N/A"] if label in counts or label in {"0", "1", "2"}]
        counts_line = "labels: " + ", ".join(f"{k}={counts.get(k, 0)}" for k in label_order)
        lines = [
            f"rank={rank} reward={reward:.4f}",
            counts_line,
            l1_line,
            f"reason: {row.get('reason', '')}",
        ]
        panel(idx, str(row.get("sample_path")), f"C{sample_idx:02d} reward={reward:.3f}", lines, border)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def sample_paths_from_dir(data_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for pattern in ["sample_*.png", "sample_*.jpg", "sample_*.jpeg", "candidate_*.png", "candidate_*.jpg", "candidate_*.jpeg"]:
        paths.extend(sorted(data_dir.glob(pattern)))
    if not paths:
        paths = sorted(
            p for p in data_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and p.stem.lower() not in {"source", "input", "original"}
        )
    return paths


def source_path_from_dir(data_dir: Path) -> Path:
    for name in ["source.jpg", "source.png", "source.jpeg", "source.webp", "input.jpg", "input.png"]:
        path = data_dir / name
        if path.is_file():
            return path
    candidates = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and p.stem.lower().startswith("source")
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No source image found in {data_dir}")


def load_v13_dynamic_rubric(path: str, prompt: str) -> Dict[str, Any]:
    rubric_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Rubric YAML must be a mapping: {rubric_path}")

    taxonomy = raw.get("taxonomy") or {}
    if not isinstance(taxonomy, dict) or not taxonomy:
        raise ValueError("v13 rubric requires non-empty taxonomy")

    facets: Dict[str, Dict[str, str]] = {}
    for l1_name, l2_map in taxonomy.items():
        if not isinstance(l2_map, dict):
            continue
        for l2_name, l3_map in l2_map.items():
            if not isinstance(l3_map, dict):
                continue
            for facet_id, spec in l3_map.items():
                if not isinstance(spec, dict):
                    continue
                facets[str(facet_id)] = {
                    "l1": str(l1_name),
                    "l2": str(l2_name),
                    "title": str(spec.get("title", facet_id)),
                    "rubric": str(spec.get("rubric", "")),
                }
    if not facets:
        raise ValueError("v13 rubric taxonomy contains no L3 facets")

    activation = raw.get("activation") or {}
    active_l3: List[str] = []

    def add(facet_id: str) -> None:
        if facet_id in facets and facet_id not in active_l3:
            active_l3.append(facet_id)

    for facet_id in activation.get("always_active_l3") or []:
        add(str(facet_id))

    prompt_lower = prompt.lower()
    intent_keywords = {
        "style_texture": [
            "hair style", "hairstyle", "style", "texture", "curl", "curly", "curls", "woolly",
            "coily", "ringlet", "wavy", "wave", "straight", "sleek", "messy", "pixie", "bob",
            "layered", "braid", "braids",
        ],
        "color_placement": [
            "color", "colour", "dye", "pink", "pastel", "blonde", "black", "brown", "red",
            "blue", "green", "purple", "silver", "gray", "grey", "highlight", "highlights",
            "ombre", "gradient", "streak", "streaks", "split dye",
        ],
        "length_volume": [
            "short", "long", "length", "volume", "voluminous", "dense", "fluffy", "thick",
            "thin", "silhouette", "ends", "outward",
        ],
        "structure_front_detail": [
            "bang", "bangs", "fringe", "parting", "part", "hairline", "forehead", "sideburn",
            "sideburns", "face-framing", "face framing", "baby hair", "ends flipped",
            "ponytail", "bun", "updo", "tucked", "scalp coverage",
        ],
        "accessory": [
            "clip", "ribbon", "flower", "headband", "hairpin", "bead", "tie", "decoration",
            "hair accessory", "hair accessories",
        ],
    }
    conditional = activation.get("conditional_l3_by_intent") or {}
    activated_intents: List[str] = []
    for intent, keywords in intent_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            activated_intents.append(intent)
            for facet_id in conditional.get(intent) or []:
                add(str(facet_id))

    requested_facets = [facet_id for facet_id in raw.get("requested_facets") or [] if facet_id in active_l3]
    score_mapping = {str(k): float(v) for k, v in (raw.get("score_mapping") or {}).items()}
    for label in ["0", "1", "2", "3", "4"]:
        if label not in score_mapping:
            raise ValueError(f"v13 score_mapping missing label {label!r}")

    return {
        "source_path": str(rubric_path),
        "task_key": str(raw.get("task_key", "")),
        "version": str(raw.get("version", "")),
        "raw": raw,
        "taxonomy": taxonomy,
        "facets": facets,
        "active_l3": active_l3,
        "requested_facets": requested_facets,
        "activated_intents": activated_intents,
        "score_mapping": score_mapping,
        "l1_weights": {str(k): float(v) for k, v in (raw.get("l1_weights") or {}).items()},
        "facet_weights": {str(k): float(v) for k, v in (raw.get("facet_weights") or {}).items()},
    }


def build_response_schema_v13(active_l3: List[str]) -> Dict[str, Any]:
    facet_properties: Dict[str, Any] = {}
    required_slots: List[str] = []
    for idx, facet_id in enumerate(active_l3):
        slot = facet_slot_key(idx)
        required_slots.append(slot)
        facet_properties[slot] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "facet_id": {"type": "string", "enum": [facet_id]},
                "score": {"type": "string", "enum": ["0", "1", "2", "3", "4"]},
                "evidence": {"type": "string"},
                "defect": {"type": "string"},
            },
            "required": ["facet_id", "score", "evidence", "defect"],
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facet_scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": facet_properties,
                "required": required_slots,
            },
            "failure_tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["facet_scores", "failure_tags"],
    }


def validate_v13_output(output: Dict[str, Any], active_l3: List[str]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("Gemini output is not a JSON object")
    facet_scores = output.get("facet_scores")
    if not isinstance(facet_scores, dict):
        raise ValueError("facet_scores must be an object keyed by stable facet slot")
    expected_slots = {facet_slot_key(i) for i in range(len(active_l3))}
    got_slots = set(str(k) for k in facet_scores.keys())
    if expected_slots != got_slots:
        raise ValueError(f"Invalid facet slots: missing={sorted(expected_slots - got_slots)}, unexpected={sorted(got_slots - expected_slots)}")

    normalized: List[Dict[str, Any]] = []
    warnings: List[str] = []
    returned_ids: List[str] = []
    for idx, expected_facet_id in enumerate(active_l3):
        slot = facet_slot_key(idx)
        item = facet_scores.get(slot)
        if not isinstance(item, dict):
            raise ValueError(f"facet_scores[{slot!r}] is not an object")
        facet_id = str(item.get("facet_id", "")).strip()
        score = str(item.get("score", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        defect = str(item.get("defect", "")).strip()
        returned_ids.append(facet_id)
        if facet_id != expected_facet_id:
            raise ValueError(f"{slot}: expected facet_id={expected_facet_id!r}, got {facet_id!r}")
        if score not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"{facet_id}: invalid score {score!r}")
        if not evidence:
            warnings.append(f"{facet_id}: empty evidence")
        if score == "4" and defect:
            raise ValueError(f"{facet_id}: score 4 cannot contain defect")
        if score != "4" and not defect:
            warnings.append(f"{facet_id}: score {score} should include the most important defect")
        normalized.append({"facet_id": facet_id, "score": score, "evidence": evidence, "defect": defect})
    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError(f"Duplicate facet ids returned: {returned_ids}")
    tags = output.get("failure_tags")
    if not isinstance(tags, list):
        raise ValueError("failure_tags must be an array")
    return {
        "facet_scores": normalized,
        "failure_tags": [str(x) for x in tags],
        "schema_version": "facet_slot_object_v13_5level_defect",
        "validation_warnings": warnings,
        "raw_object_facet_scores": facet_scores,
    }


def call_gemini_l3_judge_v13(
    *,
    base: Any,
    model: str,
    api_key: str,
    base_url: str,
    source_image_path: str,
    edited_image_path: str,
    prompt: str,
    rubric: Dict[str, Any],
    requirement: str,
    max_retries: int,
    request_timeout: float,
    max_image_side: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    active_l3 = rubric["active_l3"]
    facets = rubric["facets"]
    schema = build_response_schema_v13(active_l3)
    official_schema = convert_schema_for_gemini(schema)
    metrics = base.image_consistency_metrics(source_image_path, edited_image_path)
    source_url = base.image_to_data_url(source_image_path, max_side=max_image_side, quality=jpeg_quality)
    edited_url = base.image_to_data_url(edited_image_path, max_side=max_image_side, quality=jpeg_quality)
    raw = rubric["raw"]
    active_specs = {
        facet_id: {
            "l1": facets[facet_id]["l1"],
            "l2": facets[facet_id]["l2"],
            "title": facets[facet_id]["title"],
            "rubric": facets[facet_id]["rubric"],
            "weight": rubric["facet_weights"].get(facet_id, 1.0),
        }
        for facet_id in active_l3
    }
    system_prompt = (
        "You are a strict expert evaluator for fine-grained human hair editing. "
        "Score only the active L3 facets using the five-level 0/1/2/3/4 policy. "
        "Do not output an overall score. Use visible evidence from the source and edited image."
    )
    user_text = f"""
# Task
task_key: {rubric['task_key']}
rubric_version: {rubric['version']}
schema_version: facet_slot_object_v13_5level_defect

# Edit instruction
{prompt}

# Additional requirement
{requirement or "None"}

# Objective image consistency hints
These values compare source and edited image over the full image. They are auxiliary hints only. Do not replace visual rubric judgment with them.
{base.format_consistency_metrics(metrics)}

# Global scoring policy
{json.dumps(raw.get('global_scoring_policy') or {}, ensure_ascii=False, indent=2)}

# Active intent detection used by the scorer
activated_intents: {json.dumps(rubric['activated_intents'], ensure_ascii=False)}
active_l3: {json.dumps(active_l3, ensure_ascii=False)}

# Active L3 facet checklist
{json.dumps(active_specs, ensure_ascii=False, indent=2)}

# Failure tags enum
{json.dumps(raw.get('failure_tags_enum') or [], ensure_ascii=False)}

# Exact JSON schema
{json.dumps(schema, ensure_ascii=False)}

Return only JSON matching the schema. The facet_scores object must contain every required facet_### slot exactly once. Each score must be one of "0", "1", "2", "3", "4". Use "4" only when the facet is nearly ideal; for scores 0-3, defect must explain why the score is not higher.
""".strip()

    use_response_schema = os.getenv("GEMINI_SCHEMA_V13_USE_RESPONSE_SCHEMA", "0").strip().lower() in {"1", "true", "yes", "y"}
    generation_config = {
        "temperature": 0,
        "candidateCount": 1,
        "maxOutputTokens": int(os.getenv("GEMINI_NATIVE_MAX_OUTPUT_TOKENS", "32768")),
        "responseMimeType": "application/json",
    }
    thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
    if thinking_budget:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    if use_response_schema:
        generation_config["responseJsonSchema"] = official_schema

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_text},
                {"text": "Source image before editing:"},
                data_url_to_inline_part(source_url),
                {"text": "Edited candidate image:"},
                data_url_to_inline_part(edited_url),
            ],
        }],
        "generationConfig": generation_config,
    }
    url = gemini_generate_url(model, base_url)
    last_err: Optional[Exception] = None
    for attempt in retry_attempts(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8", "x-goog-api-key": api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
            output_text = extract_gemini_text(response_json)
            parsed = base.parse_json_object(output_text)
            normalized = validate_v13_output(parsed, active_l3)
            normalized["raw_model_response_text"] = output_text
            normalized["gemini_response_json"] = response_json
            return normalized
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")[:4000]
            last_err = RuntimeError(f"Gemini HTTP {err.code}: {body}")
            if err.code in {400, 401, 403, 404} and not retry_forever_enabled(max_retries):
                raise last_err
        except Exception as err:
            last_err = err
        if has_retry_left(max_retries, attempt):
            sleep_before_retry("Gemini v13 judge", attempt, max_retries, last_err)
    raise RuntimeError(f"Gemini v13 judge failed after {max_retries} attempts: {last_err!r}")


def _weighted_mean(values: List[Tuple[float, float]]) -> float:
    numerator = sum(value * weight for value, weight in values)
    denominator = sum(weight for _, weight in values)
    return numerator / denominator if denominator > 0 else 0.0


def aggregate_v13(base: Any, rubric: Dict[str, Any], judge_output: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    raw_by_id = {item["facet_id"]: item for item in judge_output["facet_scores"]}
    score_mapping = rubric["score_mapping"]
    facets = rubric["facets"]
    facet_weights = rubric["facet_weights"]
    raw_cfg = rubric["raw"]

    l3_scores: Dict[str, float] = {}
    l3_raw_labels: Dict[str, str] = {}
    l3_evidence: Dict[str, str] = {}
    l3_defects: Dict[str, str] = {}
    for facet_id in rubric["active_l3"]:
        item = raw_by_id[facet_id]
        label = item["score"]
        l3_raw_labels[facet_id] = label
        l3_scores[facet_id] = float(score_mapping[label])
        l3_evidence[facet_id] = item.get("evidence", "")
        l3_defects[facet_id] = item.get("defect", "")

    l2_values: Dict[str, Dict[str, List[float]]] = {}
    l1_facet_values: Dict[str, List[Tuple[float, float]]] = {}
    for facet_id, score in l3_scores.items():
        spec = facets[facet_id]
        l1, l2 = spec["l1"], spec["l2"]
        l2_values.setdefault(l1, {}).setdefault(l2, []).append(score)
        l1_facet_values.setdefault(l1, []).append((score, facet_weights.get(facet_id, 1.0)))

    l2_scores = {
        l1: {l2: sum(values) / len(values) for l2, values in l2_map.items()}
        for l1, l2_map in l2_values.items()
    }

    within = (raw_cfg.get("aggregation") or {}).get("within_l1") or {}
    mean_weight = float(within.get("mean_weight", 0.70))
    tail_weight = float(within.get("lower_tail_weight", 0.30))
    tail_count = int(within.get("lower_tail_count", 2))
    l1_scores: Dict[str, float] = {}
    for l1, values in l1_facet_values.items():
        weighted = _weighted_mean(values)
        sorted_scores = sorted(value for value, _ in values)
        lower_tail = sum(sorted_scores[:max(1, min(tail_count, len(sorted_scores)))]) / max(1, min(tail_count, len(sorted_scores)))
        l1_scores[l1] = mean_weight * weighted + tail_weight * lower_tail

    final_cfg = ((raw_cfg.get("aggregation") or {}).get("final") or {})
    l1_weight_map = rubric["l1_weights"]
    active_l1_values = [(score, l1_weight_map.get(l1, 1.0)) for l1, score in l1_scores.items()]
    arithmetic = _weighted_mean(active_l1_values)
    floor = float(final_cfg.get("geometric_floor", 0.03))
    denom = sum(weight for _, weight in active_l1_values)
    geometric = math.exp(sum(weight * math.log(max(floor, score)) for score, weight in active_l1_values) / denom) if denom > 0 else 0.0
    arithmetic_weight = float(final_cfg.get("arithmetic_weight", 0.65))
    geometric_weight = float(final_cfg.get("geometric_weight", 0.35))
    reward_before_gate = arithmetic_weight * arithmetic + geometric_weight * geometric

    request_ids = []
    if "primary_edit_match" in l3_scores:
        request_ids.append("primary_edit_match")
    request_ids.extend(fid for fid in rubric["requested_facets"] if fid in l3_scores and fid not in request_ids)
    request_score = min((l3_scores[fid] for fid in request_ids), default=1.0)
    gate_multiplier = 0.35 + 0.65 * request_score
    reward_before_caps = reward_before_gate * gate_multiplier
    reward = reward_before_caps

    caps_applied: List[Dict[str, Any]] = []
    caps = raw_cfg.get("score_caps") or {}
    for facet_id, label in l3_raw_labels.items():
        cap_sources: List[str] = []
        if facet_id in caps:
            cap_sources.append(facet_id)
        if facet_id in rubric["requested_facets"] and "active_requested_facets" in caps:
            cap_sources.append("active_requested_facets")
        for cap_source in cap_sources:
            cap_value = (caps.get(cap_source) or {}).get(label)
            if cap_value is None:
                continue
            cap_float = float(cap_value)
            if reward > cap_float:
                caps_applied.append({"facet_id": facet_id, "label": label, "cap_source": cap_source, "cap": cap_float, "before": reward})
                reward = min(reward, cap_float)

    objective = raw_cfg.get("objective_auxiliary") or {}
    metric_penalty, metric_quality_score, metric_reasons = base.metric_color_consistency_penalty(metrics)
    objective_delta = 0.0
    if objective.get("enabled", False):
        blend_weight = float(objective.get("blend_weight", 0.10))
        max_effect = float(objective.get("max_absolute_effect", 0.04))
        objective_target = metric_quality_score
        objective_delta = max(-max_effect, min(max_effect, blend_weight * (objective_target - reward)))
        reward = reward + objective_delta

    reward = max(0.0, min(1.0, reward))
    return {
        "reward": float(reward),
        "reward_raw_bottom_up": float(reward_before_gate),
        "reward_before_gate": float(reward_before_gate),
        "reward_before_caps": float(reward_before_caps),
        "reward_before_objective_aux": float(reward - objective_delta),
        "reward_for_training": float(reward),
        "l1_scores": l1_scores,
        "l2_scores": l2_scores,
        "l3_scores": l3_scores,
        "l3_raw_labels": l3_raw_labels,
        "l3_evidence": l3_evidence,
        "l3_defects": l3_defects,
        "failure_tags": judge_output.get("failure_tags", []),
        "raw_judge": judge_output,
        "aggregation_details": {
            "arithmetic_l1": arithmetic,
            "geometric_l1": geometric,
            "request_ids": request_ids,
            "request_score": request_score,
            "gate_multiplier": gate_multiplier,
            "caps_applied": caps_applied,
            "objective_delta": objective_delta,
            "metric_quality_score": metric_quality_score,
            "metric_color_penalty": metric_penalty,
            "metric_color_penalty_reasons": metric_reasons,
        },
        "metric_color_penalty": metric_penalty,
        "metric_color_quality_score": metric_quality_score,
        "metric_color_penalty_reasons": metric_reasons,
    }


def build_v13_reason(result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    labels = result.get("l3_raw_labels") or {}
    defects = result.get("l3_defects") or {}
    weakest = sorted((float(result["l3_scores"][fid]), fid) for fid in labels if fid in result.get("l3_scores", {}))[:3]
    weak_bits = [f"{fid}={labels.get(fid)}" for _, fid in weakest]
    defect_bits = [defects.get(fid, "") for _, fid in weakest if defects.get(fid)]
    details = {
        "weakest_facets": weak_bits,
        "weakest_defects": defect_bits,
        "aggregation_details": result.get("aggregation_details", {}),
    }
    reason = "weakest: " + ", ".join(weak_bits)
    if defect_bits:
        reason += "; " + defect_bits[0]
    return reason, details


def load_v18_integer_rubric(path: str) -> Dict[str, Any]:
    rubric_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Rubric YAML must be a mapping: {rubric_path}")
    active_l3 = [str(x) for x in (raw.get("active_l3") or [])]
    if not active_l3:
        raise ValueError("v18 integer rubric requires non-empty active_l3")
    taxonomy = raw.get("taxonomy") or {}
    facets: Dict[str, Dict[str, Any]] = {}
    for l1_name, l2_map in taxonomy.items():
        if not isinstance(l2_map, dict):
            continue
        for l2_name, l3_map in l2_map.items():
            if not isinstance(l3_map, dict):
                continue
            for facet_id, spec in l3_map.items():
                if not isinstance(spec, dict):
                    continue
                facets[str(facet_id)] = {
                    "l1": str(l1_name),
                    "l2": str(l2_name),
                    "title": str(spec.get("title", facet_id)),
                    "definition": str(spec.get("definition", spec.get("rubric", ""))),
                    "anchors": spec.get("anchors") or {},
                }
    missing = [facet_id for facet_id in active_l3 if facet_id not in facets]
    if missing:
        raise ValueError(f"v18 active_l3 contains undefined facets: {missing}")
    l1_weights = {str(k): float(v) for k, v in (raw.get("l1_weights") or {}).items()}
    facet_weights_within_l1 = {
        str(l1): {str(fid): float(w) for fid, w in (items or {}).items()}
        for l1, items in (raw.get("facet_weights_within_l1") or {}).items()
    }
    return {
        "source_path": str(rubric_path),
        "task_key": str(raw.get("task_key", "")),
        "version": str(raw.get("version", "")),
        "raw": raw,
        "taxonomy": taxonomy,
        "facets": facets,
        "active_l3": active_l3,
        "l1_weights": l1_weights,
        "facet_weights_within_l1": facet_weights_within_l1,
    }


def build_response_schema_v18(num_facets: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score_vector": {
                "type": "array",
                "minItems": num_facets,
                "maxItems": num_facets,
                "items": {"type": "integer", "enum": list(range(10))},
            },
            "failure_tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["score_vector", "failure_tags"],
    }


def validate_v18_output(output: Dict[str, Any], active_l3: List[str]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("Gemini output is not a JSON object")
    vector = output.get("score_vector")
    if not isinstance(vector, list):
        raise ValueError("score_vector must be an array")
    if len(vector) != len(active_l3):
        raise ValueError(f"score_vector length mismatch: expected {len(active_l3)}, got {len(vector)}")
    normalized: List[Dict[str, Any]] = []
    for idx, (facet_id, score_raw) in enumerate(zip(active_l3, vector)):
        if isinstance(score_raw, bool):
            raise ValueError(f"score_vector[{idx}] must be integer 0-9, got bool")
        try:
            score = int(score_raw)
        except Exception as err:
            raise ValueError(f"score_vector[{idx}] is not an integer: {score_raw!r}") from err
        if score < 0 or score > 9:
            raise ValueError(f"score_vector[{idx}]={score} outside 0-9")
        normalized.append({
            "facet_id": facet_id,
            "score": str(score),
            "score_int": score,
            "evidence": "",
            "defect": "",
        })
    tags = output.get("failure_tags")
    if not isinstance(tags, list):
        raise ValueError("failure_tags must be an array")
    return {
        "facet_scores": normalized,
        "score_vector": [item["score_int"] for item in normalized],
        "failure_tags": [str(x) for x in tags],
        "schema_version": "score_vector_v18_integer_0_9",
    }


def parse_v18_output_text(base: Any, output_text: str, active_l3: List[str]) -> Dict[str, Any]:
    try:
        return base.parse_json_object(output_text)
    except Exception:
        pass
    import re

    text = output_text.strip()
    match = re.search(r'"?score_vector"?\s*[:=]\s*\[([^\]]+)\]', text, re.S)
    vector_text = match.group(1) if match else ""
    if not vector_text:
        first_array = re.search(r'\[([^\]]+)\]', text, re.S)
        vector_text = first_array.group(1) if first_array else text
    values = [int(x) for x in re.findall(r'(?<!\d)([0-9])(?!\d)', vector_text)]
    if len(values) != len(active_l3):
        raise ValueError(
            f"Could not recover v18 score_vector of length {len(active_l3)} from model text; "
            f"found {len(values)} integers. text={output_text[:1000]!r}"
        )
    return {"score_vector": values, "failure_tags": ["parsed_from_non_strict_json"]}


def call_gemini_l3_judge_v18(
    *,
    base: Any,
    model: str,
    api_key: str,
    base_url: str,
    source_image_path: str,
    edited_image_path: str,
    prompt: str,
    rubric: Dict[str, Any],
    requirement: str,
    max_retries: int,
    request_timeout: float,
    max_image_side: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    active_l3 = rubric["active_l3"]
    facets = rubric["facets"]
    schema = build_response_schema_v18(len(active_l3))
    official_schema = convert_schema_for_gemini(schema)
    metrics = base.image_consistency_metrics(source_image_path, edited_image_path)
    source_url = base.image_to_data_url(source_image_path, max_side=max_image_side, quality=jpeg_quality)
    edited_url = base.image_to_data_url(edited_image_path, max_side=max_image_side, quality=jpeg_quality)
    active_specs = []
    for idx, facet_id in enumerate(active_l3):
        spec = facets[facet_id]
        active_specs.append({
            "index": idx,
            "facet_id": facet_id,
            "l1": spec["l1"],
            "l2": spec["l2"],
            "title": spec["title"],
            "definition": spec["definition"],
            "anchors": spec["anchors"],
        })
    # Optional YAML-level role/instructions. Set USE_RUBRIC_JUDGE_TEXT=0 to disable
    # them without editing the rubric or this script.
    use_rubric_judge_text = os.getenv("USE_RUBRIC_JUDGE_TEXT", "1").strip().lower() in {"1", "true", "yes", "y"}
    judge_role = str(rubric["raw"].get("judge_role") or "").strip() if use_rubric_judge_text else ""
    judge_instructions = str(rubric["raw"].get("judge_instructions") or "").strip() if use_rubric_judge_text else ""
    system_prompt = (
        (judge_role + " ") if judge_role else "You are a strict expert evaluator for fine-grained human image editing. "
    ) + (
        "Return only a compact JSON score vector. Score every active facet independently "
        "with one integer from 0 to 9."
    )
    judge_instruction_block = judge_instructions or "None"
    user_text = f"""
# Task
task_key: {rubric['task_key']}
rubric_version: {rubric['version']}
schema_version: score_vector_v18_integer_0_9

# Edit instruction
{prompt}

# Additional requirement
{requirement or "None"}

# Rubric judge instructions
{judge_instruction_block}

# Objective image consistency hints
These values compare source and edited image over the full image. They are auxiliary hints only; do not replace visual rubric judgment with them.
{base.format_consistency_metrics(metrics)}

# Global scoring policy
{json.dumps(rubric['raw'].get('global_scoring_policy') or {}, ensure_ascii=False, indent=2)}

# Active L3 facets in required output order
{json.dumps(active_specs, ensure_ascii=False, indent=2)}

# Exact JSON schema
{json.dumps(schema, ensure_ascii=False)}

Return only JSON. The score_vector must contain exactly {len(active_l3)} integers in the same order as the active L3 facet list above. Do not output decimals, explanations, markdown, or repeated facet names.
""".strip()
    use_response_schema = os.getenv("GEMINI_SCHEMA_V18_USE_RESPONSE_SCHEMA", "0").strip().lower() in {"1", "true", "yes", "y"}
    generation_config = {
        "temperature": 0,
        "candidateCount": 1,
        "maxOutputTokens": int(os.getenv("GEMINI_NATIVE_MAX_OUTPUT_TOKENS", "8192")),
        "responseMimeType": "application/json",
    }
    thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
    if thinking_budget:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    if use_response_schema:
        generation_config["responseJsonSchema"] = official_schema
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_text},
                {"text": "Source image before editing:"},
                data_url_to_inline_part(source_url),
                {"text": "Edited candidate image:"},
                data_url_to_inline_part(edited_url),
            ],
        }],
        "generationConfig": generation_config,
    }
    url = gemini_generate_url(model, base_url)
    last_err: Optional[Exception] = None
    for attempt in retry_attempts(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8", "x-goog-api-key": api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
            output_text = extract_gemini_text(response_json)
            parsed = parse_v18_output_text(base, output_text, active_l3)
            normalized = validate_v18_output(parsed, active_l3)
            normalized["raw_model_response_text"] = output_text
            normalized["gemini_response_json"] = response_json
            return normalized
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")[:4000]
            last_err = RuntimeError(f"Gemini HTTP {err.code}: {body}")
            if err.code in {400, 401, 403, 404} and not retry_forever_enabled(max_retries):
                raise last_err
        except Exception as err:
            last_err = err
        if has_retry_left(max_retries, attempt):
            sleep_before_retry("Gemini v18 integer judge", attempt, max_retries, last_err)
    raise RuntimeError(f"Gemini v18 integer judge failed after {max_retries} attempts: {last_err!r}")


def _prompt_contains_keyword(prompt_lower: str, keyword: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    if len(keyword.split()) > 1 or "-" in keyword:
        return keyword in prompt_lower
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", prompt_lower) is not None


def _extract_target_spec_for_mask(rubric: Dict[str, Any], prompt: str) -> str:
    raw = rubric.get("raw") or {}
    extraction = raw.get("target_spec_extraction") or {}
    if not extraction:
        return prompt
    text = " ".join(str(prompt).split())
    lower = text.lower()
    start_markers = [str(x).strip() for x in (extraction.get("preferred_start_markers") or []) if str(x).strip()]
    best_start = -1
    best_marker = ""
    for marker in start_markers:
        idx = lower.find(marker.lower())
        if idx >= 0 and (best_start < 0 or idx < best_start):
            best_start = idx
            best_marker = marker
    if best_start >= 0:
        text = text[best_start + len(best_marker):].lstrip(" :;,.")
        lower = text.lower()
    stop_markers = [str(x).strip() for x in (extraction.get("stop_markers") or []) if str(x).strip()]
    stop_positions = [lower.find(marker.lower()) for marker in stop_markers if lower.find(marker.lower()) >= 0]
    if stop_positions:
        text = text[:min(stop_positions)].strip(" :;,.")
    generic_rebuild = re.search(r"(?is)\brebuild the bangs,\s*hairline,\s*sideburn or face-framing strands\b.*$", text)
    if generic_rebuild:
        text = text[:generic_rebuild.start()].strip(" :;,.")
    for phrase in extraction.get("remove_exact_or_semantically_equivalent_generic_clauses") or []:
        phrase_text = str(phrase).strip()
        if not phrase_text:
            continue
        text = re.sub(rf"(?i),?\s*(?:and\s+)?{re.escape(phrase_text)}", "", text)
    regex_ref = extraction.get("regex_reference") or {}
    for pattern in regex_ref.get("remove_generic_quality_phrases") or []:
        try:
            text = re.sub(str(pattern), "", text)
        except re.error:
            continue
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+", " ", text).strip(" :;,.")
    return text or prompt


def _rule_entries_to_phrases(entries: Any) -> List[str]:
    phrases: List[str] = []
    if isinstance(entries, dict):
        for key in ["activate_when", "semantic_implications"]:
            for entry in entries.get(key) or []:
                phrases.extend(str(entry).split(","))
    else:
        for entry in entries or []:
            phrases.extend(str(entry).split(","))
    return phrases


def _infer_prompt_applicability_mask(rubric: Dict[str, Any], prompt: str) -> Dict[str, int]:
    raw = rubric.get("raw") or {}
    cfg = raw.get("prompt_applicability") or {}
    prompt_facets = [str(x) for x in (cfg.get("prompt_facet_order") or [])]
    always = {str(x) for x in (cfg.get("always_applicable") or [])}
    rules = cfg.get("activation_rules") or {}
    target_spec = _extract_target_spec_for_mask(rubric, prompt)
    prompt_lower = target_spec.lower()
    mask: Dict[str, int] = {facet_id: 1 if facet_id in always else 0 for facet_id in prompt_facets}
    for facet_id, entries in rules.items():
        if facet_id not in mask:
            continue
        phrases = _rule_entries_to_phrases(entries)
        if any(_prompt_contains_keyword(prompt_lower, phrase) for phrase in phrases):
            mask[facet_id] = 1
    for style_name, facet_ids in (cfg.get("named_style_semantic_mapping") or {}).items():
        if _prompt_contains_keyword(prompt_lower, str(style_name)):
            for facet_id in facet_ids or []:
                if str(facet_id) in mask:
                    mask[str(facet_id)] = 1
    return mask


def _uses_masked_identity_aggregation(rubric: Dict[str, Any]) -> bool:
    raw = rubric.get("raw") or {}
    aggregation_text = json.dumps(raw.get("aggregation") or {}, ensure_ascii=False)
    return bool(raw.get("prompt_applicability")) and "hair_facial_identity_landmark_consistency" in aggregation_text and "0.35" in aggregation_text


def aggregate_v18(rubric: Dict[str, Any], judge_output: Dict[str, Any], prompt: str = "") -> Dict[str, Any]:
    active_l3 = rubric["active_l3"]
    facets = rubric["facets"]
    vector = [int(x) for x in judge_output["score_vector"]]
    l3_scores: Dict[str, float] = {}
    l3_raw_labels: Dict[str, str] = {}
    tree_values: Dict[str, List[Tuple[float, float]]] = {}
    l2_scores: Dict[str, Dict[str, float]] = {}
    for facet_id, score_int in zip(active_l3, vector):
        normalized = score_int / 9.0
        l3_scores[facet_id] = normalized
        l3_raw_labels[facet_id] = str(score_int)
        spec = facets[facet_id]
        l1 = spec["l1"]
        weight = rubric["facet_weights_within_l1"].get(l1, {}).get(facet_id, 1.0)
        tree_values.setdefault(l1, []).append((normalized, weight))
        l2_scores.setdefault(l1, {}).setdefault(spec["l2"], []).append(normalized)
    l2_means = {
        l1: {l2: sum(values) / len(values) for l2, values in l2_map.items()}
        for l1, l2_map in l2_scores.items()
    }
    l1_scores = {l1: _weighted_mean(values) for l1, values in tree_values.items()}
    aggregation_details: Dict[str, Any] = {}
    if _uses_masked_identity_aggregation(rubric):
        prompt_mask = _infer_prompt_applicability_mask(rubric, prompt)
        prompt_l1 = "Prompt Compliance"
        prompt_values: List[Tuple[float, float]] = []
        for facet_id, is_active in prompt_mask.items():
            if not is_active or facet_id not in l3_scores:
                continue
            weight = rubric["facet_weights_within_l1"].get(prompt_l1, {}).get(facet_id, 1.0)
            prompt_values.append((l3_scores[facet_id], weight))
        if prompt_values:
            l1_scores[prompt_l1] = _weighted_mean(prompt_values)
        base_reward = _weighted_mean([
            (score, rubric["l1_weights"].get(l1, 1.0))
            for l1, score in l1_scores.items()
        ])
        identity_score = float(l3_scores.get("hair_facial_identity_landmark_consistency", 1.0))
        identity_power = 0.35
        reward = base_reward * (max(0.0, identity_score) ** identity_power)
        aggregation_details = {
            "masked_prompt_compliance": True,
            "prompt_applicability_mask": prompt_mask,
            "target_spec_for_mask": _extract_target_spec_for_mask(rubric, prompt),
            "active_prompt_facets": [facet_id for facet_id, is_active in prompt_mask.items() if is_active],
            "base_reward_before_identity_multiplier": base_reward,
            "identity_score": identity_score,
            "identity_power": identity_power,
            "identity_multiplier": max(0.0, identity_score) ** identity_power,
        }
    else:
        reward = _weighted_mean([
            (score, rubric["l1_weights"].get(l1, 1.0))
            for l1, score in l1_scores.items()
        ])
    return {
        "reward": round(float(reward), 3),
        "reward_raw_bottom_up": float(reward),
        "reward_for_training": float(reward),
        "display_score_0_to_9": round(9.0 * reward, 3),
        "l1_scores": {k: round(float(v), 3) for k, v in l1_scores.items()},
        "l2_scores": {l1: {l2: round(float(v), 3) for l2, v in l2_map.items()} for l1, l2_map in l2_means.items()},
        "l3_scores": {k: round(float(v), 3) for k, v in l3_scores.items()},
        "l3_raw_labels": l3_raw_labels,
        "l3_evidence": {},
        "failure_tags": judge_output.get("failure_tags", []),
        "raw_judge": judge_output,
        "aggregation_details": aggregation_details,
    }


def build_v18_reason(result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    labels = result.get("l3_raw_labels") or {}
    scores = result.get("l3_scores") or {}
    details_in = result.get("aggregation_details") or {}
    prompt_mask = details_in.get("prompt_applicability_mask") or {}
    inactive_prompt_facets = {fid for fid, is_active in prompt_mask.items() if not is_active}
    reportable_scores = {
        fid: score
        for fid, score in scores.items()
        if fid not in inactive_prompt_facets
    }
    weakest = sorted((float(reportable_scores[fid]), fid) for fid in reportable_scores)[:4]
    weak_bits = [f"{fid}={labels.get(fid)}" for _, fid in weakest]
    details = {
        "score_vector": [int(labels[fid]) for fid in labels],
        "weakest_facets": weak_bits,
        "display_score_0_to_9": result.get("display_score_0_to_9"),
        "target_spec_for_mask": details_in.get("target_spec_for_mask", ""),
        "active_prompt_facets": details_in.get("active_prompt_facets", []),
        "inactive_prompt_facets_excluded_from_reason": sorted(inactive_prompt_facets),
    }
    return "weakest: " + ", ".join(weak_bits), details


def _flatten_v22_facets(taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    facets: Dict[str, Dict[str, Any]] = {}
    for l1_name, maybe_facets in taxonomy.items():
        if not isinstance(maybe_facets, dict):
            continue
        for key, value in maybe_facets.items():
            if isinstance(value, dict) and ("title" in value or "definition" in value or "anchors" in value):
                facets[str(key)] = {
                    "l1": str(l1_name),
                    "l2": str(l1_name),
                    "title": str(value.get("title", key)),
                    "definition": str(value.get("definition", value.get("rubric", ""))),
                    "anchors": value.get("anchors") or {},
                }
            elif isinstance(value, dict):
                for facet_id, spec in value.items():
                    if isinstance(spec, dict):
                        facets[str(facet_id)] = {
                            "l1": str(l1_name),
                            "l2": str(key),
                            "title": str(spec.get("title", facet_id)),
                            "definition": str(spec.get("definition", spec.get("rubric", ""))),
                            "anchors": spec.get("anchors") or {},
                        }
    return facets


def load_v22_dynamic_integer_rubric(path: str, prompt: str) -> Dict[str, Any]:
    rubric_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Rubric YAML must be a mapping: {rubric_path}")
    facets = _flatten_v22_facets(raw.get("taxonomy") or {})
    request_order = [str(x) for x in (raw.get("request_facet_order") or [])]
    quality_order = [str(x) for x in (raw.get("quality_facet_order") or [])]
    prompt_lower = prompt.lower()
    keyword_map = {
        "requested_structure_arrangement_match": [
            "hairstyle", "hair style", "structure", "arrangement", "loose", "ponytail", "bun",
            "braid", "bob", "pixie", "buzz", "mohawk", "undercut", "slick-back", "feminine",
            "masculine", "styling",
        ],
        "requested_length_layer_silhouette_volume_match": [
            "short", "long", "length", "layer", "layers", "silhouette", "volume", "fluffy",
            "dense", "density", "ends", "outward", "inward", "compact", "wide", "top height",
        ],
        "requested_texture_curl_flow_match": [
            "curl", "curls", "curly", "woolly", "coily", "springy", "wave", "wavy", "straight",
            "sleek", "messy", "strand flow", "flow", "texture", "strand", "strands",
        ],
        "requested_front_hair_hairline_match": [
            "bang", "bangs", "fringe", "parting", "part", "hairline", "forehead", "baby hair",
            "sideburn", "sideburns", "face-framing", "face framing", "ear-side", "front hair",
            "clean hairline",
        ],
        "requested_color_pattern_match": [
            "color", "colour", "pink", "pastel", "blonde", "black", "brown", "red", "blue",
            "green", "purple", "silver", "gray", "grey", "dye", "highlight", "highlights",
            "shadow", "shadows", "root", "roots", "tips", "gradient", "ombre", "streak",
            "split dye", "underlayer", "saturation", "brightness",
        ],
        "requested_hair_accessory_decoration_match": [
            "hair clip", "hairpin", "ribbon", "bow", "headband", "hair tie", "bead", "flower",
            "hair accessory", "hair decoration",
        ],
    }
    active_request = []
    for facet_id in request_order:
        keywords = keyword_map.get(facet_id, [])
        if any(keyword in prompt_lower for keyword in keywords):
            active_request.append(facet_id)
    active_l3 = [facet_id for facet_id in active_request + quality_order if facet_id in facets]
    if not active_request:
        raise ValueError("v22 rubric did not activate any request facet from prompt")
    missing = [facet_id for facet_id in active_l3 if facet_id not in facets]
    if missing:
        raise ValueError(f"v22 active_l3 contains undefined facets: {missing}")
    return {
        "source_path": str(rubric_path),
        "task_key": str(raw.get("task_key", "")),
        "version": str(raw.get("version", "")),
        "raw": raw,
        "facets": facets,
        "active_request_l3": active_request,
        "active_l3": active_l3,
    }


def build_response_schema_v22(num_facets: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "scores": {
                "type": "array",
                "minItems": num_facets,
                "maxItems": num_facets,
                "items": {"type": "integer", "enum": list(range(10))},
            },
            "failure_tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scores", "failure_tags"],
    }


def validate_v22_output(output: Dict[str, Any], active_l3: List[str]) -> Dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("Gemini output is not a JSON object")
    vector = output.get("scores", output.get("score_vector"))
    if not isinstance(vector, list):
        raise ValueError("scores must be an array")
    if len(vector) != len(active_l3):
        raise ValueError(f"scores length mismatch: expected {len(active_l3)}, got {len(vector)}")
    normalized = []
    for idx, (facet_id, score_raw) in enumerate(zip(active_l3, vector)):
        if isinstance(score_raw, bool):
            raise ValueError(f"scores[{idx}] must be integer 0-9, got bool")
        score = int(score_raw)
        if score < 0 or score > 9:
            raise ValueError(f"scores[{idx}]={score} outside 0-9")
        normalized.append({"facet_id": facet_id, "score": str(score), "score_int": score, "evidence": "", "defect": ""})
    tags = output.get("failure_tags", [])
    if not isinstance(tags, list):
        raise ValueError("failure_tags must be an array")
    return {
        "facet_scores": normalized,
        "scores": [item["score_int"] for item in normalized],
        "failure_tags": [str(x) for x in tags],
        "schema_version": "scores_v22_dynamic_integer_0_9",
    }


def parse_v22_output_text(base: Any, output_text: str, active_l3: List[str]) -> Dict[str, Any]:
    try:
        return base.parse_json_object(output_text)
    except Exception:
        pass
    import re

    text = output_text.strip()
    match = re.search(r'"?scores"?\s*[:=]\s*\[([^\]]+)\]', text, re.S)
    vector_text = match.group(1) if match else ""
    if not vector_text:
        match = re.search(r'"?score_vector"?\s*[:=]\s*\[([^\]]+)\]', text, re.S)
        vector_text = match.group(1) if match else ""
    if not vector_text:
        first_array = re.search(r'\[([^\]]+)\]', text, re.S)
        vector_text = first_array.group(1) if first_array else text
    values = [int(x) for x in re.findall(r'(?<!\d)([0-9])(?!\d)', vector_text)]
    if len(values) != len(active_l3):
        raise ValueError(
            f"Could not recover v22 scores of length {len(active_l3)} from model text; "
            f"found {len(values)} integers. text={output_text[:1000]!r}"
        )
    return {"scores": values, "failure_tags": ["parsed_from_non_strict_json"]}


def call_gemini_l3_judge_v22(
    *,
    base: Any,
    model: str,
    api_key: str,
    base_url: str,
    source_image_path: str,
    edited_image_path: str,
    prompt: str,
    rubric: Dict[str, Any],
    requirement: str,
    max_retries: int,
    request_timeout: float,
    max_image_side: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    active_l3 = rubric["active_l3"]
    facets = rubric["facets"]
    schema = build_response_schema_v22(len(active_l3))
    official_schema = convert_schema_for_gemini(schema)
    metrics = base.image_consistency_metrics(source_image_path, edited_image_path)
    source_url = base.image_to_data_url(source_image_path, max_side=max_image_side, quality=jpeg_quality)
    edited_url = base.image_to_data_url(edited_image_path, max_side=max_image_side, quality=jpeg_quality)
    active_specs = []
    for idx, facet_id in enumerate(active_l3):
        spec = facets[facet_id]
        active_specs.append({
            "index": idx,
            "facet_id": facet_id,
            "l1": spec["l1"],
            "title": spec["title"],
            "definition": spec["definition"],
            "anchors": spec["anchors"],
        })
    system_prompt = (
        "You are a strict expert evaluator for fine-grained human hair editing. "
        "Return only a compact JSON score array. Score every active facet independently "
        "with one integer from 0 to 9."
    )
    user_text = f"""
# Task
task_key: {rubric['task_key']}
rubric_version: {rubric['version']}
schema_version: scores_v22_dynamic_integer_0_9

# Edit instruction
{prompt}

# Additional requirement
{requirement or "None"}

# Objective image consistency hints
These values compare source and edited image over the full image. They are auxiliary hints only; do not replace visual rubric judgment with them.
{base.format_consistency_metrics(metrics)}

# Global scoring policy
{json.dumps(rubric['raw'].get('global_scoring_policy') or {}, ensure_ascii=False, indent=2)}

# Active request facets selected from the prompt
{json.dumps(rubric['active_request_l3'], ensure_ascii=False)}

# Active L3 facets in required output order
{json.dumps(active_specs, ensure_ascii=False, indent=2)}

# Exact JSON schema
{json.dumps(schema, ensure_ascii=False)}

Return only JSON. The scores array must contain exactly {len(active_l3)} integers in the same order as the active L3 facet list above. Do not output decimals, explanations, markdown, or repeated facet names.
""".strip()
    use_response_schema = os.getenv("GEMINI_SCHEMA_V22_USE_RESPONSE_SCHEMA", "0").strip().lower() in {"1", "true", "yes", "y"}
    generation_config = {
        "temperature": 0,
        "candidateCount": 1,
        "maxOutputTokens": int(os.getenv("GEMINI_NATIVE_MAX_OUTPUT_TOKENS", "8192")),
        "responseMimeType": "application/json",
    }
    thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "").strip()
    if thinking_budget:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    if use_response_schema:
        generation_config["responseJsonSchema"] = official_schema
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": user_text},
                {"text": "Source image before editing:"},
                data_url_to_inline_part(source_url),
                {"text": "Edited candidate image:"},
                data_url_to_inline_part(edited_url),
            ],
        }],
        "generationConfig": generation_config,
    }
    url = gemini_generate_url(model, base_url)
    last_err: Optional[Exception] = None
    for attempt in retry_attempts(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8", "x-goog-api-key": api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
            output_text = extract_gemini_text(response_json)
            parsed = parse_v22_output_text(base, output_text, active_l3)
            normalized = validate_v22_output(parsed, active_l3)
            normalized["raw_model_response_text"] = output_text
            normalized["gemini_response_json"] = response_json
            return normalized
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")[:4000]
            last_err = RuntimeError(f"Gemini HTTP {err.code}: {body}")
            if err.code in {400, 401, 403, 404} and not retry_forever_enabled(max_retries):
                raise last_err
        except Exception as err:
            last_err = err
        if has_retry_left(max_retries, attempt):
            sleep_before_retry("Gemini v22 dynamic integer judge", attempt, max_retries, last_err)
    raise RuntimeError(f"Gemini v22 dynamic integer judge failed after {max_retries} attempts: {last_err!r}")


def _weighted_power_mean(values: List[Tuple[float, float]], power: float) -> float:
    denominator = sum(weight for _, weight in values)
    if denominator <= 0:
        return 0.0
    if abs(power) < 1e-9:
        return math.exp(sum(weight * math.log(max(1e-8, value)) for value, weight in values) / denominator)
    return (sum(weight * (max(0.0, value) ** power) for value, weight in values) / denominator) ** (1.0 / power)


def aggregate_v22(rubric: Dict[str, Any], judge_output: Dict[str, Any]) -> Dict[str, Any]:
    active_l3 = rubric["active_l3"]
    facets = rubric["facets"]
    gamma = float((rubric["raw"].get("normalization") or {}).get("gamma", 1.4))
    vector = [int(x) for x in judge_output["scores"]]
    l3_scores: Dict[str, float] = {}
    l3_raw_labels: Dict[str, str] = {}
    for facet_id, score_int in zip(active_l3, vector):
        l3_raw_labels[facet_id] = str(score_int)
        l3_scores[facet_id] = (score_int / 9.0) ** gamma

    agg = rubric["raw"].get("aggregation") or {}
    instr_cfg = agg.get("instruction_block") or {}
    quality_cfg = agg.get("quality_block") or {}
    final_cfg = agg.get("final") or {}
    base_weights = {str(k): float(v) for k, v in (instr_cfg.get("base_weights") or {}).items()}
    quality_weights = {str(k): float(v) for k, v in (quality_cfg.get("weights") or {}).items()}
    request_ids = [fid for fid in rubric["active_request_l3"] if fid in l3_scores]
    quality_ids = [fid for fid in (rubric["raw"].get("quality_facet_order") or []) if fid in l3_scores]
    instruction_score = _weighted_power_mean(
        [(l3_scores[fid], base_weights.get(fid, 1.0)) for fid in request_ids],
        float(instr_cfg.get("power", 0.7)),
    )
    quality_score = _weighted_power_mean(
        [(l3_scores[fid], quality_weights.get(fid, 1.0)) for fid in quality_ids],
        float(quality_cfg.get("power", 0.8)),
    )
    final_values = [
        (instruction_score, float(final_cfg.get("instruction_weight", 0.8))),
        (quality_score, float(final_cfg.get("quality_weight", 0.2))),
    ]
    reward = _weighted_power_mean(final_values, float(final_cfg.get("power", 0.7)))
    l1_scores = {
        "Instruction Following": round(float(instruction_score), 3),
        "Compact Quality Checks": round(float(quality_score), 3),
    }
    return {
        "reward": round(float(reward), 3),
        "reward_raw_bottom_up": float(reward),
        "reward_for_training": float(reward),
        "display_score_0_to_9": round(9.0 * reward, 3),
        "l1_scores": l1_scores,
        "l2_scores": {},
        "l3_scores": {k: round(float(v), 3) for k, v in l3_scores.items()},
        "l3_raw_labels": l3_raw_labels,
        "l3_evidence": {},
        "failure_tags": judge_output.get("failure_tags", []),
        "raw_judge": judge_output,
    }


def build_v22_reason(result: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    labels = result.get("l3_raw_labels") or {}
    scores = result.get("l3_scores") or {}
    weakest = sorted((float(scores[fid]), fid) for fid in scores)[:4]
    weak_bits = [f"{fid}={labels.get(fid)}" for _, fid in weakest]
    details = {
        "score_vector": [int(labels[fid]) for fid in labels],
        "weakest_facets": weak_bits,
        "display_score_0_to_9": result.get("display_score_0_to_9"),
    }
    return "weakest: " + ", ".join(weak_bits), details


def reward_value(row: Dict[str, Any]) -> float:
    return float(row.get("reward", row.get("final_reward", row.get("reward_for_training", 0.0))))


def label_counts_from_row(row: Dict[str, Any]) -> Dict[str, int]:
    labels = ["0", "1", "2", "3", "4", "N/A"]
    counts = {label: 0 for label in labels}
    raw_labels = row.get("l3_raw_labels") or {}
    for value in raw_labels.values():
        if value in counts:
            counts[value] += 1
    if not raw_labels and row.get("label_counts"):
        for label in labels:
            counts[label] = int((row.get("label_counts") or {}).get(label, 0))
    return counts


def write_old_new_report(old_json_path: Path, new_rows: List[Dict[str, Any]], output_path: Path) -> None:
    import statistics
    from collections import Counter

    old_rows = json.loads(old_json_path.read_text(encoding="utf-8"))
    old_scores = [reward_value(row) for row in old_rows]
    new_scores = [reward_value(row) for row in new_rows]
    labels = ["0", "1", "2", "3", "4", "N/A"]
    transition = Counter()
    facet_changes = Counter()
    valid_pairs = 0
    inflated = 0
    order = {"0": 0, "1": 1, "2": 2, "N/A": -1}
    for old_row, new_row in zip(old_rows, new_rows):
        old_labels = old_row.get("l3_raw_labels") or {}
        new_labels = new_row.get("l3_raw_labels") or {}
        for facet_id in sorted(set(old_labels) | set(new_labels)):
            old_label = old_labels.get(facet_id)
            new_label = new_labels.get(facet_id)
            if old_label is not None and new_label is not None:
                valid_pairs += 1
                if old_label in order and new_label in order and order[new_label] > order[old_label]:
                    inflated += 1
                transition[(old_label, new_label)] += 1
            if old_label != new_label:
                facet_changes[facet_id] += 1

    def stats(scores: List[float]) -> Dict[str, Any]:
        return {
            "mean": sum(scores) / max(1, len(scores)),
            "std": statistics.pstdev(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "scores": scores,
        }

    report = {
        "old_json": str(old_json_path),
        "new_json": str(output_path.parent / "schema_v2_scores.json"),
        "old_reward": stats(old_scores),
        "new_reward": stats(new_scores),
        "per_candidate": [
            {
                "index": idx + 1,
                "old_reward": old_scores[idx],
                "new_reward": new_scores[idx],
                "delta": new_scores[idx] - old_scores[idx],
                "old_counts": label_counts_from_row(old_rows[idx]),
                "new_counts": label_counts_from_row(new_rows[idx]),
                "parse_failed": bool(new_rows[idx].get("parse_failed", False)),
            }
            for idx in range(min(len(old_scores), len(new_scores)))
        ],
        "old_label_counts_total": {
            label: sum(label_counts_from_row(row)[label] for row in old_rows)
            for label in labels
        },
        "new_label_counts_total": {
            label: sum(label_counts_from_row(row)[label] for row in new_rows)
            for label in labels
        },
        "label_transition_counts": {f"{old}->{new}": count for (old, new), count in transition.items()},
        "score_inflation_rate": inflated / valid_pairs if valid_pairs else None,
        "top_changed_facets": facet_changes.most_common(20),
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--rubric_yaml", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--compare_old_json", default="")
    parser.add_argument("--model", default=os.getenv("GEMINI_REWARD_MODEL", "gemini-3.1-flash-lite"))
    parser.add_argument("--base_url", default=os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    parser.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("SINGLE_REWARD_API_KEY", ""))))
    parser.add_argument("--request_timeout", type=float, default=float(os.getenv("SINGLE_REWARD_TIMEOUT", "240")))
    parser.add_argument(
        "--max_retries",
        type=int,
        default=int(os.getenv("SINGLE_REWARD_MAX_RETRIES", "0")),
        help="Maximum Gemini retry attempts. Use 0 or a negative value to retry forever until success.",
    )
    parser.add_argument("--max_image_side", type=int, default=int(os.getenv("SINGLE_REWARD_MAX_IMAGE_SIDE", "512")))
    parser.add_argument("--jpeg_quality", type=int, default=int(os.getenv("SINGLE_REWARD_JPEG_QUALITY", "90")))
    parser.add_argument("--limit", type=int, default=0, help="Optional number of candidates to score for quick tests.")
    parser.add_argument(
        "--resume_success",
        default=os.getenv("GEMINI_SCHEMA_REJUDGE_RESUME_SUCCESS", "0"),
        help="If true, reuse successful rows already present in schema_v2_scores.json.",
    )
    parser.add_argument(
        "--fail_open",
        default=os.getenv("GEMINI_SCHEMA_V2_FAIL_OPEN", "0"),
        help="If true, keep saving rows when one candidate fails.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base = load_base_module(repo_root)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir.parent / "schema_v2_rejudge"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_path_from_dir(data_dir)
    prompt_path = data_dir / "prompt.txt"
    metadata_path = data_dir / "metadata.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.is_file() else ""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    requirement = str(metadata.get("requirement") or nested_metadata.get("requirement") or "")
    rubric_yaml = args.rubric_yaml or metadata.get("rubric_yaml") or metadata.get("rubric_file")
    if not rubric_yaml:
        raise ValueError("Missing rubric yaml. Pass --rubric_yaml or include rubric_yaml in metadata.json")
    # dynamic_v13 = False  # disabled old path
    # integer_v18 = False  # disabled old branch flag
    # dynamic_v22 = False  # disabled old path
    integer_mode_name = ""
    raw = yaml.safe_load(Path(rubric_yaml).expanduser().read_text(encoding="utf-8"))
    version = str(raw.get("version", "")) if isinstance(raw, dict) else ""
    compact_contract = raw.get("compact_output_contract") if isinstance(raw, dict) else None
    compact_scores = ((compact_contract or {}).get("format") or {}).get("scores") if isinstance(compact_contract, dict) else None
    fixed_integer_rubric = (
        isinstance(raw, dict)
        and bool(raw.get("active_l3"))
        and (
            version.startswith("v18")
            or version.startswith("v34")
            or ("fixed" in version and "integer" in version)
            or isinstance(compact_scores, str)
        )
        and not version.startswith("v22")
    )
    # Clean active path: keep one scoring contract for hair/beard/etc.
    # Rubrics must provide active_l3 and request a compact integer score_vector.
    if not fixed_integer_rubric:
        raise ValueError(
            "This cleaned Gemini rejudge entry only supports fixed-integer active_l3 rubrics. "
            "Please use a fixed-integer rubric such as rubrics/01_*.yaml, wzt.yaml, "
            "another task-specific YAML under rubrics/."
        )
    rubric = load_v18_integer_rubric(str(rubric_yaml))
    # integer_v18 = True  # old branch flag no longer needed in the clean path
    integer_mode_name = f"fixed_integer_{len(rubric['active_l3'])}facet"
    print(f"[schema_v2] using fixed integer rubric loader for version={version}", flush=True)

    # Disabled old rubric-loading branches retained for audit; do not execute them in the clean path.
    #     if fixed_integer_rubric:
    #         rubric = load_v18_integer_rubric(str(rubric_yaml))
    #         integer_v18 = True
    #         integer_mode_name = f"fixed_integer_{len(rubric['active_l3'])}facet"
    #         print(f"[schema_v2] using fixed integer rubric loader for version={version}", flush=True)
    #     else:
    #         try:
    #             rubric = base.load_rubric(str(rubric_yaml))
    #         except Exception as err:
    #             if isinstance(raw, dict) and version.startswith("v13") and raw.get("activation"):
    #                 rubric = load_v13_dynamic_rubric(str(rubric_yaml), prompt)
    #                 dynamic_v13 = True
    #                 print(f"[schema_v2] using dynamic v13 rubric loader after base loader failed: {err}", flush=True)
    #             elif isinstance(raw, dict) and version.startswith("v18") and raw.get("active_l3"):
    #                 rubric = load_v18_integer_rubric(str(rubric_yaml))
    #                 integer_v18 = True
    #                 integer_mode_name = f"fixed_integer_{len(rubric['active_l3'])}facet"
    #                 print(f"[schema_v2] using fixed integer rubric loader after base loader failed: {err}", flush=True)
    #             elif isinstance(raw, dict) and version.startswith("v22") and raw.get("prompt_parser"):
    #                 rubric = load_v22_dynamic_integer_rubric(str(rubric_yaml), prompt)
    #                 dynamic_v22 = True
    #                 print(f"[schema_v2] using v22 dynamic integer rubric loader after base loader failed: {err}", flush=True)
    #             else:
    #                 raise

    samples = sample_paths_from_dir(data_dir)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    if not samples:
        raise FileNotFoundError(f"No candidate images found in {data_dir}")

    print(f"[schema_v2] model={args.model}", flush=True)
    print(f"[schema_v2] data_dir={data_dir}", flush=True)
    print(f"[schema_v2] rubric={rubric_yaml}", flush=True)
    rubric_mode = integer_mode_name
    active_count = len(rubric["active_l3"])
    print(f"[schema_v2] rubric_mode={rubric_mode} active_l3={active_count}", flush=True)
    # Disabled old dynamic rubric logging retained conceptually: v13 activated_intents and v22 active_request_l3 are not used in the clean path.
    print(f"[schema_v2] samples={len(samples)}", flush=True)

    fail_open = str(args.fail_open).strip().lower() in {"1", "true", "yes", "y"}
    scores_json = output_dir / "schema_v2_scores.json"
    resume_success = str(args.resume_success).strip().lower() in {"1", "true", "yes", "y"}
    existing_by_index: Dict[int, Dict[str, Any]] = {}
    if resume_success and scores_json.is_file():
        try:
            for row in json.loads(scores_json.read_text(encoding="utf-8")):
                sample_index = int(row.get("sample_index", 0))
                if sample_index > 0 and not bool(row.get("parse_failed", False)):
                    existing_by_index[sample_index] = row
            if existing_by_index:
                print(f"[schema_v2] resume_success reusing {len(existing_by_index)} existing successful rows", flush=True)
        except Exception as err:
            print(f"[schema_v2][resume_warning] could not read existing scores: {err!r}", flush=True)
    rows: List[Dict[str, Any]] = []
    for idx, sample_path in enumerate(samples, start=1):
        if idx in existing_by_index:
            print(f"[schema_v2] reusing candidate {idx}/{len(samples)}: {sample_path.name}", flush=True)
            rows.append(existing_by_index[idx])
            continue
        print(f"[schema_v2] scoring candidate {idx}/{len(samples)}: {sample_path.name}", flush=True)
        try:
            # Clean active scoring path. Each candidate is judged independently with the
            # fixed active_l3 score_vector, then aggregated bottom-up as L3 -> L2/L1 -> reward.
            judge = call_gemini_l3_judge_v18(
                base=base,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                source_image_path=str(source_path),
                edited_image_path=str(sample_path),
                prompt=prompt,
                rubric=rubric,
                requirement=requirement,
                max_retries=args.max_retries,
                request_timeout=args.request_timeout,
                max_image_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
            )
            result = aggregate_v18(rubric, judge, prompt=prompt)
            metrics = base.image_consistency_metrics(str(source_path), str(sample_path))
            result["image_consistency_metrics"] = metrics
            result["image_consistency_caps"] = []
            reason, reason_details = build_v18_reason(result)

            # Disabled old scoring branches retained for audit; do not execute them in the clean path.
            #         try:
            # if dynamic_v22:
            #     judge = call_gemini_l3_judge_v22(
            #         base=base,
            #         model=args.model,
            #         api_key=args.api_key,
            #         base_url=args.base_url,
            #         source_image_path=str(source_path),
            #         edited_image_path=str(sample_path),
            #         prompt=prompt,
            #         rubric=rubric,
            #         requirement=requirement,
            #         max_retries=args.max_retries,
            #         request_timeout=args.request_timeout,
            #         max_image_side=args.max_image_side,
            #         jpeg_quality=args.jpeg_quality,
            #     )
            #     result = aggregate_v22(rubric, judge)
            #     metrics = base.image_consistency_metrics(str(source_path), str(sample_path))
            #     result["image_consistency_metrics"] = metrics
            #     result["image_consistency_caps"] = []
            #     reason, reason_details = build_v22_reason(result)
            # elif integer_v18:
            #     judge = call_gemini_l3_judge_v18(
            #         base=base,
            #         model=args.model,
            #         api_key=args.api_key,
            #         base_url=args.base_url,
            #         source_image_path=str(source_path),
            #         edited_image_path=str(sample_path),
            #         prompt=prompt,
            #         rubric=rubric,
            #         requirement=requirement,
            #         max_retries=args.max_retries,
            #         request_timeout=args.request_timeout,
            #         max_image_side=args.max_image_side,
            #         jpeg_quality=args.jpeg_quality,
            #     )
            #     result = aggregate_v18(rubric, judge, prompt=prompt)
            #     metrics = base.image_consistency_metrics(str(source_path), str(sample_path))
            #     result["image_consistency_metrics"] = metrics
            #     result["image_consistency_caps"] = []
            #     reason, reason_details = build_v18_reason(result)
            # elif dynamic_v13:
            #     judge = call_gemini_l3_judge_v13(
            #         base=base,
            #         model=args.model,
            #         api_key=args.api_key,
            #         base_url=args.base_url,
            #         source_image_path=str(source_path),
            #         edited_image_path=str(sample_path),
            #         prompt=prompt,
            #         rubric=rubric,
            #         requirement=requirement,
            #         max_retries=args.max_retries,
            #         request_timeout=args.request_timeout,
            #         max_image_side=args.max_image_side,
            #         jpeg_quality=args.jpeg_quality,
            #     )
            #     metrics = base.image_consistency_metrics(str(source_path), str(sample_path))
            #     result = aggregate_v13(base, rubric, judge, metrics)
            #     result["image_consistency_metrics"] = metrics
            #     result["image_consistency_caps"] = []
            #     reason, reason_details = build_v13_reason(result)
            # else:
            #     judge = call_gemini_l3_judge_v2(
            #         base=base,
            #         model=args.model,
            #         api_key=args.api_key,
            #         base_url=args.base_url,
            #         source_image_path=str(source_path),
            #         edited_image_path=str(sample_path),
            #         prompt=prompt,
            #         rubric=rubric,
            #         requirement=requirement,
            #         max_retries=args.max_retries,
            #         request_timeout=args.request_timeout,
            #         max_image_side=args.max_image_side,
            #         jpeg_quality=args.jpeg_quality,
            #     )
            #     result = base.aggregate_bottom_up(rubric=rubric, judge_output=judge)
            #     metrics = base.image_consistency_metrics(str(source_path), str(sample_path))
            #     result["image_consistency_metrics"] = metrics
            #     result["image_consistency_caps"] = []
            #     result = base.apply_soft_preservation_penalties(result, metrics, rubric)
            #     result = base.apply_training_reward_calibration(result, sample_path=str(sample_path), sample_index=idx)
            #     reason, reason_details = base.build_candidate_reason(result, rubric)
            # result["reason"] = reason

            result["reason"] = reason
            result["reason_details"] = reason_details
            labels = result.get("l3_raw_labels") or {}
            label_space = [str(i) for i in range(10)]
            result["label_counts"] = {label: sum(1 for v in labels.values() if v == label) for label in label_space}
            result["parse_failed"] = False
        except Exception as err:
            if not fail_open:
                raise
            print(f"[schema_v2][candidate_error] index={idx} error={err!r}", flush=True)
            result = {
                "reward": 0.0,
                "reward_raw_bottom_up": 0.0,
                "reward_for_training": 0.0,
                "l1_scores": {},
                "l2_scores": {},
                "l3_scores": {},
                "l3_raw_labels": {},
                "label_counts": {str(i): 0 for i in range(10)},
                "failure_tags": ["schema_v2_candidate_failed"],
                "reason": f"Schema-v2 scoring failed: {err!r}",
                "reason_details": {"error": repr(err)},
                "error": repr(err),
                "parse_failed": True,
            }
        result["sample_index"] = idx
        result["sample_path"] = str(sample_path)
        result["source"] = str(source_path)
        result["prompt"] = prompt
        result["metadata"] = {
            "model_name": args.model,
            "prompt_version": "fixed_integer_score_vector",
            "schema_version": "score_vector_integer_0_9",
            "rubric_version": rubric["version"],
            "rubric_yaml": str(rubric_yaml),
            "generation_config": {
                "temperature": 0,
                "candidateCount": 1,
                "responseMimeType": "application/json",
                "response_schema": "score_vector_integer_0_9_fixed_order",
                "response_schema_enforced_by_api": os.getenv("GEMINI_SCHEMA_V18_USE_RESPONSE_SCHEMA", "0"),
                "rubric_judge_text_enabled": os.getenv("USE_RUBRIC_JUDGE_TEXT", "1"),
            },
            "active_l3": rubric["active_l3"],
            "activated_intents": [],
            "active_request_l3": [],
        }
        rows.append(result)
        scores_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    scores_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    grid_path = output_dir / "schema_v2_scored_compare_grid.jpg"
    save_schema_v2_grid(str(source_path), rows, prompt, grid_path)

    summary = {
        "model": args.model,
        "schema_version": "score_vector_integer_0_9",
        "rubric_mode": rubric_mode,
        "num_samples": len(rows),
        "mean_reward": sum(float(r.get("reward", 0.0)) for r in rows) / max(1, len(rows)),
        "scores": [float(r.get("reward", 0.0)) for r in rows],
        "label_counts_total": {
            label: sum((r.get("label_counts") or {}).get(label, 0) for r in rows)
            for label in [str(i) for i in range(10)]
        },
        "output_grid": str(grid_path),
        "scores_json": str(scores_json),
    }
    (output_dir / "schema_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    old_json_path = Path(args.compare_old_json).expanduser().resolve() if args.compare_old_json else data_dir.parent / "bottom_up_rubric_scores.json"
    if old_json_path.is_file():
        report_path = output_dir / "schema_v2_vs_original_report.json"
        write_old_new_report(old_json_path, rows, report_path)
        summary["comparison_report"] = str(report_path)
        (output_dir / "schema_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[schema_v2] summary=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
