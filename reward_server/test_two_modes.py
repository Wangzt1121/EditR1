#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-mode rubric scorer for one source image + one prompt + N edited candidates.

Mode 1, single:
    Reuse the repository-local test_gemini.py behavior.
    Each candidate is judged independently, then Python ranks all candidates.

Mode 2, relative:
    Send the whole candidate group to GPT/Gemini in one request.
    The judge compares candidates against each other, but still outputs only
    L3 facet labels. Python keeps the final aggregation.

Mode 3, both:
    Run single and relative on the same inputs, then write a compact comparison
    table so you can decide which reward design is more useful.

This file intentionally imports the Original scorer and reuses its YAML
rubric parser, bottom-up aggregation, soft penalties, CSV/JSON/grid writers,
and input discovery utilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_TEST_PATH = Path(
    os.getenv(
        "EDIT_R1_ORIGINAL_REWARD_SCORER",
        str(Path(__file__).resolve().with_name("test_gemini.py")),
    )
).expanduser()


def load_base_module():
    if not BASE_TEST_PATH.exists():
        raise FileNotFoundError(f"Base scorer not found: {BASE_TEST_PATH}")
    spec = importlib.util.spec_from_file_location("wzt_base_test_scorer", BASE_TEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base scorer from {BASE_TEST_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_original_single_mode(argv: List[str]) -> None:
    """Run the original test.py CLI unchanged, minus our --mode flag."""
    base = load_base_module()
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(BASE_TEST_PATH)] + argv
        base.main()
    finally:
        sys.argv = old_argv


def _has_cli_arg(argv: List[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in argv)


def _override_cli_arg(argv: List[str], name: str, value: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == name:
            i += 2
            continue
        if item.startswith(name + "="):
            i += 1
            continue
        out.append(item)
        i += 1
    out.extend([name, value])
    return out


def _load_ranked_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ranking file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Ranking file must contain a list: {path}")
    return data


def _rank_by_sample(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    ranked = sorted(rows, key=lambda row: float(row.get("reward", 0.0)), reverse=True)
    return {int(row.get("sample_index", idx + 1)): idx + 1 for idx, row in enumerate(ranked)}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / ((vx ** 0.5) * (vy ** 0.5))


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


def write_mode_comparison(single_dir: Path, relative_dir: Path, output_dir: Path) -> Dict[str, Any]:
    single_rows = _load_ranked_json(single_dir / "ranked_samples.json")
    relative_rows = _load_ranked_json(relative_dir / "ranked_samples.json")
    single_by_index = {int(row.get("sample_index")): row for row in single_rows}
    relative_by_index = {int(row.get("sample_index")): row for row in relative_rows}
    sample_indices = sorted(set(single_by_index) & set(relative_by_index))
    if not sample_indices:
        raise RuntimeError("No overlapping sample_index values between single and relative outputs")

    single_reward_rank = _rank_by_sample(single_rows)
    relative_reward_rank = _rank_by_sample(relative_rows)
    comparison_rows: List[Dict[str, Any]] = []

    for sample_index in sample_indices:
        srow = single_by_index[sample_index]
        rrow = relative_by_index[sample_index]
        single_reward = _safe_float(srow.get("reward"))
        relative_reward = _safe_float(rrow.get("reward"))
        comparison_rows.append({
            "sample_index": sample_index,
            "sample_path": rrow.get("sample_path") or srow.get("sample_path"),
            "single_reward": single_reward,
            "relative_reward": relative_reward,
            "reward_delta_relative_minus_single": relative_reward - single_reward,
            "single_reward_rank": single_reward_rank.get(sample_index),
            "relative_reward_rank": relative_reward_rank.get(sample_index),
            "gpt_relative_rank": rrow.get("relative_rank"),
            "reward_rank_delta_relative_minus_single": (
                int(relative_reward_rank.get(sample_index, 0)) - int(single_reward_rank.get(sample_index, 0))
            ),
            "single_reason": srow.get("reason", ""),
            "relative_reason": rrow.get("relative_reason") or rrow.get("reason", ""),
        })

    single_scores = [_safe_float(single_by_index[i].get("reward")) for i in sample_indices]
    relative_scores = [_safe_float(relative_by_index[i].get("reward")) for i in sample_indices]
    single_ranks = [float(single_reward_rank[i]) for i in sample_indices]
    relative_ranks = [float(relative_reward_rank[i]) for i in sample_indices]
    top_single = min(sample_indices, key=lambda i: single_reward_rank[i])
    top_relative = min(sample_indices, key=lambda i: relative_reward_rank[i])
    top_gpt_relative = None
    gpt_rankable = [
        (int(relative_by_index[i].get("relative_rank")), i)
        for i in sample_indices
        if relative_by_index[i].get("relative_rank") is not None
    ]
    if gpt_rankable:
        top_gpt_relative = min(gpt_rankable)[1]

    summary = {
        "num_candidates": len(sample_indices),
        "single_score_std": _std(single_scores),
        "relative_score_std": _std(relative_scores),
        "score_pearson": _pearson(single_scores, relative_scores),
        "rank_spearman_approx": _pearson(single_ranks, relative_ranks),
        "top1_same_by_reward": top_single == top_relative,
        "top1_single_sample_index": top_single,
        "top1_relative_reward_sample_index": top_relative,
        "top1_gpt_relative_sample_index": top_gpt_relative,
        "single_output_dir": str(single_dir),
        "relative_output_dir": str(relative_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mode_comparison.csv"
    json_path = output_dir / "mode_comparison.json"
    summary_path = output_dir / "mode_comparison_summary.json"

    fieldnames = [
        "sample_index",
        "sample_path",
        "single_reward",
        "relative_reward",
        "reward_delta_relative_minus_single",
        "single_reward_rank",
        "relative_reward_rank",
        "gpt_relative_rank",
        "reward_rank_delta_relative_minus_single",
        "single_reason",
        "relative_reason",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    json_path.write_text(json.dumps(comparison_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["comparison_csv"] = str(csv_path)
    summary["comparison_json"] = str(json_path)
    summary["summary_json"] = str(summary_path)
    return summary


def summarize_mode_comparison(single_dir: Path, relative_dir: Path) -> Dict[str, Any]:
    single_rows = _load_ranked_json(single_dir / "ranked_samples.json")
    relative_rows = _load_ranked_json(relative_dir / "ranked_samples.json")
    single_by_index = {int(row.get("sample_index")): row for row in single_rows}
    relative_by_index = {int(row.get("sample_index")): row for row in relative_rows}
    sample_indices = sorted(set(single_by_index) & set(relative_by_index))
    if not sample_indices:
        raise RuntimeError("No overlapping sample_index values between single and relative outputs")

    single_reward_rank = _rank_by_sample(single_rows)
    relative_reward_rank = _rank_by_sample(relative_rows)
    single_scores = [_safe_float(single_by_index[i].get("reward")) for i in sample_indices]
    relative_scores = [_safe_float(relative_by_index[i].get("reward")) for i in sample_indices]
    single_ranks = [float(single_reward_rank[i]) for i in sample_indices]
    relative_ranks = [float(relative_reward_rank[i]) for i in sample_indices]
    top_single = min(sample_indices, key=lambda i: single_reward_rank[i])
    top_relative = min(sample_indices, key=lambda i: relative_reward_rank[i])
    gpt_rankable = [
        (int(relative_by_index[i].get("relative_rank")), i)
        for i in sample_indices
        if relative_by_index[i].get("relative_rank") is not None
    ]
    top_gpt_relative = min(gpt_rankable)[1] if gpt_rankable else None
    return {
        "num_candidates": len(sample_indices),
        "single_score_std": _std(single_scores),
        "relative_score_std": _std(relative_scores),
        "score_pearson": _pearson(single_scores, relative_scores),
        "rank_spearman_approx": _pearson(single_ranks, relative_ranks),
        "top1_same_by_reward": top_single == top_relative,
        "top1_single_sample_index": top_single,
        "top1_relative_reward_sample_index": top_relative,
        "top1_gpt_relative_sample_index": top_gpt_relative,
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _font(base: Any, size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return base.ImageFont.truetype(path, size=size)
    return base.ImageFont.load_default()


def _text_w(draw: Any, text: str, font: Any) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _wrap(draw: Any, text: Any, font: Any, max_width: int, max_lines: int) -> List[str]:
    words = str(text or "").replace("\n", " ").split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if _text_w(draw, trial, font) <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
            current = word
        else:
            clipped = word
            while clipped and _text_w(draw, clipped + "...", font) > max_width:
                clipped = clipped[:-1]
            lines.append((clipped + "...") if clipped else "...")
            current = ""
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and words:
        full = " ".join(words)
        shown = " ".join(lines)
        if len(shown) < len(full) and not lines[-1].endswith("..."):
            while lines[-1] and _text_w(draw, lines[-1] + "...", font) > max_width:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "..."
    return lines[:max_lines]


def _metric(row: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return _safe_float(value)
    return None


def _compact_l1(row: Dict[str, Any]) -> str:
    l1 = row.get("l1_scores") or {}
    if not isinstance(l1, dict):
        l1 = {}

    def by_words(words: List[str], exclude: Optional[List[str]] = None) -> Optional[float]:
        exclude = exclude or []
        for name, value in l1.items():
            lower = str(name).lower()
            if all(word in lower for word in words) and not any(word in lower for word in exclude):
                return _safe_float(value)
        return None

    edit = _metric(row, "edit_group_score") or by_words(["target"]) or by_words(["edit"], ["identity"])
    preserve = _metric(row, "preservation_group_score") or by_words(["preservation"])
    color = _metric(row, "color_detail_group_score") or row.get("three_pillar_color")
    quality = _metric(row, "quality_group_score") or by_words(["quality"]) or by_words(["visual"])
    return (
        f"edit={edit if edit is not None else 0.0:.3f}  "
        f"keep={preserve if preserve is not None else 0.0:.3f}  "
        f"color={_safe_float(color) if color is not None else 0.0:.3f}  "
        f"quality={quality if quality is not None else 0.0:.3f}"
    )


def save_compact_grid(
    *,
    base: Any,
    source_path: str,
    rows: List[Dict[str, Any]],
    prompt: str,
    output_dir: Path,
    mode_label: str,
) -> Path:
    ordered = sorted(rows, key=lambda row: int(row.get("sample_index", 0)))
    ranked = sorted(ordered, key=lambda row: float(row.get("reward", 0.0)), reverse=True)
    rank_by_index = {int(row.get("sample_index", idx + 1)): idx + 1 for idx, row in enumerate(ranked)}

    cols = 3 if len(ordered) <= 8 else 4
    tile_w = 560
    image_h = 440
    label_h = 250
    cell_h = image_h + label_h
    total = len(ordered) + 1
    rows_n = math.ceil(total / cols)
    canvas = base.Image.new("RGB", (cols * tile_w, rows_n * cell_h), "white")
    draw = base.ImageDraw.Draw(canvas)
    title_font = _font(base, 24, True)
    metric_font = _font(base, 18, False)
    small_font = _font(base, 16, False)
    pad = 18

    def paste_panel(index: int, image_path: str, title: str, lines: List[str], border: tuple[int, int, int]) -> None:
        col = index % cols
        row = index // cols
        x = col * tile_w
        y = row * cell_h
        draw.rectangle((x, y, x + tile_w - 1, y + cell_h - 1), outline=(210, 210, 210), width=1)
        try:
            img = base.Image.open(image_path).convert("RGB")
            img = base.ImageOps.contain(img, (tile_w - 2 * pad, image_h - 2 * pad), method=base.Image.Resampling.LANCZOS)
            px = x + (tile_w - img.width) // 2
            py = y + (image_h - img.height) // 2
            canvas.paste(img, (px, py))
        except Exception as err:
            draw.text((x + pad, y + pad), f"image load error: {err}", fill=(180, 0, 0), font=small_font)
        draw.rectangle((x + 6, y + 6, x + tile_w - 7, y + image_h - 7), outline=border, width=5)
        ty = y + image_h + 12
        draw.text((x + pad, ty), title, fill=(0, 0, 0), font=title_font)
        ty += 34
        for line_idx, line in enumerate(lines):
            font = metric_font if line_idx == 0 else small_font
            for wrapped in _wrap(draw, line, font, tile_w - 2 * pad, 2 if line_idx < 2 else 3):
                draw.text((x + pad, ty), wrapped, fill=(25, 25, 25), font=font)
                ty += 24 if font == metric_font else 21
                if ty > y + cell_h - 22:
                    return

    paste_panel(
        0,
        source_path,
        "SOURCE",
        [
            f"mode={mode_label}",
            f"prompt: {prompt}",
        ],
        (65, 105, 210),
    )

    for index, row in enumerate(ordered, start=1):
        sample_index = int(row.get("sample_index", index))
        rank = rank_by_index.get(sample_index, index)
        reward = _safe_float(row.get("reward"))
        raw = _safe_float(row.get("reward_raw_bottom_up"))
        relative_rank = row.get("relative_rank")
        rel_part = f" | rel-rank {relative_rank}" if relative_rank is not None else ""
        title = f"C{sample_index:02d} | rank {rank}{rel_part} | reward {reward:.3f}"
        border = (35, 145, 75) if rank == 1 else (210, 80, 65) if rank == len(ordered) else (145, 145, 145)
        reason = row.get("relative_reason") or row.get("reason") or ""
        failures = row.get("failure_tags") or []
        if isinstance(failures, list) and failures:
            failure_text = "fail: " + ", ".join(str(item) for item in failures[:4])
        else:
            failure_text = ""
        lines = [
            _compact_l1(row),
            f"raw={raw:.3f}  soft_penalty={_safe_float(row.get('soft_penalty_total')):.3f}",
            f"reason: {reason}",
        ]
        if failure_text:
            lines.append(failure_text)
        paste_panel(index, str(row.get("sample_path", "")), title, lines, border)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"scored_compare_grid_{mode_label}.jpg"
    canvas.save(out_path, quality=95)
    return out_path


def compact_grid_from_output(base: Any, output_dir: Path, mode_label: str) -> Path:
    rows = _load_jsonl(output_dir / "scored_samples.jsonl")
    if not rows:
        rows = _load_ranked_json(output_dir / "ranked_samples.json")
    if not rows:
        raise RuntimeError(f"No rows found under {output_dir}")
    first = rows[0]
    source_path = str(first.get("source") or "")
    prompt = str(first.get("prompt") or "")
    if not source_path:
        raise RuntimeError(f"Cannot find source path in rows under {output_dir}")
    return save_compact_grid(
        base=base,
        source_path=source_path,
        rows=rows,
        prompt=prompt,
        output_dir=output_dir,
        mode_label=mode_label,
    )


def cleanup_output_dir(output_dir: Path, keep_paths: List[Path]) -> None:
    keep = {path.resolve() for path in keep_paths}
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        resolved = child.resolve()
        if resolved in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                pass


RELATIVE_DIMENSIONS = [
    "target_edit_accuracy",
    "identity_preservation",
    "non_target_preservation",
    "color_lighting_texture_preservation",
    "photorealism_artifact_control",
]


RELATIVE_DIMENSION_WEIGHTS = {
    "target_edit_accuracy": 0.34,
    "identity_preservation": 0.18,
    "non_target_preservation": 0.20,
    "color_lighting_texture_preservation": 0.18,
    "photorealism_artifact_control": 0.10,
}


def build_group_response_schema(active_l3: List[str], candidate_ids: List[str]) -> Dict[str, Any]:
    n = len(candidate_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_results": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_id": {"type": "string", "enum": candidate_ids},
                        "relative_rank": {"type": "integer", "minimum": 1, "maximum": n},
                        "relative_reason": {"type": "string"},
                        "dimension_scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                key: {"type": "integer", "minimum": 0, "maximum": 9}
                                for key in RELATIVE_DIMENSIONS
                            },
                            "required": RELATIVE_DIMENSIONS,
                        },
                        "dimension_reasons": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                key: {"type": "string"}
                                for key in RELATIVE_DIMENSIONS
                            },
                            "required": RELATIVE_DIMENSIONS,
                        },
                        "failure_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "candidate_id",
                        "relative_rank",
                        "relative_reason",
                        "dimension_scores",
                        "dimension_reasons",
                        "failure_tags",
                    ],
                },
            }
        },
        "required": ["candidate_results"],
    }


def validate_group_judge_output(output: Dict[str, Any], active_l3: List[str], candidate_ids: List[str]) -> None:
    if not isinstance(output, dict):
        raise ValueError("Group judge output must be a JSON object")
    results = output.get("candidate_results")
    if not isinstance(results, list):
        raise ValueError("Group judge output missing candidate_results list")

    got_ids = [str(item.get("candidate_id", "")) for item in results if isinstance(item, dict)]
    if set(got_ids) != set(candidate_ids) or len(got_ids) != len(candidate_ids):
        raise ValueError(f"Candidate id mismatch. expected={candidate_ids}, got={got_ids}")
    if len(got_ids) != len(set(got_ids)):
        raise ValueError(f"Duplicate candidate_id in output: {got_ids}")

    ranks = []
    for item in results:
        cid = item["candidate_id"]
        rank = item.get("relative_rank")
        if not isinstance(rank, int) or not (1 <= rank <= len(candidate_ids)):
            raise ValueError(f"Invalid relative_rank for {cid}: {rank!r}")
        ranks.append(rank)
        scores = item.get("dimension_scores")
        reasons = item.get("dimension_reasons")
        if not isinstance(scores, dict):
            raise ValueError(f"Missing dimension_scores for {cid}")
        if not isinstance(reasons, dict):
            raise ValueError(f"Missing dimension_reasons for {cid}")
        for key in RELATIVE_DIMENSIONS:
            value = scores.get(key)
            if not isinstance(value, int) or not (0 <= value <= 9):
                raise ValueError(f"Invalid {key} score for {cid}: {value!r}")
            if key not in reasons or not isinstance(reasons.get(key), str):
                raise ValueError(f"Missing {key} reason for {cid}")

    # Unique ranks are preferable. Some models may tie despite instructions,
    # but ties weaken relative supervision, so treat them as invalid.
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"Duplicate relative_rank in output: {ranks}")


def call_gpt_group_l3_judge(
    *,
    base: Any,
    client: Any,
    model: str,
    source_image_path: str,
    edited_image_paths: List[str],
    prompt: str,
    rubric: Any,
    requirement: str = "",
    max_retries: int = 3,
    max_image_side: int = 1024,
    jpeg_quality: int = 90,
) -> Dict[str, Any]:
    active_l3 = rubric.active_l3
    candidate_ids = [f"candidate_{idx:02d}" for idx in range(1, len(edited_image_paths) + 1)]
    rubric_text = base.build_rubric_text(rubric)
    schema = build_group_response_schema(active_l3, candidate_ids)

    source_url = base.image_to_data_url(source_image_path, max_side=max_image_side, quality=jpeg_quality)
    candidate_urls = [
        base.image_to_data_url(path, max_side=max_image_side, quality=jpeg_quality)
        for path in edited_image_paths
    ]

    system_prompt = (
        "You are a strict expert evaluator for fine-grained human image editing. "
        "You will judge one source image, one edit instruction, and multiple edited candidates together. "
        "The candidate order is anonymous; do not assume model identity. "
        "You must compare candidates against each other for calibration, but output only five dimension scores. "
        "Do not output an overall score; Python will compute the final reward. "
        "Be strict about prompt mismatch, identity drift, non-target changes, global color shifts, "
        "saturation/contrast drift, edit leakage, blur, smoothing, and artifacts."
    )

    user_text = f"""
# Task
task_key: {rubric.task_key}
rubric_version: {rubric.version}

# Edit instruction
{prompt}

# Additional requirement
{requirement or "None"}

# Relative judging requirement
You are given {len(edited_image_paths)} candidates from the same source image and the same edit instruction.
Judge all candidates together. Use the other candidates as references for calibration.
A candidate should receive lower dimension scores when another candidate better satisfies the same edit while preserving identity, color, saturation, brightness, contrast, texture, non-target regions, and realism.

Important:
- Do not score candidates as isolated images.
- Do not reveal or infer model identities.
- Do not output an overall score.
- For each candidate, output relative_rank, relative_reason, failure_tags, five 0-9 dimension_scores, and short dimension_reasons.
- relative_rank must be a strict permutation from 1 to {len(edited_image_paths)}, where rank 1 is best.

# Five required dimensions, each scored 0-9
- target_edit_accuracy: whether the requested edit is correct, visible, localized, and has appropriate strength.
- identity_preservation: whether the same person/face/pose/expression/face shape is preserved.
- non_target_preservation: whether background, clothing, hair, accessories, body, and unedited regions are unchanged.
- color_lighting_texture_preservation: whether source color, saturation, brightness, contrast, skin texture, and image tone are preserved.
- photorealism_artifact_control: whether the edited area is realistic, sharp, naturally blended, and free from blur/smoothing/artifacts.

Score calibration:
- 9: excellent, clearly among the best in this group for that dimension.
- 7-8: good with only minor visible issues.
- 4-6: usable but has clear defects.
- 1-3: severe defects.
- 0: missing/wrong edit or catastrophic preservation/quality failure for that dimension.

Calibration priority:
1. The requested edit must be correct.
2. If target edit quality is similar, identity preservation and non-target preservation decide the ranking.
3. Color, saturation, brightness, contrast, texture, blur, smoothing, and artifacts are decisive preservation/quality evidence.
4. A good target edit cannot compensate for obvious global or non-target damage.

# Rubric-specific judge instructions
{rubric.judge_instructions or "None"}

# Rubric source audit
rubric_source_path: {rubric.source_path}
rubric_task_key: {rubric.task_key}
rubric_version: {rubric.version}
rubric_checklist_sha256: {hashlib.sha256(rubric_text.encode("utf-8", errors="replace")).hexdigest()}

# Original active L3 facet ids from the YAML rubric
Use these only as detailed judging guidance. Do not output L3 facet scores in relative mode.
{json.dumps(active_l3, ensure_ascii=False)}

# Candidate ids, in image order
{json.dumps(candidate_ids, ensure_ascii=False)}

# L1/L2/L3 rubric checklist used as judging guidance
{rubric_text}

# Exact JSON schema
{json.dumps(schema, ensure_ascii=False)}

Return only JSON matching the schema.
""".strip()

    last_err: Optional[Exception] = None
    base_url_text = str(getattr(client, "base_url", "") or "").lower()
    use_chat_completions = model.lower().startswith("gemini") or "generativelanguage.googleapis.com" in base_url_text

    for attempt in range(1, max_retries + 1):
        try:
            if use_chat_completions:
                content: List[Dict[str, Any]] = [
                    {"type": "text", "text": user_text},
                    {"type": "text", "text": "Source image before editing:"},
                    {"type": "image_url", "image_url": {"url": source_url}},
                ]
                for candidate_id, candidate_url in zip(candidate_ids, candidate_urls):
                    content.append({"type": "text", "text": f"Edited candidate {candidate_id}:"})
                    content.append({"type": "image_url", "image_url": {"url": candidate_url}})

                chat_kwargs = {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 4096,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                }
                try:
                    response = client.chat.completions.create(
                        **chat_kwargs,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "relative_group_dimension_scores",
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    )
                except Exception:
                    try:
                        response = client.chat.completions.create(
                            **chat_kwargs,
                            response_format={"type": "json_object"},
                        )
                    except Exception:
                        response = client.chat.completions.create(**chat_kwargs)
                output_text = response.choices[0].message.content
            else:
                content = [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_text", "text": "Source image before editing:"},
                    {"type": "input_image", "image_url": source_url},
                ]
                for candidate_id, candidate_url in zip(candidate_ids, candidate_urls):
                    content.append({"type": "input_text", "text": f"Edited candidate {candidate_id}:"})
                    content.append({"type": "input_image", "image_url": candidate_url})

                response = client.responses.create(
                    model=model,
                    temperature=0,
                    max_output_tokens=4096,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "relative_group_dimension_scores",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                )
                output_text = response.output_text

            if hasattr(base, "parse_json_object"):
                output = base.parse_json_object(output_text)
            else:
                output = json.loads(base.extract_json_object_text(output_text))
            validate_group_judge_output(output, active_l3, candidate_ids)
            return output
        except Exception as err:
            last_err = err
            if attempt < max_retries:
                print(f"[relative] judge attempt {attempt}/{max_retries} failed: {err!r}; retrying...")
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Relative judge failed after {max_retries} attempts: {last_err}") from err

    raise RuntimeError("Unexpected relative judge failure")


def parse_relative_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-wise blind relative rubric scorer.")
    parser.add_argument("--data_dir", default="/nvmedata/workspace2/users/wzt/data")
    parser.add_argument("--source", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt_file", default="")
    parser.add_argument("--requirement", default="")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--sample_dir", default=None)
    parser.add_argument("--candidate_prefix", default="candidate_")
    parser.add_argument("--expected_samples", type=int, default=int(os.getenv("EXPECTED_SAMPLES", "0")))
    parser.add_argument("--rubric_yaml", default="/nvmedata/workspace2/users/wzt/hair_edit.yaml")
    parser.add_argument("--output_dir", default="/nvmedata/workspace2/users/wzt/data/rubric_relative_out")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--model", default=os.getenv("GPT_RUBRIC_MODEL", os.getenv("GPT_REWARD_MODEL", os.getenv("GEMINI_REWARD_MODEL", "gemini-2.5-flash"))))
    parser.add_argument(
        "--api_key",
        default=os.getenv(
            "OPENAI_API_KEY",
            os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("GPT_REWARD_API_KEY", ""))),
        ),
    )
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", os.getenv("GPT_REWARD_BASE_URL", "https://grsaiapi.com/v1/chat/completions")))
    parser.add_argument("--request_timeout", type=float, default=float(os.getenv("GPT_RUBRIC_TIMEOUT", "120")))
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=1, help="Accepted for CLI compatibility; relative mode uses one grouped API call.")
    return parser.parse_args(argv)


def build_relative_dimension_result(base: Any, item: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    dim_raw = item.get("dimension_scores") or {}
    dim = {key: _safe_float(dim_raw.get(key), 0.0) / 9.0 for key in RELATIVE_DIMENSIONS}
    weighted = sum(dim[key] * RELATIVE_DIMENSION_WEIGHTS[key] for key in RELATIVE_DIMENSIONS)

    rgb = abs(_safe_float(metrics.get("rgb_mean_l2")))
    brightness = abs(_safe_float(metrics.get("brightness_delta")))
    contrast = abs(_safe_float(metrics.get("contrast_delta")))
    saturation = abs(_safe_float(metrics.get("saturation_delta")))
    edge_loss = max(0.0, -_safe_float(metrics.get("edge_sharpness_delta")))
    metric_penalty = min(rgb / 180.0, 0.10) + min(brightness / 120.0, 0.08) + min(contrast / 90.0, 0.06) + min(saturation / 160.0, 0.10) + min(edge_loss / 60.0, 0.06)
    reward = max(0.0, min(0.995, weighted - metric_penalty))

    preservation = (dim["identity_preservation"] + dim["non_target_preservation"]) / 2.0
    reasons = item.get("dimension_reasons") or {}
    return {
        "reward": reward,
        "reward_raw_bottom_up": weighted,
        "reward_raw_discrete": weighted,
        "reward_for_training": reward,
        "reward_before_soft_penalty": weighted,
        "reward_after_soft_penalty": reward,
        "soft_penalty_total": metric_penalty,
        "edit_group_score": dim["target_edit_accuracy"],
        "preservation_group_score": preservation,
        "color_detail_group_score": dim["color_lighting_texture_preservation"],
        "quality_group_score": dim["photorealism_artifact_control"],
        "metric_color_quality_score": max(0.0, min(1.0, 1.0 - metric_penalty)),
        "l1_scores": {
            "target_edit_accuracy": dim["target_edit_accuracy"],
            "identity_preservation": dim["identity_preservation"],
            "non_target_preservation": dim["non_target_preservation"],
            "color_lighting_texture_preservation": dim["color_lighting_texture_preservation"],
            "photorealism_artifact_control": dim["photorealism_artifact_control"],
        },
        "dimension_scores_raw_0_9": dim_raw,
        "dimension_scores_0_1": dim,
        "dimension_reasons": reasons,
        "failure_tags": item.get("failure_tags", []),
    }


def run_relative_mode(argv: List[str], *, clean_outputs: bool = True) -> Dict[str, Any]:
    base = load_base_module()
    base.load_default_judge_env()
    args = parse_relative_args(argv)
    args = base.resolve_inputs(args)
    rubric_yaml = base.resolve_rubric_yaml_path(args.rubric_yaml, args.data_dir)
    rubric = base.load_rubric(rubric_yaml)
    sample_paths = base.collect_sample_paths(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rubric_text = base.build_rubric_text(rubric)
    rubric_text_sha256 = hashlib.sha256(rubric_text.encode("utf-8", errors="replace")).hexdigest()
    rubric_used_path = output_dir / "rubric_used.json"
    checklist_path = output_dir / "rubric_checklist.txt"
    rubric_used_path.write_text(json.dumps(base.rubric_to_jsonable(rubric), ensure_ascii=False, indent=2), encoding="utf-8")
    checklist_path.write_text(rubric_text + "\n", encoding="utf-8")

    print("========== Relative Group Rubric Test ==========")
    print(f"source           : {args.source}")
    print(f"prompt           : {args.prompt}")
    print(f"requirement      : {args.requirement}")
    print(f"rubric_yaml      : {rubric_yaml}")
    print(f"rubric_task_key  : {rubric.task_key}")
    print(f"rubric_version   : {rubric.version}")
    print(f"rubric_sha256    : {rubric_text_sha256}")
    print(f"active_l3        : {len(rubric.active_l3)}")
    print(f"model            : {args.model}")
    print(f"base_url         : {args.base_url or '<default openai>'}")
    print(f"num candidates   : {len(sample_paths)}")
    print(f"output_dir       : {output_dir}")
    print("================================================")

    if args.dry_run:
        print("\nDry run enabled. No GPT/Gemini calls were made.")
        if clean_outputs:
            cleanup_output_dir(output_dir, [])
        return {"status": "dry_run", "output_dir": str(output_dir)}

    if (
        not args.api_key
        and not os.getenv("OPENAI_API_KEY")
        and not os.getenv("GEMINI_API_KEY")
        and not os.getenv("GOOGLE_API_KEY")
        and not os.getenv("OPENAI_ADMIN_KEY")
    ):
        raise RuntimeError("Missing API key. Export OPENAI_API_KEY/GEMINI_API_KEY or pass --api_key.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency: openai. Install with `pip install openai`.") from exc

    client_kwargs: Dict[str, Any] = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = base.normalize_openai_compatible_base_url(args.base_url)
    if args.request_timeout and args.request_timeout > 0:
        client_kwargs["timeout"] = args.request_timeout

    client = OpenAI(**client_kwargs)
    group_output = call_gpt_group_l3_judge(
        base=base,
        client=client,
        model=args.model,
        source_image_path=args.source,
        edited_image_paths=sample_paths,
        prompt=args.prompt,
        rubric=rubric,
        requirement=args.requirement,
        max_retries=args.max_retries,
        max_image_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
    )

    by_id = {item["candidate_id"]: item for item in group_output["candidate_results"]}
    source_hash = base.file_sha256(args.source)
    rows: List[Dict[str, Any]] = []
    valid_count = 0
    failed_count = 0

    for i, sample_path in enumerate(sample_paths, start=1):
        candidate_id = f"candidate_{i:02d}"
        item = by_id[candidate_id]
        metadata = {
            "judging_mode": "blind_relative_group",
            "group_size": len(sample_paths),
            "candidate_id": candidate_id,
            "relative_rank": item.get("relative_rank"),
            "relative_reason": item.get("relative_reason", ""),
            "rubric_task_key": rubric.task_key,
            "rubric_version": rubric.version,
            "rubric_yaml": rubric_yaml,
            "rubric_text_sha256": rubric_text_sha256,
            "model": args.model,
            "requirement": args.requirement,
            "sample_index": i,
            "sample_path": sample_path,
            "source_sha256": source_hash,
            "edited_sha256": base.file_sha256(sample_path),
        }
        try:
            consistency_metrics = base.image_consistency_metrics(args.source, sample_path)
            result = build_relative_dimension_result(base, item, consistency_metrics)
            result["image_consistency_metrics"] = consistency_metrics
            result["image_consistency_caps"] = []
            result = base.apply_training_reward_calibration(result, sample_path=sample_path, sample_index=i)
            dim_reasons = item.get("dimension_reasons") or {}
            compact_reason = (
                f"relative_rank={item.get('relative_rank')}; "
                f"{item.get('relative_reason', '').strip()} | "
                f"edit={dim_reasons.get('target_edit_accuracy', '')}; "
                f"keep={dim_reasons.get('non_target_preservation', '')}; "
                f"color={dim_reasons.get('color_lighting_texture_preservation', '')}"
            )
            result["reason"] = compact_reason
            result["reason_details"] = {
                "relative_rank": item.get("relative_rank"),
                "relative_reason": item.get("relative_reason", ""),
                "dimension_reasons": dim_reasons,
                "dimension_scores": item.get("dimension_scores", {}),
            }
            rows.append({
                "sample_index": i,
                "sample_path": sample_path,
                "source": args.source,
                "prompt": args.prompt,
                "metadata": metadata,
                "judging_mode": "blind_relative_group",
                "candidate_id": candidate_id,
                "relative_rank": item.get("relative_rank"),
                "relative_reason": item.get("relative_reason", ""),
                "error": None,
                **result,
            })
            valid_count += 1
            print(
                f"candidate={candidate_id} rank={item.get('relative_rank')} "
                f"reward={float(result.get('reward', 0.0)):.4f} "
                f"raw={float(result.get('reward_raw_bottom_up', 0.0)):.4f}"
            )
            print("Reason:", rows[-1].get("reason"))
        except Exception as err:
            failed_count += 1
            rows.append({
                "sample_index": i,
                "sample_path": sample_path,
                "source": args.source,
                "prompt": args.prompt,
                "metadata": metadata,
                "judging_mode": "blind_relative_group",
                "candidate_id": candidate_id,
                "relative_rank": item.get("relative_rank"),
                "relative_reason": item.get("relative_reason", ""),
                "reward": 0.001 + i * 0.0001,
                "reward_raw_discrete": 0.0,
                "reward_for_training": 0.001 + i * 0.0001,
                "reward_raw_bottom_up": 0.0,
                "reason": f"Aggregation failed: {err!r}",
                "reason_details": {"error": repr(err), "score_is_placeholder": True},
                "error": repr(err),
                "score_is_placeholder": True,
            })

    base.enforce_unique_training_rewards(rows)
    rows = sorted(rows, key=lambda row: int(row.get("sample_index", 0)))
    ranked = sorted(rows, key=lambda row: float(row.get("reward", 0.0)), reverse=True)
    scores = [float(row.get("reward", 0.0)) for row in rows]

    jsonl_path = output_dir / "scored_samples.jsonl"
    ranked_json_path = output_dir / "ranked_samples.json"
    csv_path = output_dir / "ranked_samples.csv"
    scores_path = output_dir / "scores_only.json"
    raw_group_path = output_dir / "raw_relative_group_output.json"

    base.write_jsonl(jsonl_path, rows)
    base.write_csv(csv_path, ranked, rubric)
    ranked_json_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    scores_path.write_text(json.dumps({"scores": scores}, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_group_path.write_text(json.dumps(group_output, ensure_ascii=False, indent=2), encoding="utf-8")

    grid_path = save_compact_grid(
        base=base,
        source_path=args.source,
        rows=rows,
        prompt=args.prompt,
        output_dir=output_dir,
        mode_label="relative",
    )
    if clean_outputs:
        cleanup_output_dir(output_dir, [grid_path])

    print("\n========== Final scores: List[float] ==========")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print("\n========== Ranking ==========")
    for rank, row in enumerate(ranked, start=1):
        print(
            f"rank={rank} sample={row['sample_index']} "
            f"relative_rank={row.get('relative_rank')} "
            f"reward={float(row.get('reward', 0.0)):.4f} path={row['sample_path']}"
        )
    print("\nSaved:")
    print(f"- {grid_path}")
    if clean_outputs:
        print("Compact output enabled: intermediate JSON/CSV/TXT files were removed.")
    return {
        "status": "ok" if failed_count == 0 else "partial",
        "output_dir": str(output_dir),
        "valid_scores": valid_count,
        "failed_scores": failed_count,
        "scores": scores,
        "grid": str(grid_path),
    }


def run_both_mode(argv: List[str]) -> Dict[str, Any]:
    base = load_base_module()
    base.load_default_judge_env()
    probe_args = parse_relative_args(argv)
    if _has_cli_arg(argv, "--output_dir"):
        root_dir = Path(probe_args.output_dir)
    else:
        root_dir = Path(probe_args.data_dir) / "rubric_mode_compare_out"
    single_dir = root_dir / "single"
    relative_dir = root_dir / "relative"

    single_argv = _override_cli_arg(argv, "--output_dir", str(single_dir))
    relative_argv = _override_cli_arg(argv, "--output_dir", str(relative_dir))
    if single_dir.exists():
        shutil.rmtree(single_dir, ignore_errors=True)
    if relative_dir.exists():
        shutil.rmtree(relative_dir, ignore_errors=True)

    print("========== Both Mode: single then relative ==========")
    print(f"root_output : {root_dir}")
    print(f"single_dir  : {single_dir}")
    print(f"relative_dir: {relative_dir}")
    print("=====================================================")

    dry_run = _has_cli_arg(argv, "--dry_run")
    print("\n[1/2] Running original single-candidate scorer...")
    run_original_single_mode(single_argv)
    single_grid: Optional[Path] = None
    if not dry_run:
        single_grid = compact_grid_from_output(base, single_dir, "single")

    print("\n[2/2] Running blind relative group scorer...")
    relative_result = run_relative_mode(relative_argv, clean_outputs=False)

    if dry_run:
        print("\nDry run enabled. Comparison files are skipped because no scores were produced.")
        cleanup_output_dir(single_dir, [])
        cleanup_output_dir(relative_dir, [])
        return {
            "status": "dry_run",
            "root_output": str(root_dir),
            "single_output_dir": str(single_dir),
            "relative_output_dir": str(relative_dir),
        }
    relative_grid = Path(str(relative_result.get("grid")))

    print("\n[3/3] Summarizing mode comparison...")
    summary = summarize_mode_comparison(single_dir, relative_dir)
    if single_grid is None:
        raise RuntimeError("single grid was not created")
    cleanup_output_dir(single_dir, [single_grid])
    cleanup_output_dir(relative_dir, [relative_grid])
    old_comparison_dir = root_dir / "comparison"
    if old_comparison_dir.exists():
        shutil.rmtree(old_comparison_dir, ignore_errors=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nMain files:")
    print(f"- single grid   : {single_grid}")
    print(f"- relative grid : {relative_grid}")
    print("Compact output enabled: only the two grid JPG files were kept.")

    return {
        "status": relative_result.get("status", "ok"),
        "root_output": str(root_dir),
        "single_output_dir": str(single_dir),
        "relative_output_dir": str(relative_dir),
        "comparison": summary,
    }


def score_relative_candidates(
    *,
    source: str,
    prompt: str,
    samples: List[str],
    rubric_yaml: str,
    output_dir: str,
    requirement: str = "",
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    max_image_side: int = 1024,
    jpeg_quality: int = 90,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Programmatic relative reward entry for later Edit-R1 integration.

    The returned dict contains the final per-candidate scores under "scores".
    This function still writes the same audit files as the CLI so reward
    decisions remain inspectable during experiments.
    """
    argv: List[str] = [
        "--source", source,
        "--prompt", prompt,
        "--rubric_yaml", rubric_yaml,
        "--output_dir", output_dir,
        "--expected_samples", str(len(samples)),
        "--max_image_side", str(max_image_side),
        "--jpeg_quality", str(jpeg_quality),
        "--max_retries", str(max_retries),
        "--samples",
    ] + list(samples)
    if requirement:
        argv.extend(["--requirement", requirement])
    if model:
        argv.extend(["--model", model])
    if api_key:
        argv.extend(["--api_key", api_key])
    if base_url:
        argv.extend(["--base_url", base_url])
    return run_relative_mode(argv)


def infer_cli_output_dir(argv: List[str], default: str) -> Path:
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--output_dir" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if item.startswith("--output_dir="):
            return Path(item.split("=", 1)[1])
        i += 1
    return Path(default)


def split_mode(argv: List[str]) -> tuple[str, List[str]]:
    mode = "relative"
    rest: List[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == "--mode":
            if i + 1 >= len(argv):
                raise SystemExit("--mode requires one value: single or relative")
            mode = argv[i + 1].strip().lower()
            i += 2
            continue
        if item.startswith("--mode="):
            mode = item.split("=", 1)[1].strip().lower()
            i += 1
            continue
        rest.append(item)
        i += 1
    if mode not in {"single", "relative", "both"}:
        raise SystemExit(f"Unknown --mode {mode!r}; use single, relative, or both")
    return mode, rest


def main() -> None:
    mode, rest = split_mode(sys.argv[1:])
    if mode == "single":
        print("Running original single-candidate independent scorer...")
        run_original_single_mode(rest)
        output_dir = infer_cli_output_dir(rest, "/nvmedata/workspace2/users/wzt/data/rubric_test_out")
        if _has_cli_arg(rest, "--dry_run"):
            cleanup_output_dir(output_dir, [])
        else:
            base = load_base_module()
            grid = compact_grid_from_output(base, output_dir, "single")
            cleanup_output_dir(output_dir, [grid])
            print(f"Compact output enabled. Kept only: {grid}")
    elif mode == "relative":
        print("Running blind relative group scorer...")
        run_relative_mode(rest)
    else:
        print("Running both scorers and comparing outputs...")
        run_both_mode(rest)


if __name__ == "__main__":
    main()
