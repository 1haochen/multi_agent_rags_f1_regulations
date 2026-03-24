"""
Remove strikethrough text from PaddleOCR-VL JSON using PyMuPDF.

Block-level strike detection fails when one OCR block contains many PDF spans and only
small substrings are struck (e.g. magenta ', and ' with a strike line, while magenta
'and Inspection' next to it is not struck). This module rebuilds each block's text from
PDF spans whose bboxes intersect the block region (mapped PDF↔image), skipping spans
that intersect revision strike geometry — same idea as notebooks/test_strikethrough_pymupdf.ipynb.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import fitz  # PyMuPDF


PdfSegment = Tuple[float, float, float, float]  # (x0, mid_y, x1, mid_y) in PDF user space
PdfRect = Tuple[float, float, float, float]  # (x0, y0, x1, y1)


def _is_revision_fill_or_stroke(fill, stroke, strict: bool = True) -> bool:
    """Heuristic for magenta / red revision strike graphics."""

    def ok_rgb(t, r_lo, g_hi, b_hi) -> bool:
        if t is None or len(t) < 3:
            return False
        r, g, b = t[0], t[1], t[2]
        return r >= r_lo and g <= g_hi and b <= b_hi

    if not strict:
        return True

    if fill and fill[0] > 0.85 and fill[2] > 0.85 and fill[1] < 0.2:
        return True
    if stroke and stroke[0] > 0.85 and stroke[2] > 0.85 and stroke[1] < 0.2:
        return True
    if ok_rgb(fill, 0.7, 0.35, 0.35) or ok_rgb(stroke, 0.7, 0.35, 0.35):
        return True

    return False


def collect_horizontal_segments(
    drawings,
    *,
    angle_tol_deg: float = 5.0,
    max_rect_height_pt: float = 3.0,
    min_rect_width_pt: float = 4.0,
    revision_color_only: bool = True,
) -> List[PdfSegment]:
    """Strike-like segments in PDF user space (notebook logic)."""
    import numpy as np

    segs: List[PdfSegment] = []

    for d in drawings:
        fill, stroke = d.get("fill"), d.get("color")
        if revision_color_only and not _is_revision_fill_or_stroke(fill, stroke, strict=True):
            continue

        for item in d.get("items", []):
            itype = item[0]

            if itype == "l":
                _, p1, p2 = item
                x0, y0 = float(p1.x), float(p1.y)
                x1, y1 = float(p2.x), float(p2.y)
                dx, dy = x1 - x0, y1 - y0
                L = (dx * dx + dy * dy) ** 0.5
                if L <= 0:
                    continue
                ang = float(np.degrees(np.arctan2(dy, dx)))
                ang = (ang + 180.0) % 180.0 - 90.0
                if abs(ang) <= angle_tol_deg:
                    ym = 0.5 * (y0 + y1)
                    segs.append((min(x0, x1), ym, max(x0, x1), ym))

            elif itype == "re":
                r = item[1]
                x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                w, h = abs(x1 - x0), abs(y1 - y0)
                if h <= max_rect_height_pt and w >= min_rect_width_pt:
                    ym = 0.5 * (y0 + y1)
                    segs.append((min(x0, x1), ym, max(x0, x1), ym))

            elif itype == "qu":
                q = item[1]
                pts = [q.ul, q.ur, q.ll, q.lr]
                xs = [float(p.x) for p in pts]
                ys = [float(p.y) for p in pts]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w, h = x1 - x0, y1 - y0
                if h <= max_rect_height_pt * 2.0 and w >= min_rect_width_pt:
                    ym = 0.5 * (y0 + y1)
                    segs.append((x0, ym, x1, ym))

    return segs


def span_is_struck(
    span: Dict,
    horizontal_segments: List[PdfSegment],
    *,
    coverage_ratio_min: float = 0.25,
    midline_frac: float = 0.45,
    min_overlap_pt: float = 6.0,
    center_dist_frac: float = 0.35,
    len_ratio_min: float = 0.35,
    len_ratio_max: float = 1.6,
) -> bool:
    """True if a horizontal strike segment crosses this span's bbox (PDF space)."""
    from math import isfinite

    text = (span.get("text") or "").strip()
    if not text:
        return False
    bbox = span.get("bbox")
    if not bbox or len(bbox) < 4:
        return False
    sx0, sy0, sx1, sy1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    span_mid_y = 0.5 * (sy0 + sy1)
    span_height = abs(sy1 - sy0)
    span_width = abs(sx1 - sx0)
    if span_width <= 0 or span_height <= 0:
        return False

    for (lx0, ly0, lx1, ly1) in horizontal_segments:
        ly_mid = 0.5 * (ly0 + ly1)
        if abs(ly_mid - span_mid_y) > midline_frac * span_height:
            continue
        s_min_x, s_max_x = min(sx0, sx1), max(sx0, sx1)
        l_min_x, l_max_x = min(lx0, lx1), max(lx0, lx1)
        overlap = max(0.0, min(s_max_x, l_max_x) - max(s_min_x, l_min_x))
        if overlap <= 0:
            continue
        line_len = abs(lx1 - lx0)
        if not isfinite(line_len) or line_len <= 0:
            continue
        # Strong overlap required (both relative and absolute)
        if (overlap / span_width) < coverage_ratio_min:
            continue
        if overlap < min_overlap_pt:
            continue

        # Strike segment length should be close-ish to span width.
        # This avoids removing neighboring pink words when a strike just touches them.
        len_ratio = line_len / span_width
        if not (len_ratio_min <= len_ratio <= len_ratio_max):
            continue

        # Strike center should stay near span center in x.
        span_cx = 0.5 * (s_min_x + s_max_x)
        line_cx = 0.5 * (l_min_x + l_max_x)
        if abs(line_cx - span_cx) > center_dist_frac * span_width:
            continue

        return True

    return False


def char_is_struck(
    char_bbox: PdfRect,
    horizontal_segments: List[PdfSegment],
    *,
    midline_frac: float = 0.6,
    coverage_ratio_min: float = 0.45,
) -> bool:
    """
    Char-level strike test for cases where one span contains both struck and non-struck words.
    """
    cx0, cy0, cx1, cy1 = char_bbox
    char_h = abs(cy1 - cy0)
    char_w = abs(cx1 - cx0)
    if char_h <= 0 or char_w <= 0:
        return False
    c_mid_y = 0.5 * (cy0 + cy1)
    c_min_x, c_max_x = min(cx0, cx1), max(cx0, cx1)
    for (lx0, ly0, lx1, ly1) in horizontal_segments:
        ly_mid = 0.5 * (ly0 + ly1)
        if abs(ly_mid - c_mid_y) > midline_frac * char_h:
            continue
        l_min_x, l_max_x = min(lx0, lx1), max(lx0, lx1)
        overlap = max(0.0, min(c_max_x, l_max_x) - max(c_min_x, l_min_x))
        if overlap <= 0:
            continue
        if (overlap / char_w) >= coverage_ratio_min:
            return True
    return False


def _rects_overlap(a: PdfRect, b: PdfRect, margin: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ax0, ax1 = min(ax0, ax1) - margin, max(ax0, ax1) + margin
    ay0, ay1 = min(ay0, ay1) - margin, max(ay0, ay1) + margin
    bx0, bx1 = min(bx0, bx1), max(bx0, bx1)
    by0, by1 = min(by0, by1), max(by0, by1)
    if ax1 < bx0 or bx1 < ax0:
        return False
    if ay1 < by0 or by1 < ay0:
        return False
    return True


def iter_spans_reading_order(text_dict: dict) -> Iterator[Dict]:
    """Walk blocks → lines → spans in extraction order."""
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                yield span


def _join_span_parts(parts: List[str]) -> str:
    """
    PDF spans usually carry their own trailing spaces; after dropping struck spans,
    neighbors may need an explicit space (e.g. 'manufacture' + 'assembly ').
    """
    if not parts:
        return ""
    out = parts[0]
    for n in parts[1:]:
        if not n:
            continue
        if not out:
            out = n
            continue
        if out[-1].isspace() or n[0].isspace():
            out += n
        elif n[0] in ",.;:!?)]}%":
            out += n
        elif out[-1] in "([{-":
            out += n
        else:
            out += " " + n
    out = out.rstrip()
    # Strike removal can leave doubled spaces at deletion boundaries.
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


def img_bbox_to_pdf_rect(
    bbox_img: Iterable[float],
    sx: float,
    sy: float,
) -> PdfRect:
    """Map PaddleOCR block_bbox [x0,y0,x1,y1] to PDF user-space rectangle."""
    ix0, iy0, ix1, iy1 = (float(bbox_img[0]), float(bbox_img[1]), float(bbox_img[2]), float(bbox_img[3]))
    if sx <= 0 or sy <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(ix0, ix1) / sx,
        min(iy0, iy1) / sy,
        max(ix0, ix1) / sx,
        max(iy0, iy1) / sy,
    )


def rebuild_block_text_from_pdf_spans(
    block_bbox_img: List[float],
    text_dict: dict,
    raw_dict: dict,
    segments_pdf: List[PdfSegment],
    sx: float,
    sy: float,
    *,
    margin_pdf: float = 4.0,
    coverage_ratio_min: float = 0.25,
    midline_frac: float = 0.45,
    min_overlap_pt: float = 6.0,
    center_dist_frac: float = 0.35,
    len_ratio_min: float = 0.35,
    len_ratio_max: float = 1.6,
) -> Optional[str]:
    """
    Concatenate PDF span texts that fall inside the OCR block region, excluding struck spans.

    Returns None if no span bbox overlapped the block (caller should keep original OCR text).
    """
    region = img_bbox_to_pdf_rect(block_bbox_img, sx, sy)
    parts: List[str] = []
    any_overlap = False

    # Prefer rawdict chars so we can keep non-struck text inside a mixed span.
    for block in raw_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sb = span.get("bbox")
                if not sb or len(sb) < 4:
                    continue
                srect = (float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]))
                if not _rects_overlap(region, srect, margin=margin_pdf):
                    continue
                any_overlap = True

                chars = span.get("chars") or []
                if chars:
                    kept = []
                    for ch in chars:
                        cb = ch.get("bbox")
                        if not cb or len(cb) < 4:
                            continue
                        cb_rect = (float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3]))
                        if not char_is_struck(
                            cb_rect,
                            segments_pdf,
                            midline_frac=min(0.7, max(0.45, midline_frac + 0.1)),
                            coverage_ratio_min=0.45,
                        ):
                            kept.append(ch.get("c", ""))
                    parts.append("".join(kept))
                else:
                    # Fallback for non-char spans
                    if span_is_struck(
                        span,
                        segments_pdf,
                        coverage_ratio_min=coverage_ratio_min,
                        midline_frac=midline_frac,
                        min_overlap_pt=min_overlap_pt,
                        center_dist_frac=center_dist_frac,
                        len_ratio_min=len_ratio_min,
                        len_ratio_max=len_ratio_max,
                    ):
                        continue
                    parts.append(span.get("text") or "")

    if not any_overlap:
        return None
    return _join_span_parts(parts)


def clean_page_entry(
    page_entry: dict,
    pdf_doc: fitz.Document,
    *,
    margin_pdf: float = 4.0,
    coverage_ratio_min: float = 0.25,
    midline_frac: float = 0.45,
    min_overlap_pt: float = 6.0,
    center_dist_frac: float = 0.35,
    len_ratio_min: float = 0.35,
    len_ratio_max: float = 1.6,
) -> dict:
    """
    For each layout block with a bbox, replace block_content with PDF span text
    minus struck spans when spans overlap the block; otherwise keep OCR text.
    """
    page_index = page_entry["page_index"]
    page = pdf_doc[page_index]

    text_dict = page.get_text("dict", sort=True)
    raw_dict = page.get_text("rawdict", sort=True)
    pdf_w = float(text_dict.get("width"))
    pdf_h = float(text_dict.get("height"))

    ocr_w = float(page_entry.get("width"))
    ocr_h = float(page_entry.get("height"))

    sx = ocr_w / pdf_w if pdf_w else 1.0
    sy = ocr_h / pdf_h if pdf_h else 1.0

    drawings = page.get_drawings()
    segments_pdf = collect_horizontal_segments(drawings, revision_color_only=True)

    new_parsing: List[dict] = []
    for blk in page_entry.get("parsing_res_list", []):
        bbox = blk.get("block_bbox")
        if not bbox or len(bbox) != 4:
            new_parsing.append(blk)
            continue

        original = blk.get("block_content") or ""
        if not str(original).strip():
            new_parsing.append(blk)
            continue

        rebuilt = rebuild_block_text_from_pdf_spans(
            bbox,
            text_dict,
            raw_dict,
            segments_pdf,
            sx,
            sy,
            margin_pdf=margin_pdf,
            coverage_ratio_min=coverage_ratio_min,
            midline_frac=midline_frac,
            min_overlap_pt=min_overlap_pt,
            center_dist_frac=center_dist_frac,
            len_ratio_min=len_ratio_min,
            len_ratio_max=len_ratio_max,
        )

        new_blk = dict(blk)
        if rebuilt is not None:
            new_blk["block_content"] = rebuilt
            new_blk["block_content_ocr_original"] = original
        new_parsing.append(new_blk)

    new_entry = dict(page_entry)
    new_entry["parsing_res_list"] = new_parsing

    lines = []
    for blk in new_parsing:
        txt = (blk.get("block_content") or "").strip()
        if txt:
            lines.append(txt)
    new_entry["text_no_strike"] = "\n".join(lines)

    return new_entry


def process_ocr_json(
    ocr_json_path: Path,
    *,
    margin_pdf: float = 4.0,
    coverage_ratio_min: float = 0.25,
    midline_frac: float = 0.45,
    min_overlap_pt: float = 6.0,
    center_dist_frac: float = 0.35,
    len_ratio_min: float = 0.35,
    len_ratio_max: float = 1.6,
) -> Path:
    data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list of pages in {ocr_json_path}")

    first = data[0]
    pdf_path = Path(first["input_path"])
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found for OCR JSON: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    cleaned_pages = [
        clean_page_entry(
            page_entry,
            doc,
            margin_pdf=margin_pdf,
            coverage_ratio_min=coverage_ratio_min,
            midline_frac=midline_frac,
            min_overlap_pt=min_overlap_pt,
            center_dist_frac=center_dist_frac,
            len_ratio_min=len_ratio_min,
            len_ratio_max=len_ratio_max,
        )
        for page_entry in data
    ]

    out_path = ocr_json_path.with_name(ocr_json_path.stem + "_no_strike.json")
    out_path.write_text(json.dumps(cleaned_pages, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove intra-block strikethrough from PaddleOCR-VL JSON by rebuilding "
            "block_content from PyMuPDF spans (minus strike geometry), per validated notebook logic."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Either a single PaddleOCR-VL JSON file, or a directory under which all "
            "matching JSONs will be processed recursively."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_by_PaddleOCR-VL.json",
        help=(
            "When PATH is a directory, glob pattern to match OCR JSON files "
            "(default: '*_by_PaddleOCR-VL.json')."
        ),
    )
    parser.add_argument(
        "--margin-pdf",
        type=float,
        default=4.0,
        help="Expand OCR block rect in PDF points when matching spans (default: 4).",
    )
    parser.add_argument(
        "--coverage-min",
        type=float,
        default=0.25,
        help="Min horizontal overlap ratio (strike vs span width) to count as struck (default: 0.25).",
    )
    parser.add_argument(
        "--midline-frac",
        type=float,
        default=0.45,
        help="Max |strike_mid_y - span_mid_y| as fraction of span height (default: 0.45).",
    )
    parser.add_argument(
        "--min-overlap-pt",
        type=float,
        default=6.0,
        help="Minimum absolute x-overlap (PDF points) between strike and span (default: 6).",
    )
    parser.add_argument(
        "--center-dist-frac",
        type=float,
        default=0.35,
        help="Max center distance / span width for strike-span pairing (default: 0.35).",
    )
    parser.add_argument(
        "--len-ratio-min",
        type=float,
        default=0.35,
        help="Minimum strike-length / span-width ratio (default: 0.35).",
    )
    parser.add_argument(
        "--len-ratio-max",
        type=float,
        default=1.6,
        help="Maximum strike-length / span-width ratio (default: 1.6).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    target: Path = args.path
    if target.is_dir():
        json_files = sorted(target.rglob(args.pattern))
        if not json_files:
            print(f"[remove_strike] No files matching pattern {args.pattern!r} under {target}")
            return 0
        print(f"[remove_strike] Found {len(json_files)} JSON file(s) under {target} matching {args.pattern!r}")
        for p in json_files:
            print(f"[remove_strike] processing {p}")
            out = process_ocr_json(
                p,
                margin_pdf=args.margin_pdf,
                coverage_ratio_min=args.coverage_min,
                midline_frac=args.midline_frac,
                min_overlap_pt=args.min_overlap_pt,
                center_dist_frac=args.center_dist_frac,
                len_ratio_min=args.len_ratio_min,
                len_ratio_max=args.len_ratio_max,
            )
            print(f"  -> wrote {out}")
    else:
        out = process_ocr_json(
            target,
            margin_pdf=args.margin_pdf,
            coverage_ratio_min=args.coverage_min,
            midline_frac=args.midline_frac,
            min_overlap_pt=args.min_overlap_pt,
            center_dist_frac=args.center_dist_frac,
            len_ratio_min=args.len_ratio_min,
            len_ratio_max=args.len_ratio_max,
        )
        print(f"[remove_strike] wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
