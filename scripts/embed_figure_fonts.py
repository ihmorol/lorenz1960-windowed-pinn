#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import fitz
import matplotlib
import numpy as np
from fontTools import subset
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    ROOT / "figures" / "plateau-results.pdf",
    ROOT / "figures" / "solution_vs_reference.pdf",
)
ORIGINAL_SHA256 = {
    "plateau-results.pdf": "b9698646caae91780588a1ec42a75fd4e97ac3ea8f1b5722e7d30b384e82fdd0",
    "solution_vs_reference.pdf": "85ee08e9970783350d66bf863c8bd7d16dd2211e15984eb897b02af3bcd95342",
}
FONT_SHA256 = {
    "DejaVuSerif.ttf": "107244956e9962b9e96faccdc551825e0ae0898ae13737133e1b921a2fd35ffa",
    "DejaVuSerif-Bold.ttf": "c3753f2ed6bc673f15846dc45addbeb3b9c872f32fb18fd53a21f1bef1ed7676",
    "DejaVuSerif-Italic.ttf": "3e7994fbc54fa10ce3352a42d548fadd7d9cadb69cb1109bc9d960f6dac57f04",
}
FONT_BY_BASE_NAME = {
    "DejaVuSerif": "DejaVuSerif.ttf",
    "DejaVuSerif-Bold": "DejaVuSerif-Bold.ttf",
    "DejaVuSerif-Italic": "DejaVuSerif-Italic.ttf",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def content_streams(doc: fitz.Document) -> dict[tuple[int, int], tuple[bytes, bytes]]:
    streams: dict[tuple[int, int], tuple[bytes, bytes]] = {}
    for page_number, page in enumerate(doc):
        for xref in page.get_contents():
            streams[(page_number, xref)] = (
                doc.xref_stream_raw(xref),
                doc.xref_stream(xref),
            )
    return streams


def page_boxes(doc: fitz.Document) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(page.rect) for page in doc)


def render_rgb(doc: fitz.Document, zoom: float = 4.0) -> tuple[np.ndarray, ...]:
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
        pages.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy())
    return tuple(pages)


def type3_fonts(doc: fitz.Document) -> list[tuple[int, str]]:
    found: dict[int, str] = {}
    for page in doc:
        for xref, _ext, subtype, base_font, *_rest in page.get_fonts(full=True):
            if subtype == "Type3":
                found[xref] = base_font
    return sorted(found.items())


def encoding_glyph_names(doc: fitz.Document, font_xref: int) -> set[str]:
    kind, value = doc.xref_get_key(font_xref, "Encoding")
    if kind not in {"dict", "xref"}:
        raise RuntimeError(f"font {font_xref}: unsupported Encoding {kind}")
    if kind == "xref":
        value = doc.xref_object(int(value.split()[0]), compressed=False)
    if "/Differences" not in value:
        raise RuntimeError(f"font {font_xref}: missing Encoding Differences")
    names = set(re.findall(r"/([A-Za-z][A-Za-z0-9_.]*)", value.split("/Differences", 1)[1]))
    names.discard("Type")
    names.discard("Encoding")
    if not names:
        raise RuntimeError(f"font {font_xref}: empty Encoding Differences")
    return names


def matching_font(base_font: str) -> Path:
    family = base_font.split("+", 1)[-1]
    filename = FONT_BY_BASE_NAME.get(family)
    if filename is None:
        raise RuntimeError(f"unsupported Type 3 font: {base_font}")
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename
    expected = FONT_SHA256[filename]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"unexpected {filename} SHA-256: {actual}")
    return path


def subset_font(path: Path, glyph_names: set[str]) -> bytes:
    font = TTFont(path, recalcBBoxes=False, recalcTimestamp=False)
    missing = glyph_names.difference(font.getGlyphOrder())
    if missing:
        raise RuntimeError(f"{path.name}: missing glyphs {sorted(missing)}")
    options = subset.Options()
    options.glyph_names = True
    options.hinting = True
    options.recalc_bounds = False
    options.recalc_timestamp = False
    options.canonical_order = True
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6]
    options.name_languages = [0x409]
    sub = subset.Subsetter(options=options)
    sub.populate(glyphs=glyph_names | {".notdef"})
    sub.subset(font)
    from io import BytesIO

    output = BytesIO()
    font.save(output, reorderTables=False)
    embedded = output.getvalue()
    check = TTFont(BytesIO(embedded), lazy=False)
    remaining = glyph_names.difference(check.getGlyphOrder())
    if remaining:
        raise RuntimeError(f"subset lost glyphs {sorted(remaining)}")
    return embedded


def replace_font(doc: fitz.Document, font_xref: int, base_font: str) -> None:
    glyph_names = encoding_glyph_names(doc, font_xref)
    embedded = subset_font(matching_font(base_font), glyph_names)

    descriptor_kind, descriptor_value = doc.xref_get_key(font_xref, "FontDescriptor")
    if descriptor_kind != "xref":
        raise RuntimeError(f"font {font_xref}: missing FontDescriptor")
    descriptor_xref = int(descriptor_value.split()[0])

    font_file_xref = doc.get_new_xref()
    doc.update_object(font_file_xref, "<<>>")
    doc.update_stream(font_file_xref, embedded, compress=True)
    doc.xref_set_key(font_file_xref, "Length1", str(len(embedded)))
    doc.xref_set_key(descriptor_xref, "FontFile2", f"{font_file_xref} 0 R")

    required = {}
    for key in ("BaseFont", "FirstChar", "LastChar", "Widths", "Encoding", "FontDescriptor"):
        kind, value = doc.xref_get_key(font_xref, key)
        if kind == "null":
            raise RuntimeError(f"font {font_xref}: missing {key}")
        required[key] = value
    to_unicode_kind, to_unicode = doc.xref_get_key(font_xref, "ToUnicode")

    entries = [
        "/Type /Font",
        "/Subtype /TrueType",
        f"/BaseFont {required['BaseFont']}",
        f"/FirstChar {required['FirstChar']}",
        f"/LastChar {required['LastChar']}",
        f"/Widths {required['Widths']}",
        f"/Encoding {required['Encoding']}",
        f"/FontDescriptor {required['FontDescriptor']}",
    ]
    if to_unicode_kind != "null":
        entries.append(f"/ToUnicode {to_unicode}")
    doc.update_object(font_xref, "<< " + " ".join(entries) + " >>")


def visual_match(before: tuple[np.ndarray, ...], after: tuple[np.ndarray, ...]) -> None:
    if len(before) != len(after):
        raise RuntimeError("page count changed")
    for page_number, (old, new) in enumerate(zip(before, after), start=1):
        if old.shape != new.shape:
            raise RuntimeError(f"page {page_number}: rendered dimensions changed")
        difference = np.abs(old.astype(np.int16) - new.astype(np.int16))
        mean_error = float(difference.mean())
        changed_fraction = float(np.any(difference > 24, axis=2).mean())
        old_ink = np.any(old < 245, axis=2)
        new_ink = np.any(new < 245, axis=2)
        union = np.logical_or(old_ink, new_ink).sum()
        iou = float(np.logical_and(old_ink, new_ink).sum() / union) if union else 1.0
        if mean_error > 0.55 or changed_fraction > 0.012 or iou < 0.97:
            raise RuntimeError(
                f"page {page_number}: visual mismatch "
                f"(mean error {mean_error:.4f}, changed {changed_fraction:.4%}, ink IoU {iou:.5f})"
            )
        print(
            f"  page {page_number}: mean pixel error={mean_error:.4f}, "
            f"changed={changed_fraction:.4%}, ink IoU={iou:.5f}"
        )


def convert(path: Path) -> None:
    source_hash = sha256_file(path)
    with fitz.open(path) as before_doc:
        fonts = type3_fonts(before_doc)
        if not fonts:
            for page in before_doc:
                for font in page.get_fonts(full=True):
                    if font[2] == "Type3":
                        raise RuntimeError(f"{path.name}: Type 3 font remains")
            print(f"{path.name}: already contains no Type 3 fonts")
            return
        if source_hash != ORIGINAL_SHA256[path.name]:
            raise RuntimeError(f"{path.name}: unexpected source SHA-256 {source_hash}")
        original_streams = content_streams(before_doc)
        original_boxes = page_boxes(before_doc)
        original_render = render_rgb(before_doc)

        for font_xref, base_font in fonts:
            replace_font(before_doc, font_xref, base_font)

        temporary = path.with_suffix(".font-embed.tmp.pdf")
        before_doc.save(temporary, garbage=0, deflate=False, clean=False)

    try:
        with fitz.open(temporary) as after_doc:
            if type3_fonts(after_doc):
                raise RuntimeError(f"{path.name}: Type 3 font remains")
            if page_boxes(after_doc) != original_boxes:
                raise RuntimeError(f"{path.name}: page geometry changed")
            if content_streams(after_doc) != original_streams:
                raise RuntimeError(f"{path.name}: page content streams changed")
            for page in after_doc:
                for xref, _ext, subtype, base_font, *_rest in page.get_fonts(full=True):
                    if subtype != "TrueType":
                        raise RuntimeError(f"{path.name}: unexpected font subtype {subtype}")
                    _name, extension, _font_type, content = after_doc.extract_font(xref)
                    if extension != "ttf" or not content:
                        raise RuntimeError(f"{path.name}: {base_font} is not embedded TrueType")
            visual_match(original_render, render_rgb(after_doc))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"{path.name}: {len(fonts)} fonts converted; SHA-256 {sha256_file(path)}")


def main() -> int:
    selected = tuple(Path(arg).resolve() for arg in sys.argv[1:]) or FIGURES
    allowed = {path.resolve() for path in FIGURES}
    for path in selected:
        if path not in allowed:
            raise RuntimeError(f"refusing unapproved file: {path}")
        convert(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
