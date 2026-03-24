import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

# Paddle documents PyTorch↔Paddle API differences (e.g. torch.max) here; the runtime
# still emits bridge warnings from paddle.utils.decorator_utils — not actionable in app code.
# https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.max.html


def _configure_warnings(*, verbose: bool) -> None:
    """
    PaddleOCR-VL uses Paddle layers that mirror PyTorch APIs. Paddle warns when an op
    does not match PyTorch 1:1 (see torch.max / torch.split docs above). Separately,
    paddle.tensor.creation warns about clone().detach() vs to_tensor — also internal.

    ``warnings.filterwarnings(..., message=...)`` uses re.match() on the *start* of the
    message, so regex-based filters often miss. Filtering by issuing module is reliable.

    With ``--verbose-warnings``, we do not install these filters (you will see the spam).
    """
    if verbose:
        return
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"paddle\.tensor\.creation",
    )
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        module=r"paddle\.utils\.decorator_utils",
    )


def _quiet_third_party_loggers(*, verbose: bool) -> None:
    """Drop INFO lines like 'If your task is similar...' from transformers/paddlex."""
    if verbose:
        return
    for name in (
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "transformers.generation.configuration_utils",
        "paddlex",
        "paddle",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def find_pdfs(data_dir: Path) -> List[Path]:
    """
    Return all PDF files under data_dir (recursive), sorted for determinism.
    """
    return sorted(p for p in data_dir.rglob("*.pdf") if p.is_file())


def _print_paddle_device() -> None:
    """Log Paddle version and which device Place is active (cpu / gpu:N)."""
    try:
        import paddle  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[paddle_ocr] Could not import paddle for device info: {exc}", file=sys.stderr)
        return
    try:
        dev = paddle.device.get_device()
    except Exception:
        dev = "unknown"
    cuda_build = paddle.is_compiled_with_cuda()
    n_gpu = 0
    if cuda_build:
        try:
            n_gpu = int(paddle.device.cuda.device_count())
        except Exception:
            n_gpu = 0
    print(
        f"[paddle_ocr] Paddle {paddle.__version__} | default device: {dev} | "
        f"CUDA build: {cuda_build} | visible GPUs: {n_gpu}"
    )


def load_paddle_ocr_vl():
    """
    Import and construct the PaddleOCR-VL doc parser pipeline.

    This uses the official PaddleOCR integration as recommended in the
    PaddleOCR-VL model card on Hugging Face:
    https://huggingface.co/PaddlePaddle/PaddleOCR-VL
    """
    try:
        from paddleocr import PaddleOCRVL  # type: ignore
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "paddleocr":
            raise RuntimeError(
                "Cannot import `paddleocr` — it is not installed for the Python you are using.\n"
                f"  Current interpreter: {sys.executable}\n\n"
                "Fix (pick one):\n"
                "  • Use this project's venv (recommended):\n"
                "      source .venv/bin/activate\n"
                "      python -m src.ocr.paddle_ocr\n"
                "    or:  .venv/bin/python -m src.ocr.paddle_ocr\n"
                "  • Or install into this interpreter:\n"
                "      pip install 'paddleocr[doc-parser]'  # + paddlepaddle or paddlepaddle-gpu per requirements.txt\n"
                "See: https://huggingface.co/PaddlePaddle/PaddleOCR-VL"
            ) from exc
        raise
    except Exception as exc:  # pragma: no cover - import-time environment issue
        raise RuntimeError(
            "Failed to import PaddleOCRVL from paddleocr. "
            "Make sure you've installed PaddlePaddle + paddleocr[doc-parser] "
            "as described in the PaddleOCR-VL documentation."
        ) from exc

    # Default to v1 pipeline; callers can adjust if needed.
    return PaddleOCRVL(pipeline_version="v1")


def run_ocr_on_pdf(pipeline, pdf_path: Path, out_dir: Path) -> Path:
    """
    Run PaddleOCR-VL doc_parser on a single PDF and write JSON output.

    - pdf_path: input PDF
    - out_dir: directory where JSON will be saved

    Returns the path to the written JSON file.
    """
    # Use the same naming convention as PaddleOCR-VL's save_to_json, but
    # keep everything under our own out_dir.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.name}_by_PaddleOCR-VL.json"

    # The official API takes images/paths; for PDFs, PaddleOCR doc_parser
    # will internally rasterize each page.
    # See HF model card for reference usage:
    # https://huggingface.co/PaddlePaddle/PaddleOCR-VL
    results = pipeline.predict(str(pdf_path))

    # Each element in results can be saved via its own save_to_json method,
    # but for convenience we aggregate everything into a single JSON file
    # per PDF here.
    serialised = []
    for res in results:
        # res.save_to_json would write to disk; instead, ask it for a dict
        # representation via print() redirection or a custom to_dict attr
        # if available. To stay robust across versions, fall back to the
        # public save_to_json API and re-read the JSON if needed.
        #
        # To avoid depending on private attributes, we call save_to_json
        # into a temporary directory when needed. However, many PaddleOCR
        # result objects expose .to_dict() in recent versions; use it when
        # present.
        if hasattr(res, "to_dict"):
            serialised.append(res.to_dict())
        else:  # pragma: no cover - compatibility path
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                res.save_to_json(save_path=str(tmp))
                # Assume a single JSON file is written there
                json_files: Iterable[Path] = tmp.glob("*.json")
                for jf in json_files:
                    serialised.append(json.loads(jf.read_text(encoding="utf-8")))

    out_path.write_text(json.dumps(serialised, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR-VL doc_parser on all PDFs under a data directory "
            "and write one JSON file per PDF."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing PDF files (searched recursively).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Directory where JSON outputs are written. "
            "Defaults to the same root as data-dir."
        ),
    )
    parser.add_argument(
        "--verbose-warnings",
        action="store_true",
        help=(
            "Show Paddle tensor/decorator bridge warnings and INFO logs from "
            "transformers/paddlex (default: hide known-noise warnings only)."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _configure_warnings(verbose=args.verbose_warnings)
    _quiet_third_party_loggers(verbose=args.verbose_warnings)

    data_dir: Path = args.data_dir
    out_root: Path = args.out_dir

    if not data_dir.exists():
        print(f"[paddle_ocr] Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    pdfs = find_pdfs(data_dir)
    if not pdfs:
        print(f"[paddle_ocr] No PDF files found under {data_dir}", file=sys.stderr)
        return 0

    print(f"[paddle_ocr] Found {len(pdfs)} PDF(s) under {data_dir}")

    pipeline = load_paddle_ocr_vl()
    _print_paddle_device()

    for pdf in tqdm(pdfs, desc="PaddleOCR-VL PDFs", unit="pdf"):
        try:
            rel = pdf.relative_to(data_dir)
        except ValueError:
            # Fallback: flatten into out_root
            rel = pdf.name
        out_dir = out_root / Path(rel).parent
        tqdm.write(f"[paddle_ocr] OCR {pdf} -> {out_dir}")
        out_path = run_ocr_on_pdf(pipeline, pdf, out_dir)
        tqdm.write(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

