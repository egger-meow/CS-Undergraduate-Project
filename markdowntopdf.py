import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def which(cmd: str) -> Optional[str]:
 """Return full path to executable if found, else None."""
 return shutil.which(cmd)


def ensure_abs(path: Path) -> Path:
 return path if path.is_absolute() else (Path.cwd() / path).resolve()


def md_to_html(md_text: str, base_dir: Path) -> str:
    """Convert Markdown to HTML with decent defaults and a base href for assets."""
    try:
        import markdown  # type: ignore
    except Exception:
        # Minimal fallback if markdown package is missing
        # Wrap raw text in <pre> so at least content is visible
        escaped = (md_text
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   )
        body = f"<pre>{escaped}</pre>"
        head = ""
    else:
        # Use common extensions if available
        extensions = [
            "extra",
            "admonition",
            "codehilite",
            "fenced_code",
            "tables",
            "toc",
        ]
        body = markdown.markdown(md_text, extensions=extensions)  # type: ignore
        head = (
            "<meta charset='utf-8'>"
        )

    css = _default_css()
    base = f"<base href='{base_dir.as_uri()}/'>"
    html = f"""
<!doctype html>
<html>
  <head>
    {base}
    {head}
    <style>
    {css}
    </style>
  </head>
  <body class="markdown-body">
    {body}
  </body>
</html>
"""
    return html


def _default_css() -> str:
    # Lightweight GitHub-like styling
    return """
 body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, Noto Sans, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol, Noto Color Emoji; line-height: 1.6; color: #24292f; }
 .markdown-body { max-width: 900px; margin: 0 auto; padding: 2rem; }
 h1, h2, h3, h4 { border-bottom: 1px solid #eaecef; padding-bottom: .3rem; }
 pre, code { background: #f6f8fa; }
 pre { padding: 12px; overflow: auto; }
 code { padding: 2px 4px; }
 table { border-collapse: collapse; width: 100%; }
 th, td { border: 1px solid #dfe2e5; padding: 6px 13px; }
 img { max-width: 100%; }
 """


def try_pdfkit(html: str, output_pdf: Path, base_dir: Path) -> Tuple[bool, str]:
    """Try to render HTML to PDF using pdfkit (wkhtmltopdf required)."""
    try:
        import pdfkit  # type: ignore
    except Exception as e:
        return False, f"pdfkit not available: {e}"

    wkhtmltopdf = which("wkhtmltopdf")
    if not wkhtmltopdf:
        return False, "wkhtmltopdf not found in PATH."

    cfg = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf)
    options = {
        "enable-local-file-access": None,  # flag without value
        "quiet": "",
        "page-size": "A4",
        "margin-top": "10mm",
        "margin-right": "10mm",
        "margin-bottom": "10mm",
        "margin-left": "10mm",
    }
    try:
        pdfkit.from_string(html, str(output_pdf), configuration=cfg, options=options)
        return True, "ok"
    except Exception as e:
        return False, f"pdfkit failed: {e}"


def try_pandoc(input_md: Path, output_pdf: Path, resource_dir: Path) -> Tuple[bool, str]:
    """Try to convert using pandoc. Prefers wkhtmltopdf if available to avoid LaTeX dependency."""
    pandoc = which("pandoc")
    if not pandoc:
        return False, "pandoc not found in PATH."

    wkhtml = which("wkhtmltopdf")
    # Build command
    cmd = [pandoc, str(input_md), "-o", str(output_pdf)]
    if wkhtml:
        cmd += ["--pdf-engine", "wkhtmltopdf"]
    # Help pandoc find local images
    cmd += ["--resource-path", str(resource_dir)]
    # Improve default look a bit
    cmd += ["-V", "margin-left=10mm", "-V", "margin-right=10mm", "-V", "margin-top=10mm", "-V", "margin-bottom=10mm"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and output_pdf.exists():
            return True, "ok"
        return False, f"pandoc failed (code {completed.returncode}): {completed.stderr.strip()}"
    except Exception as e:
        return False, f"pandoc error: {e}"


def convert_markdown_to_pdf(input_md: Path, output_pdf: Path) -> None:
    input_md = ensure_abs(input_md)
    output_pdf = ensure_abs(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not input_md.exists():
        raise FileNotFoundError(f"Input markdown not found: {input_md}")

    # Strategy 1: pdfkit (wkhtmltopdf)
    try:
        md_text = input_md.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        md_text = input_md.read_text(errors="ignore")

    html = md_to_html(md_text, base_dir=input_md.parent)
    ok, msg = try_pdfkit(html, output_pdf, base_dir=input_md.parent)
    if ok:
        print(f"PDF written: {output_pdf}")
        return
    else:
        print(f"[info] pdfkit path not used: {msg}")

    # Strategy 2: pandoc (prefer wkhtmltopdf if present)
    ok, msg = try_pandoc(input_md, output_pdf, resource_dir=input_md.parent)
    if ok:
        print(f"PDF written: {output_pdf}")
        return
    else:
        print(f"[info] pandoc path not used: {msg}")

    # If we reach here, we couldn't render
    help_msg = (
        "Failed to convert Markdown to PDF. Install one of the following and rerun:\n"
        "1) wkhtmltopdf + pdfkit (recommended):\n"
        "   - Install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html\n"
        "   - pip install pdfkit markdown\n"
        "2) pandoc (with wkhtmltopdf or a LaTeX engine):\n"
        "   - Install pandoc: https://pandoc.org/installing.html\n"
        "   - Optionally install wkhtmltopdf or a TeX distribution (MiKTeX/TeX Live)\n"
    )
    raise RuntimeError(help_msg)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF.")
    parser.add_argument("input", nargs="?", default=str(Path("docs") / "README.md"), help="Path to input Markdown (default: docs/README.md)")
    parser.add_argument("-o", "--output", default=None, help="Path to output PDF (default: same folder/name as input with .pdf)")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    in_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_suffix(".pdf")

    try:
        convert_markdown_to_pdf(in_path, out_path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
