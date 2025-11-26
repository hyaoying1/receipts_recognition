import os
from glob import glob
import subprocess
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "artifacts" / "docx"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "images"

DPI = 150  # 清晰度，200/300 都行，看你显存和需求


# 如果系统里是 `soffice`，就写 "soffice"；如果是 `libreoffice` 就改成 "libreoffice"
LIBREOFFICE_BIN = "/Applications/LibreOffice.app/Contents/MacOS/soffice"  # 或 "libreoffice"
PDFTOPPM_BIN = "pdftoppm"


def run_cmd(cmd):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print("  [ERR]", result.stderr.strip())
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    else:
        if result.stdout.strip():
            print("  [OUT]", result.stdout.strip())
    return result


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def docx_to_pdf(docx_path: str, pdf_out_dir: str) -> str:
    """
    使用 LibreOffice 无头模式将 docx 转为 pdf
    """
    ensure_dir(pdf_out_dir)

    cmd = [
        LIBREOFFICE_BIN,
        "--headless",
        "--nologo",
        "--norestore",
        "--convert-to", "pdf",
        "--outdir", pdf_out_dir,
        docx_path,
    ]
    run_cmd(cmd)

    base = os.path.splitext(os.path.basename(docx_path))[0]
    pdf_path = os.path.join(pdf_out_dir, base + ".pdf")
    if not os.path.exists(pdf_path):
        # 兜底：某些版本大小写/名称可能稍有不同
        from glob import glob as _glob
        candidates = _glob(os.path.join(pdf_out_dir, base + "*.pdf"))
        if not candidates:
            raise FileNotFoundError(f"PDF not found after conversion: {pdf_path}")
        pdf_path = candidates[0]

    return pdf_path


def pdf_to_jpg(pdf_path: str, jpg_out_dir: str, dpi: int = 200):
    """
    使用 pdftoppm 将 pdf 转成 jpg：
    - 单页：<base>.jpg
    - 多页：<base>_page_1.jpg, ...
    """
    ensure_dir(jpg_out_dir)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    output_prefix = os.path.join(jpg_out_dir, base)

    cmd = [
        PDFTOPPM_BIN,
        "-jpeg",
        "-r",
        str(dpi),
        pdf_path,
        output_prefix,
    ]
    run_cmd(cmd)

    from glob import glob as _glob
    generated = sorted(_glob(output_prefix + "-*.jpg"))
    if not generated:
        raise FileNotFoundError(f"No JPG generated for {pdf_path}")

    jpg_paths = []

    if len(generated) == 1:
        # 单页：直接改名为 base.jpg
        old_path = generated[0]
        new_path = os.path.join(jpg_out_dir, f"{base}.jpg")
        os.rename(old_path, new_path)
        jpg_paths.append(new_path)
        print("  生成 JPG:", new_path)
    else:
        # 多页：base_page_1.jpg ...
        for idx, old_path in enumerate(generated, start=1):
            new_path = os.path.join(jpg_out_dir, f"{base}_page_{idx}.jpg")
            os.rename(old_path, new_path)
            jpg_paths.append(new_path)
            print("  生成 JPG:", new_path)

    return jpg_paths


def main():
    input_dir = os.path.abspath(INPUT_DIR)
    output_dir = os.path.abspath(OUTPUT_DIR)

    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 输出目录: {output_dir}")

    ensure_dir(output_dir)

    docx_files = sorted(glob(os.path.join(input_dir, "*.docx")))
    print(f"[INFO] 共发现 {len(docx_files)} 个 DOCX")

    if not docx_files:
        print("[WARN] 没有发现 docx，直接结束。")
        return

    # 临时 pdf 目录
    tmp_pdf_dir = tempfile.mkdtemp(prefix="docx2pdf_")
    print(f"[INFO] 临时 PDF 目录: {tmp_pdf_dir}")

    try:
        for idx, docx_path in enumerate(docx_files, start=1):
            base = os.path.splitext(os.path.basename(docx_path))[0]

            # ==== 增量逻辑：如果已经有对应 JPG，就跳过 ====
            single_jpg = os.path.join(output_dir, f"{base}.jpg")
            multi_jpg_pattern = os.path.join(output_dir, f"{base}_page_*.jpg")

            existed_single = os.path.exists(single_jpg)
            existed_multi = bool(glob(multi_jpg_pattern))

            if existed_single or existed_multi:
                print(
                    f"\n=== ({idx}/{len(docx_files)}) 处理: {docx_path} ===\n"
                    f"[SKIP] 已存在对应 JPG，跳过此 docx"
                )
                continue

            print(f"\n=== ({idx}/{len(docx_files)}) 处理: {docx_path} ===")

            # 1) docx -> pdf（无头，不开窗口）
            pdf_path = docx_to_pdf(docx_path, tmp_pdf_dir)

            # 2) pdf -> jpg
            pdf_to_jpg(pdf_path, output_dir, dpi=DPI)

            # 如果确定不需要 pdf，可以顺手删掉
            # os.remove(pdf_path)
    finally:
        # 不想保留中间 pdf：删临时目录
        shutil.rmtree(tmp_pdf_dir, ignore_errors=True)
        print(f"[INFO] 已删除临时 PDF 目录: {tmp_pdf_dir}")


if __name__ == "__main__":
    main()