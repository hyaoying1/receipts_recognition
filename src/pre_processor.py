import os
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from PIL import Image



def preprocess_file(input_path: Path, output_dir: Path) -> List[Path]:
    """
    处理一个输入文件 (PDF / JPG / PNG)，返回这个文档对应的所有 JPG 页面路径列表。

    - 单页 PDF：导出 1 张 jpg，文件名 = {stem}.jpg
    - 多页 PDF：每页导出 1 张 jpg，文件名 = {stem}_page{idx}.jpg
    - 图片：导出 1 张 jpg，文件名 = {stem}.jpg

    返回：List[Path]  (这个文档的所有页面)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    out_paths: List[Path] = []

    # ---- PDF ----
    if suffix == ".pdf":
        doc = fitz.open(input_path)
        n_pages = len(doc)
        if n_pages == 0:
            doc.close()
            raise ValueError(f"PDF has no pages: {input_path}")

        if n_pages == 1:
            # 单页 PDF：只导出第一页，文件名不加 _page1
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            out_path = output_dir / f"{input_path.stem}.jpg"
            pix.save(out_path)
            out_paths.append(out_path)
        else:
            # 多页 PDF：每一页导出为 {stem}_page{idx+1}.jpg
            for page_idx in range(n_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                out_path = output_dir / f"{input_path.stem}_page{page_idx + 1}.jpg"
                pix.save(out_path)
                out_paths.append(out_path)

        doc.close()
        return out_paths

    # ---- Image (jpg/png) ----
    elif suffix in [".jpg", ".jpeg", ".png"]:
        img = Image.open(input_path).convert("RGB")
        out_path = output_dir / f"{input_path.stem}.jpg"
        img.save(out_path, "JPEG", quality=95)
        out_paths.append(out_path)
        return out_paths

    else:
        raise ValueError(f"Unsupported file type: {input_path}")



class PreProcessor:
    """Batch processing class (文档级)."""

    def __init__(self, input_dir: str, output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[List[Path]]:
        """
        返回值：List[List[Path]]
        - 外层 list：每个元素是一份文档
        - 内层 list：该文档的所有页面 jpg 路径
        """
        processed_docs: List[List[Path]] = []

        for file in self.input_dir.iterdir():
            try:
                pages: List[Path] = preprocess_file(file, self.output_dir)
                processed_docs.append(pages)
            except Exception as e:
                print(f"[WARNING] Cannot process {file}: {e}")

        return processed_docs
