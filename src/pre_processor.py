import os
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from PIL import Image



def preprocess_file(input_path: Path, output_dir: Path) -> Path:
    """
    Process a single input file (PDF / JPG / PNG) and output a JPG file.
    Returns the output JPG path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()

    # ---- PDF ----
    if suffix == ".pdf":
        # Always convert only page 1 for processing
        doc = fitz.open(input_path)
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {input_path}")

        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        out_path = output_dir / f"{input_path.stem}_page1.jpg"
        pix.save(out_path)
        doc.close()
        return out_path

    # ---- Image (jpg/png) ----
    elif suffix in [".jpg", ".jpeg", ".png"]:
        img = Image.open(input_path).convert("RGB")
        out_path = output_dir / f"{input_path.stem}.jpg"
        img.save(out_path, "JPEG", quality=95)
        return out_path

    else:
        raise ValueError(f"Unsupported file type: {input_path}")



class PreProcessor:
    """Batch processing class."""

    def __init__(self, input_dir: str, output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Path]:
        processed_paths = []

        for file in self.input_dir.iterdir():
            try:
                processed_paths.append(
                    preprocess_file(file, self.output_dir)
                )
            except Exception as e:
                print(f"[WARNING] Cannot process {file}: {e}")

        return processed_paths
