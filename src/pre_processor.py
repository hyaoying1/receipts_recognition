import os
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from PIL import Image


class PreProcessor:
    """
    Convert all input files (PDF / JPG / PNG) into unified JPG images.
    Produces clean, standardized images for classifier + extractor.
    """

    def __init__(self, input_dir: str, output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Path]:
        """
        Process all files in input_dir and return list of processed JPG paths.
        """
        processed_paths = []

        for file in self.input_dir.iterdir():
            if file.suffix.lower() in [".pdf"]:
                processed_paths.extend(self._process_pdf(file))

            elif file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                processed_paths.append(self._process_image(file))

            else:
                print(f"[WARNING] Unsupported file type: {file}")

        return processed_paths


    def _process_pdf(self, pdf_path: Path) -> List[Path]:
        """
        Convert each page of a PDF into a JPG image.
        """
        doc = fitz.open(pdf_path)
        output_paths = []

        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x for clarity
            output_path = self.output_dir / f"{pdf_path.stem}_page{i+1}.jpg"
            pix.save(output_path)
            output_paths.append(output_path)

        doc.close()
        return output_paths


    def _process_image(self, img_path: Path) -> Path:
        """
        Convert an image into JPG (even if already JPG, re-save for consistency).
        """
        img = Image.open(img_path).convert("RGB")
        output_path = self.output_dir / f"{img_path.stem}.jpg"
        img.save(output_path, "JPEG", quality=95)
        return output_path



if __name__ == "__main__":
    processor = PreProcessor(input_dir="data/raw", output_dir="data/processed")
    results = processor.run()
    print("\nProcessed files:")
    for r in results:
        print(" →", r)
