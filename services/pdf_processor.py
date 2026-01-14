"""PDF processing service."""
from pathlib import Path
import json
from typing import Dict, Any
from PyPDF2 import PdfReader
import pdf2image
from PIL import Image

from config import DEBUG as APP_DEBUG


class PDFProcessor:
    """Helper for working with PDF files."""
    
    def __init__(self):
        # Text-density threshold (not currently used directly; kept for future tuning).
        self.ocr_threshold = 0.1
    
    async def needs_ocr(self, pdf_path: Path) -> bool:
        """
        Check whether a PDF likely needs OCR.

        Simple heuristic: measure extracted text density.
        """
        try:
            reader = PdfReader(str(pdf_path))
            total_chars = 0
            total_pages = len(reader.pages)
            
            for page in reader.pages:
                text = page.extract_text() or ""
                total_chars += len(text)
            
            # If the average text per page is low, OCR may be needed.
            avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
            needs_ocr = avg_chars_per_page < 100  # if avg < 100 chars/page, OCR may be needed

            # Debug output (enable via DEBUG=True in .env).
            if APP_DEBUG:
                print(
                    "[OCR_CHECK] file="
                    f"{pdf_path}, pages={total_pages}, "
                    f"total_chars={total_chars}, "
                    f"avg_chars_per_page={avg_chars_per_page:.2f}, "
                    f"needs_ocr={needs_ocr}"
                )
            
            return needs_ocr
        except Exception as e:
            # If text extraction fails, OCR is likely needed.
            print(f"Error while checking OCR requirement: {e}")
            return True
    
    async def extract_text_to_json(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text directly from a PDF and convert to JSON (mocking MinerU output format).
        """
        try:
            reader = PdfReader(str(pdf_path))
            pages_data = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                pages_data.append({
                    "page_number": page_num,
                    "text": text,
                    "elements": [
                        {
                            "type": "text",
                            "content": text,
                            "bbox": [0, 0, 0, 0]  # simplified bounding box
                        }
                    ]
                })
            
            return {
                "document": {
                    "pages": pages_data,
                    "total_pages": len(pages_data)
                }
            }
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def convert_to_images(self, pdf_path: Path) -> list:
        """
        Convert a PDF to images (for OCR).
        """
        try:
            images = pdf2image.convert_from_path(str(pdf_path))
            return images
        except Exception as e:
            raise Exception(f"Error converting PDF to images: {str(e)}")

