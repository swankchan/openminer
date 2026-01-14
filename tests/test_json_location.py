"""Test JSON file location logic."""
import sys
from pathlib import Path

from config import MINERU_OUTPUT_DIR

# Ensure repo root is importable when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Simulate a real scenario
pdf_path = REPO_ROOT / "uploads" / "ba056e7f-c732-4dbe-8954-69568ad6ee35_YYC-2.pdf"
output_dir = Path(MINERU_OUTPUT_DIR)
mineru_method = "auto"

# Calculate expected path
pdf_name = pdf_path.stem  # ba056e7f-c732-4dbe-8954-69568ad6ee35_YYC-2
method_to_folder = {
    "auto": "hybrid_auto",
    "ocr": "hybrid_ocr",
    "txt": "hybrid_txt",
}
hybrid_folder = method_to_folder.get(mineru_method, "hybrid_auto")
expected_json_path = output_dir / pdf_name / hybrid_folder

print("=" * 60)
print("JSON file location test")
print("=" * 60)
print(f"PDF file: {pdf_path}")
print(f"PDF name (stem): {pdf_name}")
print(f"MinerU method: {mineru_method}")
print(f"Hybrid folder: {hybrid_folder}")
print(f"\nExpected JSON path: {expected_json_path}")

# Check whether the actual path exists
actual_path = expected_json_path
if actual_path.exists():
    print("\n✓ Actual path exists")
    json_files = list(actual_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    for json_file in json_files[:5]:  # Show only the first 5
        print(f"  - {json_file.name}")
else:
    print("\n✗ Actual path does not exist")
