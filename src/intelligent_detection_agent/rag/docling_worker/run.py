"""只接收本地任务文件；不依赖主项目的 Torch/FastAPI 环境。"""
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    from build import build_pdf_dataset

    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    build_pdf_dataset(
        pdf_file=spec["pdf"], output_dir=spec["output"],
        page_range=tuple(spec["page_range"]), force_ocr=spec["force_ocr"],
        min_image_area=900,
    )
