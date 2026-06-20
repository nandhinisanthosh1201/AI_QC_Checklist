"""
metadata_writer.py
------------------
Responsible for persisting the page-level metadata produced by Stage 1
as a single, well-structured JSON file.

Public API
----------
save_metadata(records, output_dir, filename)  → Path
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


def save_metadata(
    records: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
) -> Path:
    """
    Serialise all page metadata records to a JSON file.

    The written document has the shape::

        {
            "stage": "stage1_pdf_to_images",
            "generated_at": "<ISO-8601 UTC timestamp>",
            "total_pages": <int>,
            "pages": [
                {
                    "pdf_name": "architectural.pdf",
                    "page_number": 1,
                    "image_path": "C:\\...\\architectural_page_1.png"
                },
                ...
            ]
        }

    Parameters
    ----------
    records : list[dict]
        Metadata records returned by :func:`pdf_converter.convert_pdf`.
    output_dir : Path
        Directory where the JSON file will be written (created if absent).
    filename : str
        Name of the output JSON file.

    Returns
    -------
    Path
        Absolute path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    payload: dict[str, Any] = {
        "stage": "stage1_pdf_to_images",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_pages": len(records),
        "pages": records,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    log.info("Metadata saved → %s  (%d records)", output_path.resolve(), len(records))
    return output_path
