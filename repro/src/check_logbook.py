"""Validate the published Trackio evidence tree without running experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / ".trackio" / "logbook"
IMAGE_RE = re.compile(r"!\[[^]]*\]\((images/[^)]+)\)")


def walk(nodes: list[dict[str, object]]) -> list[Path]:
    pages: list[Path] = []
    for node in nodes:
        filename = node.get("file")
        if isinstance(filename, str):
            pages.append(ROOT / filename)
        children = node.get("children", [])
        if not isinstance(children, list):
            raise ValueError(f"children is not a list for {node.get('slug')!r}")
        pages.extend(walk(children))
    return pages


def main() -> int:
    manifest = json.loads((ROOT / "logbook.json").read_text())
    root = manifest["root"]
    if not isinstance(root, dict):
        raise ValueError("manifest root is not an object")
    pages = walk([root])
    missing = [page.relative_to(ROOT) for page in pages if not page.is_file()]
    for page in pages:
        if not page.is_file():
            continue
        for image in IMAGE_RE.findall(page.read_text()):
            image_path = ROOT / image
            if not image_path.is_file():
                missing.append(image_path.relative_to(ROOT))
    if missing:
        print("LOGBOOK CHECK FAILED: missing files")
        for item in missing:
            print(f"  - {item}")
        return 1
    print(f"LOGBOOK CHECK PASSED: {len(pages)} pages and all referenced images exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
