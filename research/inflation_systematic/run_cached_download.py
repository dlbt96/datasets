#!/usr/bin/env python3
"""Run the public-data builder while reusing archives predownloaded by CI."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).with_name("download_external_public.py")
spec = importlib.util.spec_from_file_location("inflation_downloader", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_download = module.download


def cached_download(url: str, path: Path, *, optional: bool = False, timeout: int = 180):
    path = Path(path)
    if path.exists() and path.stat().st_size > 20:
        data = path.read_bytes()
        module.MANIFEST.append({
            "url": url,
            "path": str(path.relative_to(module.ROOT)),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "status": "predownloaded_cache",
        })
        print(f"using cached archive {path} ({len(data):,} bytes)", flush=True)
        return path
    return original_download(url, path, optional=optional, timeout=timeout)


module.download = cached_download
module.main()
