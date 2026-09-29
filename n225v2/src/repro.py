"""第13章: environment.txt / data_hash.txt の出力。"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys

from .common import OUT, PROJ, RAW


def main():
    pkgs = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (OUT / "environment.txt").write_text(f"python {platform.python_version()}\nplatform {platform.platform()}\n\n{pkgs}")
    lines = []
    from .panel import target_codes
    want = target_codes()
    files = sorted([RAW / "data_j.xlsx", RAW / "jpx_list.parquet"] + list((RAW / "n225").glob("*"))
                   + [f for f in (RAW / "prices").glob("*.parquet") if f.stem in want])
    agg = hashlib.sha256()
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        agg.update(h.encode())
        lines.append(f"{h}  {f.relative_to(PROJ)}")
    (OUT / "data_hash.txt").write_text(f"# combined sha256 of all input files: {agg.hexdigest()}\n" + "\n".join(lines) + "\n")
    print("files hashed", len(files), agg.hexdigest())


if __name__ == "__main__":
    main()
