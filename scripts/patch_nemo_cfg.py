#!/usr/bin/env python3
"""Patch a NeMo .nemo archive to disable CUDA-graph decoding.

Why: NeMo's RNNT greedy decoder probes CUDA-graph support through
``cuda-python``. On ROCm there is no ``libcuda.so.1``, so the probe raises
``Failed to dlopen libcuda.so.1`` -- an error type NeMo's own fallback does
not catch -- and model loading dies. Setting
``decoding.greedy.use_cuda_graph_decoder: false`` in the archive's
``model_config.yaml`` avoids the probe entirely.

Usage:
    patch_nemo_cfg.py <input.nemo> <output.nemo>
"""

import sys
import tarfile
from pathlib import Path

import yaml


CONFIG_NAME = "model_config.yaml"


def patch_config(raw: bytes) -> bytes:
    cfg = yaml.safe_load(raw)
    decoding = cfg.setdefault("decoding", {})
    greedy = decoding.setdefault("greedy", {})
    greedy["use_cuda_graph_decoder"] = False
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False).encode()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        return 1

    with tarfile.open(src, "r") as fin:
        members = fin.getmembers()
        payload = {m.name: fin.extractfile(m).read() for m in members if m.isfile()}

    if CONFIG_NAME not in payload:
        print(f"{src} has no {CONFIG_NAME}; not a NeMo archive?", file=sys.stderr)
        return 1

    payload[CONFIG_NAME] = patch_config(payload[CONFIG_NAME])

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tarfile.open(tmp, "w") as fout:
        for m in members:
            if m.isfile() and m.name in payload:
                data = payload[m.name]
                m.size = len(data)
                out = tarfile.TarInfo(name=m.name)
                out.size = len(data)
                out.mode = m.mode
                out.mtime = m.mtime
                out.uid, out.gid = m.uid, m.gid
                out.uname, out.gname = m.uname, m.gname
                out.type = tarfile.REGTYPE
                fout.addfile(out, __import__("io").BytesIO(data))
            else:
                fout.addfile(m)

    tmp.replace(dst)
    print(f"patched {dst} ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
