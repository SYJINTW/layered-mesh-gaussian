"""CLI: python -m lmg_format {deflate,inflate,verify,inspect} ..."""
import argparse
import json
import os
import sys
import tempfile

import torch

from .codec import PARAMS_NAME, deflate, inflate, verify

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _inspect(bundle):
    path = os.path.join(bundle, PARAMS_NAME) if os.path.isdir(bundle) else bundle
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    meta = header.pop("__metadata__", {})
    for k in sorted(meta):
        print("  %-16s %s" % (k, meta[k]))
    print("  %-16s %s" % ("arrays", ""))
    for k in sorted(header):
        info = header[k]
        lo, hi = info["data_offsets"]
        print("    %-16s %-12s %-8s %10d B" % (k, tuple(info["shape"]), info["dtype"], hi - lo))


def main(argv=None):
    p = argparse.ArgumentParser(prog="lmg_format")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deflate", help="Full checkpoint dir -> Lean bundle")
    d.add_argument("full_dir")
    d.add_argument("-o", "--out", required=True)
    d.add_argument("--mesh", required=True)
    d.add_argument("--link-mesh", action="store_true",
                   help="symlink mesh.ply instead of copying (local use only)")

    i = sub.add_parser("inflate", help="Lean bundle -> Full checkpoint dir")
    i.add_argument("bundle")
    i.add_argument("-o", "--out", required=True)
    i.add_argument("--mesh", default=None)
    i.add_argument("--device", default=DEFAULT_DEVICE)
    i.add_argument("--no-check-mesh", action="store_true")

    v = sub.add_parser("verify", help="deflate -> inflate -> compare")
    v.add_argument("full_dir")
    v.add_argument("--mesh", required=True)
    v.add_argument("--device", default=DEFAULT_DEVICE)
    v.add_argument("--tol", type=float, default=1e-6)
    v.add_argument("--keep", default=None, help="keep intermediates in this dir")

    s = sub.add_parser("inspect", help="print a bundle's header")
    s.add_argument("bundle")

    a = p.parse_args(argv)
    if a.cmd == "deflate":
        meta = deflate(a.full_dir, a.out, a.mesh, link_mesh=a.link_mesh)
        print(json.dumps(meta, indent=2))
    elif a.cmd == "inflate":
        print(inflate(a.bundle, a.out, mesh_path=a.mesh, device=a.device,
                      check_mesh=not a.no_check_mesh))
    elif a.cmd == "inspect":
        _inspect(a.bundle)
    else:
        tmp = a.keep or tempfile.mkdtemp(prefix="lmg_verify_")
        os.makedirs(tmp, exist_ok=True)
        r = verify(a.full_dir, a.mesh, tmp, device=a.device, tol=a.tol)
        print(a.full_dir)
        print("  variant=%s barycentric=%s scale=%s N=%s F=%s" % (
            r["meta"]["model_variant"], r["meta"]["barycentric_mode"],
            r["meta"]["scale_mode"], r["meta"]["num_splats"], r["meta"]["num_faces"]))
        for k, ok in sorted(r["exact"].items()):
            print("  exact   %-24s %s" % (k, "OK" if ok else "MISMATCH"))
        for k, dv in sorted(r["derived"].items()):
            print("  derived %-24s max|diff| = %.3g" % (k, dv))
        sz = r["sizes"]
        print("  size    full %.2f MB -> lean %.2f MB  (%.2fx)" % (
            sz["full_bytes"] / 1e6, sz["lean_bytes"] / 1e6, sz["ratio"]))
        print("  ==> %s" % ("PASS" if r["ok"] else "FAIL"))
        return 0 if r["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
