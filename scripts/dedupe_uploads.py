"""Store each uploaded clip once, however many jobs use it.

WHAT THIS IS INSTEAD OF
-----------------------
`uploads/` reached 7.8 GB and the obvious answer was a retention policy -
delete the clips of old jobs. That answer is worse than it looks: the
source footage is what a refine re-cuts from, so deleting it quietly
takes away the ability to change a video, permanently, for every job it
touches.

Measuring first showed the real shape. Of 7.80 GB in 76 files there were
NINE distinct clips: one 430 MB file stored twelve times, another sixty
times its own size across twenty-three copies. 7.03 GB of the 7.8 was
the same bytes written again.

So nothing needs deleting. Identical files become hardlinks to one copy:
every job keeps every file it had, every path still opens, refine still
works, and the duplicates stop occupying disk. The only thing lost is
the duplication.

WHY THIS IS SAFE HERE AND WOULD NOT BE EVERYWHERE
-------------------------------------------------
A hardlink means one inode with several names: writing through any name
changes what all of them see. That is a real hazard and the reason this
is not a general-purpose tool.

It is safe for THIS folder because nothing writes into `uploads/` after
the upload. The agents read the footage; the pipeline reads it; the
measure cache is keyed on its bytes precisely because they never change.
`--verify` re-reads every file afterwards and compares its hash against
what was recorded before the link was made.

    python scripts/dedupe_uploads.py              # report only
    python scripts/dedupe_uploads.py --apply
    python scripts/dedupe_uploads.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "uploads"
LEDGER = ROOT / "data" / "_dedupe_ledger.json"

# Below this there is nothing worth linking and the bookkeeping costs more
# than the space.
MIN_BYTES = 1_000_000


def _hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _same_file(a: pathlib.Path, b: pathlib.Path) -> bool:
    """Already one inode under two names."""
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino) and sa.st_ino != 0


def survey(root: pathlib.Path = UPLOADS) -> dict:
    """Group every upload by content. Reads only."""
    groups: dict[str, list[pathlib.Path]] = {}
    total = 0
    if not root.is_dir():
        return {"groups": {}, "total": 0, "files": 0}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size < MIN_BYTES:
            continue
        total += size
        groups.setdefault(f"{_hash(p)}:{size}", []).append(p)
    return {"groups": groups, "total": total,
            "files": sum(len(v) for v in groups.values())}


def _reclaimable(groups: dict) -> int:
    out = 0
    for paths in groups.values():
        if len(paths) < 2:
            continue
        keep = paths[0]
        for other in paths[1:]:
            if not _same_file(keep, other):
                out += other.stat().st_size
    return out


def report(root: pathlib.Path = UPLOADS) -> int:
    s = survey(root)
    g = s["groups"]
    free = _reclaimable(g)
    print(f"uploads: {s['total'] / 2**30:.2f} GB in {s['files']} file(s), "
          f"{len(g)} distinct clip(s)")
    print(f"already linked or unique: "
          f"{(s['total'] - free) / 2**30:.2f} GB")
    print(f"reclaimable by linking:   {free / 2**30:.2f} GB")
    if free:
        print()
        rows = sorted(((paths[0].stat().st_size, len(paths), paths[0].name)
                       for paths in g.values() if len(paths) > 1),
                      key=lambda r: -r[0] * (r[1] - 1))
        for size, n, name in rows[:8]:
            print(f"  {name[:34]:34} {size / 2**20:7.0f} MB x {n}")
        print()
        print("Nothing is deleted. Run with --apply to replace the "
              "duplicates with\nhardlinks to one copy; every path keeps "
              "working and every job keeps its clips.")
    return 0


def apply(root: pathlib.Path = UPLOADS) -> int:
    s = survey(root)
    ledger, freed, linked, skipped = {}, 0, 0, []
    for key, paths in s["groups"].items():
        if len(paths) < 2:
            continue
        keep = paths[0]
        for other in paths[1:]:
            if _same_file(keep, other):
                continue
            size = other.stat().st_size
            spare = other.with_suffix(other.suffix + ".relinking")
            try:
                # rename first: if the link fails the original is still
                # there under a name we can put back, so a crash in the
                # middle cannot lose a clip
                other.rename(spare)
                os.link(keep, other)
            except OSError as exc:
                if spare.exists() and not other.exists():
                    spare.rename(other)
                skipped.append(f"{other.name}: {exc}")
                continue
            if _hash(other) != key.split(":")[0]:   # pragma: no cover
                other.unlink()
                spare.rename(other)
                skipped.append(f"{other.name}: content differed after link")
                continue
            spare.unlink()
            ledger[str(other.relative_to(ROOT))] = {
                "same_as": str(keep.relative_to(ROOT)), "sha256": key}
            freed += size
            linked += 1

    if ledger:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        old = {}
        if LEDGER.exists():
            try:
                old = json.loads(LEDGER.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = {}
        LEDGER.write_text(json.dumps({**old, **ledger}, indent=2),
                          encoding="utf-8")

    print(f"linked {linked} duplicate(s), freed {freed / 2**30:.2f} GB")
    if skipped:
        print(f"left alone ({len(skipped)}):")
        for line in skipped[:8]:
            print(f"  {line}")
    if ledger:
        print(f"recorded in {LEDGER.relative_to(ROOT)}")
    return 0


def verify(root: pathlib.Path = UPLOADS) -> int:
    """Re-read every linked file and check it is what it was."""
    if not LEDGER.exists():
        print("nothing has been linked")
        return 0
    try:
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ledger unreadable: {exc}")
        return 1
    bad = []
    for rel, rec in ledger.items():
        p = ROOT / rel
        if not p.exists():
            bad.append(f"{rel}: gone")
            continue
        want = rec["sha256"].split(":")[0]
        if _hash(p) != want:
            bad.append(f"{rel}: content changed")
    print(f"checked {len(ledger)} linked file(s)")
    for line in bad:
        print(f"  FAIL {line}")
    print("every linked file still reads as the clip it was"
          if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="replace duplicates with hardlinks")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash every linked file")
    a = ap.parse_args()
    if a.verify:
        return verify()
    if a.apply:
        return apply() or verify()
    return report()


if __name__ == "__main__":
    sys.exit(main())
