"""Download the reference sounds of the fidelity benchmark and check them against reference.json.

Usage: python fetch_reference.py OUT [--pin]   (standard library only)

Every file of the manifest goes to OUT/reference/<save>; of a tarball only the listed members are kept. Files
already present with the right SHA-256 are skipped. The files are not committed to the repository: the ICMC 2000
panel files carry no licence statement and are fetched from the Wayback Machine's copy of the panel's site.
--pin fills in missing SHA-256 values in reference.json (used once, to write the manifest).
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import tarfile
import time
import urllib.error
import urllib.request

MANIFEST = pathlib.Path(__file__).resolve().parent / "reference.json"
TRIES, WAIT = 10, 30  # the Wayback Machine answers 503 or an HTML page for minutes at a time


def get(url: str) -> bytes:
    err = ""
    for attempt in range(1, TRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "audiospylt-fidelity-benchmark"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if not data[:100].lstrip().lower().startswith((b"<!doctype html", b"<html")):
                return data
            err = "an HTML page instead of the file"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = str(e)
        if attempt < TRIES:
            print(f"  {err}; retry {attempt} in {WAIT} s", flush=True)
            time.sleep(WAIT)
    raise SystemExit(f"giving up on {url}: {err}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(data: bytes, entry: dict, label: str, pin: bool) -> bool:
    """True when entry's hash was missing and is now filled in."""
    if entry.get("sha256") is None:
        if not pin:
            raise SystemExit(f"{label}: no SHA-256 in the manifest (run with --pin to record one)")
        entry["sha256"] = sha256(data)
        return True
    if sha256(data) != entry["sha256"]:
        raise SystemExit(f"{label}: SHA-256 {sha256(data)} does not match the manifest's {entry['sha256']}")
    return False


def present(path: pathlib.Path, entry: dict) -> bool:
    return entry.get("sha256") is not None and path.is_file() and sha256(path.read_bytes()) == entry["sha256"]


def save(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(out: pathlib.Path, pin: bool) -> None:
    manifest = json.loads(MANIFEST.read_text())
    root = out / "reference"
    pinned = 0
    for entry in manifest["files"]:
        members = entry.get("members")
        targets = [(root / m["save"], m) for m in members.values()] if members else [(root / entry["save"], entry)]
        if not (pin and entry.get("sha256") is None) and all(present(p, e) for p, e in targets):
            continue
        print("fetch", entry["url"], flush=True)
        data = get(entry["url"])
        pinned += check(data, entry, entry["url"], pin)
        if not members:
            save(root / entry["save"], data)
            continue
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for name, m in members.items():
                fh = tar.extractfile(name)
                if fh is None:
                    raise SystemExit(f"{entry['url']}: {name} is not a regular file")
                member = fh.read()
                pinned += check(member, m, name, pin)
                save(root / m["save"], member)
    if pinned:
        MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        print(f"pinned {pinned} SHA-256 values in {MANIFEST.name}")
    n = sum(len(e.get("members", {})) or 1 for e in manifest["files"])
    print(f"{n} files in {root}, all match the manifest")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), "--pin" in sys.argv[2:])
