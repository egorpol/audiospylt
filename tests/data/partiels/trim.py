"""Cut the Partiels PM2 test fixtures in this folder from full exports (see README.md).

Usage: python tests/data/partiels/trim.py FLUTE_EXPORT CELLO_EXPORT
where each EXPORT folder holds the ``csv/`` and ``json/`` subfolders written by the Partiels CLI.
Frames, files and channels are kept as they are in the export; nothing else is changed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

# fixture folder -> (export, formats, files, first frame, frame count, channels)
CUTS = {
    "flute": ("FLUTE", ("csv", "json"),
              ("Group_1_Partial_1", "Group_1_Partial_2", "Group_1_Partial_3", "Group_1_Partial_20",
               "Group_1_Feature_Scoring"), 140, 40, (0, 1)),
    "cello": ("CELLO", ("json",),
              ("Group_1_Partial_8", "Group_1_Partial_9", "Group_1_Partial_10", "Group_1_Partial_12",
               "Group_1_Partial_15", "Group_1_Partial_18"), 570, 40, (0,)),
}


def cut_csv(text: str, first: int, count: int, channels: tuple[int, ...]) -> str:
    blocks = text.rstrip("\n").split("\n\n")
    out = []
    for channel in channels:
        lines = blocks[channel].split("\n")
        header = lines[:1] if lines[0].startswith("TIME") else []
        rows = lines[len(header):]
        out.append("\n".join(header + rows[first:first + count]))
    return "\n\n".join(out) + "\n"


def cut_json(text: str, first: int, count: int, channels: tuple[int, ...]) -> str:
    document = json.loads(text)
    document["results"] = [document["results"][c][first:first + count] for c in channels]
    return json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"


def main(flute: Path, cello: Path) -> None:
    sources = {"FLUTE": flute, "CELLO": cello}
    for name, (export, formats, files, first, count, channels) in CUTS.items():
        for fmt in formats:
            target = HERE / f"{name}_{fmt}"
            target.mkdir(exist_ok=True)
            for stem in files:
                source = sources[export] / fmt / f"{stem}.{fmt}"
                cut = cut_csv if fmt == "csv" else cut_json
                text = cut(source.read_text(encoding="utf-8"), first, count, channels)
                (target / source.name).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
