"""Make loris_flute_rbep.sdif: a Loris analysis of samples/saariaho_flute.wav cut to 1.10-1.20 s, as loristrck writes RBEP.

Needs loristrck 1.7.2, pysdif3 1.0.1 and soundfile (Python 3.12), outside the audiospylt venv:
    uv venv --python 3.12 LV && uv pip install --python LV/bin/python loristrck pysdif3 soundfile
    LV/bin/python -I tests/data/sdif/make_loris.py samples/saariaho_flute.wav tests/data/sdif/loris_flute_rbep.sdif
Prints the cut's partials and a few breakpoints (index, time, freq, amp, phase, bandwidth) for the tests.
"""
import importlib.util  # noqa: F401  (loristrck 1.7.2 write_sdif uses it without importing it)
import sys

import loristrck
import numpy as np
import soundfile

np.row_stack = np.vstack  # removed in numpy 2.5; loristrck's 1TRC writer still calls it
START, STOP = 1.10, 1.20

wav, out = sys.argv[1], sys.argv[2]
x, sr = soundfile.read(wav, always_2d=True)
partials = loristrck.analyze(np.ascontiguousarray(x[:, 0], dtype=float), sr, resolution=60, windowsize=120)
cut = [p[(p[:, 0] >= START) & (p[:, 0] <= STOP)] for p in partials]
cut = [p for p in cut if len(p)]
loristrck.write_sdif(cut, out, fmt="RBEP")
print(f"{len(partials)} partials in the full analysis; {len(cut)} in {START}-{STOP} s, "
      f"{sum(len(p) for p in cut)} breakpoints (columns time, freq, amp, phase, bandwidth)")
for index in (0, 1, len(cut) // 2):
    for row in cut[index][:2]:
        print(index, ", ".join(repr(float(v)) for v in row))
