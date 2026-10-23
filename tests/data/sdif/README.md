# SDIF partial tracks (test fixtures)

Real SDIF files for `tests/test_interop_sdif.py`. Synthetic files (1HRM with gaps, RBEP offsets, errors) are
built inside the tests.

| File | Content | Made with |
|---|---|---|
| `kraft_snare_mono-trc_45hz.sdif` | `1TRC` partial tracking of a mono snare drum: 71 partials, 320 points, 28 frames (the first empty, at -1.1e-5 s) | AudioSculpt (PM2, NVT `Pm2_Version_unknown`), 2012-07-22 |
| `pheek_bdrum-trc.sdif` | `1TRC` of a stereo bass drum: stream 0 (69 partials, 245 points) and stream 1 (64, 243), 27 frames each | AudioSculpt (PM2), 2012-07-11 |
| `kraft_snare_mono-cs.sdif` | Chord sequence of the snare: 19 `1MRK` markers, 235 partials in 18 segments | AudioSculpt (PM2), 2012-07-22 |
| `loris_flute_rbep.sdif` | `RBEP` breakpoints of a Loris analysis of `samples/saariaho_flute.wav` (left channel, resolution 60 Hz, window 120 Hz), cut to 1.10-1.20 s: 209 partials, 778 breakpoints before loristrck drops its last frame group | loristrck 1.7.2 + pysdif3 1.0.1, 2026-10-23, `make_loris.py` |

The three AudioSculpt analyses are the author's own (Egor Polyakov, 2012; see `LICENSE`), unchanged, contributed to the
project under its licence. Their source sounds are not included.

`make_loris.py` rebuilds the Loris file (instructions in its docstring) and prints the breakpoints the tests
check. loristrck 1.7.2 never writes its last frame group, so the file holds fewer than 778 breakpoints; the
script also works around two loristrck bugs (`importlib.util` used without import, `np.row_stack` removed in
numpy 2.5).
