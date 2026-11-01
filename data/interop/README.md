# Interoperability examples

Partial-tracking output of other programs, read by the tutorials in `tutorials_tech/` with `read_partiels`,
`read_sdif` and `read_chord_seq` (see [`docs/interop.md`](../../docs/interop.md)).

| File or folder | Sound | Made with | Holds |
|---|---|---|---|
| `saariaho_flute_pm2.sdif` | `samples/saariaho_flute.wav`, channel 0 | Partiels 2.6.0, Ircam Vamp Plugins 2.1.0 (PM2 5), template `PM2-Partials-Tracking-Harmonic-20`; written by AudioSpylt `write_sdif` | The full harmonic PM2 analysis as `1TRC` frames: 398 tracks, 17,310 points, 0.01-7.37 s, on PM2's own frame grid |
| `partiels/flute_json/`, `partiels/flute_csv/` | `samples/saariaho_flute.wav` (stereo) | Partiels 2.6.0, Ircam Vamp Plugins 2.1.0 (PM2 5), template `PM2-Partials-Tracking-Harmonic-20` | `Partial_1`, `_2`, `_3`, `_20` (PM2 partial ID 19) and the SuperVP f0 track `Feature_Scoring`, frames 140-179 (1.097-1.400 s), both channels |
| `partiels/cello_json/` | `samples/cello_arco_d_sul_c_01.wav` (dual mono) | as above, template `PM2-Partials-Tracking-Inharmonic-20` | `Partial_8`, `_9`, `_10`, `_12`, `_15`, `_18`, frames 570-609 (3.320-3.547 s), channel 0 |
| `sdif/kraft_snare_mono-trc_45hz.sdif` | mono snare drum (not included) | AudioSculpt (PM2, NVT `Pm2_Version_unknown`), 2012-07-22 | `1TRC` partial tracking: 71 partials, 320 points, 28 frames (the first empty, at -1.1e-5 s) |
| `sdif/pheek_bdrum-trc.sdif` | stereo bass drum (not included) | AudioSculpt (PM2), 2012-07-11 | `1TRC`: stream 0 (69 partials, 245 points) and stream 1 (64, 243), 27 frames each |
| `sdif/kraft_snare_mono-cs.sdif` | mono snare drum (not included) | AudioSculpt (PM2), 2012-07-22 | Chord sequence: 19 `1MRK` markers, 235 partials in 18 segments |
| `sdif/loris_flute_rbep.sdif` | `samples/saariaho_flute.wav`, left channel | loristrck 1.7.2 + pysdif3 1.0.1 (Loris, resolution 60 Hz, window 120 Hz), 2026-10-23 | `RBEP` breakpoints cut to 1.10-1.20 s: 205 partials, 757 breakpoints (209 and 778 before loristrck drops its last frame group) |

`saariaho_flute_pm2.sdif` is AudioSpylt's own export, not a file written by PM2 or AudioSculpt: it was made with
`write_sdif(read_partiels(EXPORT), path)` from the 19 `Group_1_Partial_*.json` files of the Partiels JSON export
(`read_partiels` skips the f0 track `Feature_Scoring`). Values are float32, as in PM2's files, and the SDIF
matrix index is `track_id + 1`. Its 1NVT table names the sound and the analysis.

The Partiels folders and the SDIF files are copies of the test fixtures in `tests/data/partiels/` and
`tests/data/sdif/`; the READMEs there give the Partiels command lines and the scripts that cut the files
(`trim.py`, `make_loris.py`). Partiels frames, values and JSON `track` descriptions are as exported. Inharmonic
PM2 in Partiels writes copies of points across its slots, the reason for `read_partiels(dedupe=True)`. The
three AudioSculpt analyses are the author's own (Egor Polyakov, 2012; see `LICENSE`), unchanged, contributed to
the project under its licence.
