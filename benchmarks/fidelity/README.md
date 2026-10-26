# Resynthesis fidelity benchmark

This benchmark measures how close a resynthesis comes to the original sound for the three case-study excerpts.
Each row analyses an excerpt and renders it back to audio. The rows are AudioSpylt v1's per-frame peak
resynthesis, AudioSpylt 0.6's tutorial render path, Loris and sms-tools with their own synthesizers, and the
tracks of Loris, sms-tools and PM2 (via Partiels) played through `audiospylt.synthesize`. The scores are a
log-spectral distance per STFT bin, a log-spectral distance per frequency band, and the level of the render.
The evaluator also writes level-matched WAVs for listening.

The external tools are not part of AudioSpylt's dependencies or CI. Results from the run of 2026-10-26 are in
[`results/`](results/).

## Excerpts

The excerpts are the files in [`samples/`](../../samples/README.md): 44.1 kHz, stereo, 24 bit, analysed as
mono (the mean of the two channels). They are copyrighted recordings, included for research and teaching; the
listening WAVs are written outside the repository and are not distributed.

| excerpt | duration | peak / RMS (dBFS) | content |
| --- | ---: | --- | --- |
| `bayle` | 7.69 s | −11.6 / −23.6 | François Bayle; broadband, 160 Hz–5 kHz within 14 dB, dense (283 peaks per frame at −80 dBFS with the matched analysis) |
| `de_natura_son` | 10.00 s | −10.7 / −20.0 | Bernard Parmegiani; sub-bass rumble: the band levels fall by 50–66 dB from 20–80 Hz to the bands above 1.3 kHz |
| `saariaho_flute` | 7.38 s | −24.2 / −36.2 | Kaija Saariaho, *NoaNoa*, flute; strongest band 640–1280 Hz |

PM2 needs a harmonic or an inharmonic template. The flute is the only harmonic excerpt, so it gets the
harmonic template and the other two get the inharmonic one.

## Rows and parameters

"Matched" means a Hann window of 2048 samples (46.4 ms) and a hop of 256 samples (5.8 ms) for every tool, as in
the [accuracy benchmark](../accuracy/README.md).

| row | analysis | synthesis |
| --- | --- | --- |
| `audiospylt-v1/default` | AudioSpylt 1.0.0.dev0: `spectrogram` defaults (frame 2048, hop 512, Hann, `zero_pad=1`, centred frames), `find_peaks(min_db=-60)` per frame | `synthesize` defaults: continuous phase, rows linked within `link_cents=100`, `link_ramp=0.005`, `fade=0` |
| `audiospylt-v1/matched` | frame 2048, hop 256, Hann, `zero_pad=4`; `find_peaks(min_db=-80)` | as above |
| `audiospylt-v1/matched-fade` | as `matched` | as above with `fade=0.0025` (the setting of the G1 listening check) |
| `audiospylt-0.6/matched` | 0.6 API (`dft_analysis.analyze_signal`, kept as a deprecated shim) per frame: frame 2048, hop 256, Hann, frame k centred at k·hop, `thresh_amp_low=2.5e-5` (−80 dBFS on 0.6's scale), `thresh_amp_high=1` | the path of `tutorials_tech/resynth.ipynb`: `plot_waves(k=0.00015, phase_mode="chirp", edge_fade_s=0.0025)`, then `render_audio(fs_target_name="44.1kHz", alias_above_nyquist=True)` |
| `loris/matched` | loristrck 1.7.2: `resolution` 43.07 Hz, `windowsize` 86.13 Hz, `hoptime` 256/44100 s, `ampfloor` −80 | `loristrck.synthesize(start=0, end=duration)` |
| `loris/matched+asp` | the same tracks | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `sms-tools/matched` | sms-tools 1.2 `sineModelAnal`: `np.hanning(2049)`, N 8192, H 256, t −86.02 (= −80 dBFS), `maxnSines` 100, `minSineDur` 0.01 s | `sineModelSynth`, Ns 1024, H 256 |
| `sms-tools/matched+asp` | the same tracks | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `pm2-harmonic/matched+asp` | Partiels 2.6.0 with Ircam Vamp Plugins 2.1.0 (PM2 v5), `PM2-Partials-Tracking-Harmonic-20.ptldoc` with window 46.44 ms, overlap 8x (hop 5.805 ms), FFT oversampling 4x, Hann; read with `read_partiels` | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `pm2-inharmonic/matched+asp` | same with `PM2-Partials-Tracking-Inharmonic-20.ptldoc`; five runs | as above |

Notes on the rows:

- AudioSpylt v1 has no partial tracker. Every peak above `min_db` in every frame becomes an event held over the
  frame centre ± hop/2, and `synthesize` joins events of consecutive frames into one phase-continuous voice
  when their frequencies are within 100 cents. The matched rows keep 283 events per frame on `bayle`, 14 on
  `de_natura_son` and 59 on the flute. For comparison, Loris keeps 245, 11 and 50 track points per frame,
  sms-tools 88, 14 and 55, PM2 inharmonic 11–12 on `bayle` (20 partials at most) and PM2 harmonic 18 on the
  flute. A v1 row is a sum of all spectral peaks of every frame, closer to an STFT resynthesis than to a
  compact model of the sound.
- With `fade=0`, an event that does not link starts and stops at full amplitude. On `de_natura_son`, the peaks
  of the 20–80 Hz rumble (bins 1–4 of a 2048-sample frame, 21.5 Hz each) move by more than 100 cents from
  frame to frame, so few of them link, and each one switches on and off every 5.8 ms. The result is broadband clicks: in `audiospylt-v1/matched`,
  every band above 354 Hz has a band LSD between 18.9 and 42.1 dB. `fade=0.0025` removes the clicks
  (`matched-fade`), but it also fades every unlinked event inside its own 5.8 ms span, which raises the flute's
  band LSD from 6.46 to 11.71 dB.
- `audiospylt-v1/default` has hop 512, no zero padding and a −60 dB threshold, so it keeps 37, 4 and 9 events
  per frame.
- `loristrck.synthesize` starts its output at the first breakpoint unless `start` is given (83 ms into
  `bayle`, 16 ms into the flute). The runner passes `start=0` and the excerpt's duration so that the render
  lines up with the excerpt. Every other renderer starts at time 0.
- The 0.6 row synthesises on a grid of k = 0.15 ms (6.67 kHz), so partials above 3.33 kHz fold down.
  `render_audio` interpolates linearly up to 44.1 kHz and normalises the peak to full scale. That is why its
  gain is +9.5 to +21 dB: the excerpts peak between −24.2 and −10.7 dBFS. See [Level](#level) for the scores
  at matched level.
- The `+asp` rows replay frequency and amplitude only. Loris's own synthesizer also uses the analysed phase and
  bandwidth, and sms-tools' uses the analysed phase. Comparing a tool's own row with its `+asp` row therefore
  isolates the synthesizer, not the analysis.
- `sms-tools/matched` keeps at most 100 sines per frame. On `bayle` that leaves the bands above 9 kHz 8–16 dB
  low over the excerpt, and the top band (18–22 kHz) empty in 42 % of the frames.
- PM2 inharmonic output differs from run to run on the same file (as in the accuracy benchmark). The row ran
  five times (`r1` … `r5`); `summary.md` shows the mean and the min–max spread. On `bayle`, with 11–12
  partials per frame, many frames have no partial in a given band: in run 1, the render is at least 40 dB
  below the excerpt in 891–1122 Hz in 42 % of the frames, in 1782–2245 Hz in 63 %, and in each band above
  5.7 kHz in 51–100 %. On `de_natura_son`, it misses most of the sub-bass that carries the excerpt's energy:
  the 0–177 Hz band is 22 dB low (band LSD 31 dB), and the gain is −21.6 dB.
- Partiels has no synthesizer, so the PM2 rows exist only as `+asp` rows.

## Metrics (`evaluate.py`)

Each render is cut or zero-padded to the length of the excerpt and compared with the mono excerpt. Both go
through the same STFT: `scipy.signal.stft`, Hann 2048, hop 512, `boundary=None`, `padded=False`.

- **Bin LSD.** Magnitudes are floored at 10⁻⁴ × the loudest reference bin (−80 dB), in the reference and in
  the render. The score is the RMS over all bins and frames of the difference in dB. This is the definition
  of the 2026-10-10 audit and of `tests/test_synth.py`, so the numbers can be compared with the 0.6 ones.
- **Band LSD.** Band powers (sums of |X|²) in third-octave bands with edges at 1 kHz · 2^(n/3 + 1/6). Bands
  are merged upward until each holds at least 8 STFT bins (172 Hz), and the remainder up to Nyquist joins the
  last band, which gives 18 bands. Powers are floored at 10⁻⁸ × the loudest reference band (−80 dB). The
  score is the RMS over all bands and frames of the difference in dB.
- **Gain.** RMS level of the render re the excerpt, in dB.
- `per_band.csv` gives both LSDs for each of the 18 bands as well (the bin LSD over the bins of that band).

Renders are scored as written, with no gain fit, so a level error is part of both LSDs.

### Noise-pair floor

LSD compares magnitudes frame by frame. Two independent white-noise realisations of the same level (10 s, the
same STFT) score **7.89 dB bin LSD and 1.86 dB band LSD**. Any noise-like part of an excerpt therefore costs
about 8 dB of bin LSD even when the model reproduces the noise's spectrum exactly. Merging bins into bands
averages most of that randomness away, so the band LSD is the more telling number for noisy material. The
floor is listed as the `noise-pair` row in `per_excerpt.csv` and `per_band.csv`.

A third metric was tried and dropped: LSD over tonal regions only (reference peaks with a prominence of
≥ 10 dB, present in the neighbouring frames). Its mask still passes 7 % of the bins of white noise, and it
did not separate the rows better than the bin LSD.

### Floor effects

Bins below the bin-LSD floor in both signals count as a difference of 0 dB. On `de_natura_son`, 74.5 % of the
reference bins lie below the floor (40.2 % on the flute, 1.5 % on `bayle`), so its bin LSD mostly reads the
loud low bands, and a render that is silent where the reference is quiet scores well. Its band LSD is less
affected. Compare rows within one excerpt, not across excerpts.

### Level

Because the 0.6 renderer normalises its output, the level difference dominates that row's scores. With the
render scaled to the RMS of the excerpt, the 0.6 row scores (bin / band LSD in dB) 17.33 / 11.38 on `bayle`,
3.51 / 9.49 on `de_natura_son` and 10.00 / 8.81 on the flute. These values come from `evaluate.score` on the
rescaled render; they are not in `results/`. Every other row is within −3.6 to +0.2 dB of the excerpt, except
PM2 inharmonic on `de_natura_son` (see above).

## Headline results

These tables are copied from [`results/summary.md`](results/summary.md) (run of 2026-10-26). Lower is better.
"–" means the row has no render of that excerpt. The PM2 inharmonic row shows the mean of five runs and the
min–max spread.

**Bin LSD (dB)**; noise-pair floor 7.89:

| row | bayle | de_natura_son | saariaho_flute |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | 18.23 | 16.97 | 10.34 |
| audiospylt-v1/matched | 6.19 | 22.20 | 6.51 |
| audiospylt-v1/matched-fade | 6.51 | 2.93 | 7.03 |
| audiospylt-0.6/matched | 16.76 | 5.33 | 17.34 |
| loris/matched | 7.48 | 3.11 | 7.08 |
| loris/matched+asp | 10.58 | 3.19 | 8.99 |
| sms-tools/matched | 20.32 | 2.17 | 7.31 |
| sms-tools/matched+asp | 20.95 | 2.47 | 7.78 |
| pm2-harmonic/matched+asp | – | – | 10.27 |
| pm2-inharmonic/matched+asp (5 runs) | 27.77 (27.51–28.01) | 4.03 (3.98–4.07) | – |

**Band LSD (dB, 18 bands)**; noise-pair floor 1.86:

| row | bayle | de_natura_son | saariaho_flute |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | 5.96 | 23.56 | 14.26 |
| audiospylt-v1/matched | 1.14 | 29.98 | 6.46 |
| audiospylt-v1/matched-fade | 1.12 | 9.21 | 11.71 |
| audiospylt-0.6/matched | 12.82 | 13.17 | 21.88 |
| loris/matched | 2.02 | 10.11 | 12.39 |
| loris/matched+asp | 2.49 | 10.47 | 14.04 |
| sms-tools/matched | 11.67 | 8.71 | 12.23 |
| sms-tools/matched+asp | 12.07 | 9.71 | 12.73 |
| pm2-harmonic/matched+asp | – | – | 16.92 |
| pm2-inharmonic/matched+asp (5 runs) | 30.74 (28.18–32.85) | 12.05 (11.90–12.18) | – |

**Gain (dB re excerpt RMS)**:

| row | bayle | de_natura_son | saariaho_flute |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | -0.16 | -1.60 | -0.18 |
| audiospylt-v1/matched | -0.17 | 0.21 | -0.08 |
| audiospylt-v1/matched-fade | -0.20 | -1.47 | -0.09 |
| audiospylt-0.6/matched | 9.55 | 10.61 | 20.99 |
| loris/matched | -0.47 | -3.62 | -0.44 |
| loris/matched+asp | -0.49 | -3.62 | -0.46 |
| sms-tools/matched | -0.65 | -2.03 | -0.44 |
| sms-tools/matched+asp | -0.39 | -1.81 | -0.18 |
| pm2-harmonic/matched+asp | – | – | -0.40 |
| pm2-inharmonic/matched+asp (5 runs) | -0.87 (-1.00–-0.80) | -21.59 (-21.83–-21.39) | – |

In short:

- With the matched analysis, v1's per-frame peaks give the lowest LSDs on `bayle` and the flute, because they
  keep every peak above −80 dBFS. On `de_natura_son` they need `fade=0.0025` to avoid clicks, which costs the
  flute 5.3 dB of band LSD.
- Loris is the closest tracker on `bayle` (band LSD 2.02 dB, 0.9 dB above v1 with 14 % fewer points) and on
  the flute's bin LSD. sms-tools is the closest on `de_natura_son`, where it also beats every v1 row. On
  `bayle`, the sms-tools and PM2 limits on the number of partials show up as empty bands.
- Played through `asp.synthesize` (frequency and amplitude only), the same tracks score 0.1–3.1 dB higher
  than with the tool's own synthesizer.

## Listening

`evaluate.py` writes `OUT/listening/<excerpt>/NN_<row>.wav`: `00_original` (the mono excerpt) and one file
per row, using the first run of the PM2 inharmonic row. Every file is scaled to the RMS of the excerpt, then
one factor per excerpt puts the loudest peak of the set at −1 dBFS, so the files of one excerpt can be
compared at equal loudness. They are 16-bit WAVs, about 22 MB in total.

## Reproducing

The commands are for bash on Linux and run from the repository root. Each tool needs its own environment:
sms-tools 1.2 requires `numpy<2`, and loristrck is not an AudioSpylt dependency. `run_audiospylt.py` runs
last because it also replays the track tables that the other runners write.

```sh
OUT=$HOME/asp-fidelity     # renders, track tables, listening WAVs; ~110 MB
VENVS=$HOME/asp-bench-venvs

uv venv --python 3.12 $VENVS/asp
uv pip install --python $VENVS/asp/bin/python '.[plot]'   # the 0.6 shims import plotly
uv venv --python 3.12 $VENVS/loris
uv pip install --python $VENVS/loris/bin/python loristrck==1.7.2 soundfile
uv venv --python 3.12 $VENVS/sms
uv pip install --python $VENVS/sms/bin/python sms-tools==1.2 'numpy<2' soundfile

A=$VENVS/asp/bin/python
$VENVS/loris/bin/python benchmarks/fidelity/run_loris.py $OUT     # ~6 s
$VENVS/sms/bin/python benchmarks/fidelity/run_sms.py $OUT         # ~6 s
$A benchmarks/fidelity/run_partiels.py $OUT TEMPLATE_DIR [PARTIELS]   # optional, see below
$A benchmarks/fidelity/run_audiospylt.py $OUT                     # ~2.5 min
$A benchmarks/fidelity/evaluate.py $OUT benchmarks/fidelity/results   # ~2 s
```

Every runner writes `OUT/raw/<row>/<excerpt>.wav` (float32, at the level the tool renders) and
`OUT/raw/<row>/meta.json` (version, parameters, synthesizer). The Loris, sms-tools and PM2 runners also write
their tracks to `OUT/tracks/<row>/<excerpt>.csv` in the format of the accuracy benchmark (time, freq, amp,
track). `evaluate.py` scores whatever renders are present, so the PM2 step can be skipped. On 2026-10-26, a
run from fresh environments, with the PM2 track tables copied in, reproduced every file in `results/` exactly.

### Partiels and PM2

Partiels 2.6.0 and the Ircam Vamp Plugins 2.1.0 are installed as for the
[accuracy benchmark](../accuracy/README.md#partiels-and-pm2); `TEMPLATE_DIR` is the folder with the two PM2
`.ptldoc` templates from the plugins' manuals. The runner reuses that benchmark's template editing and export
code: it writes the matched templates to `OUT/partiels-templates/`, the mono excerpts it analyses to
`OUT/mono/`, and runs the inharmonic template five times per excerpt.

## Files

| file | role |
| --- | --- |
| `excerpts.py` | excerpt list, mono loading, render and metadata writers (numpy and soundfile only, imported by every runner) |
| `run_audiospylt.py` | the v1 and 0.6 rows, and the `+asp` replays of every track table in `OUT/tracks/` |
| `run_loris.py`, `run_sms.py`, `run_partiels.py` | one runner per external tool |
| `evaluate.py` | scoring, listening WAVs, writes `results/` (no AudioSpylt import) |
| `results/summary.md` | generated tables per excerpt and the noise-pair floor |
| `results/per_excerpt.csv` | bin LSD, band LSD and gain per row and excerpt, every PM2 run separately |
| `results/per_band.csv` | both LSDs per row, excerpt and band |
| `results/repeats.csv` | mean, min and max over the PM2 inharmonic runs |
