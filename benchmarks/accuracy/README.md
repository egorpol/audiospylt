# Partial-analysis accuracy benchmark

This benchmark measures how closely AudioSpylt and three other analysis tools recover the partials of synthetic
signals whose frequencies and amplitudes are known exactly at every instant. Each tool runs with its documented
defaults (or shipped template) and with a matched setting (Hann window of about 46 ms, hop 5.8 ms). AudioSpylt
0.6 runs with the matched setting only, as a reference for what v1 changed. The scores are frequency error in
cents, amplitude error in dB, recall and false detections.

The external tools are not part of AudioSpylt's dependencies or CI. Results from the run of 2026-10-23 are in
[`results/`](results/).

## Signals

`suite.py` writes 13 files: 44.1 kHz, float32 mono, 2 s, 10 ms raised-cosine fades, peak ≤ 0.9. Starting phases
come from a fixed seed per file. The truth is analytic (`truth.py`), so the true frequency and amplitude of every
partial can be evaluated at any time. Amplitudes are sine peak amplitudes re full scale.

| family | files | content |
| --- | --- | --- |
| stationary | `stat_harm`, `stat_inharm` | 12 harmonics of 220 Hz, amps 0.25/k; 5 inharmonic sines 223.37, 457.81, 911.29, 1843.43, 3299.61 Hz, amps 0.12 … 0.04 |
| glissando | `gliss_600`, `gliss_1200` | 6 harmonics, f0 exponential 220 → 440 Hz (600 c/s) and 220 → 880 Hz (1200 c/s) |
| vibrato | `vib_50`, `vib_100` | 8 harmonics, f0 440 Hz ± 50 c and ± 100 c at 5.5 Hz |
| close pair | `pair_5`, `pair_10`, `pair_20`, `pair_40` | 440 Hz + (440 + Δ) Hz, Δ = 5, 10, 20, 40 Hz, amps 0.4 each |
| noise | `noise_20`, `noise_10`, `noise_0` | 12 harmonics of 220 Hz (amps 0.1/k) + white noise at 20, 10, 0 dB SNR re the total partial power |

## Rows and parameters

A row is one tool with one setting. Output times are frame centres, or the times a tool reports for its track
points.

| row | tool | parameters |
| --- | --- | --- |
| `audiospylt-v1/default` | AudioSpylt 1.0.0.dev0 (this repository) | `spectrogram` defaults: frame 2048, hop 512, Hann, `zero_pad=1`, centred frames; `find_peaks(min_db=-60)` per frame (QIFFT on dB) |
| `audiospylt-v1/matched` | AudioSpylt 1.0.0.dev0 | frame 2048, hop 256, Hann, `zero_pad=4` (8192-point FFT); `find_peaks(min_db=-80)` |
| `audiospylt-0.6/matched` | AudioSpylt 0.6 API (`dft_analysis.analyze_signal`, kept as a deprecated shim) | frame 2048, hop 256, Hann, frame k centred at k·hop (half a frame of zeros at both ends); `thresh_amp_low=2.5e-5` (−80 dBFS on 0.6's scale, see below), `thresh_amp_high=1` |
| `loris/default` | Loris via loristrck 1.7.2 | `resolution` = 0.9 × the file's lowest true frequency (the rule of thumb in the loristrck docs); everything else at its defaults (window size derived from the resolution, hop = 1/window size, amplitude floor −90 dB) |
| `loris/matched` | loristrck 1.7.2 | `resolution` 43.07 Hz, `windowsize` 86.13 Hz (main-lobe width of a 2048-sample Hann), `hoptime` 256/44100 s, `ampfloor` −80 |
| `sms-tools/default` | sms-tools 1.2, `sineModelAnal` | the settings of its `tests/test_models_ground_truth.py` (the package ships no other example): `np.hanning(1025)`, N 2048, H 128, t −80, `maxnSines` 25, `minSineDur` 0.01 s; `freqDevOffset` 20 Hz and `freqDevSlope` 0.01 (signature defaults) |
| `sms-tools/matched` | sms-tools 1.2 | `np.hanning(2049)`, N 8192, H 256, t −86.02 (= −80 dBFS), `maxnSines` 100, `minSineDur` 0.01 s |
| `pm2-inharmonic/template` | Partiels 2.6.0 with Ircam Vamp Plugins 2.1.0 (PM2 v5), `PM2-Partials-Tracking-Inharmonic-20.ptldoc` | as shipped: Hann, window 23.22 ms (1024 samples), overlap 4x (hop 5.805 ms), FFT oversampling 1x, 20 partial tracks, maximum 20 partials, amplitude threshold 90, smoothing time difference and minimum duration 0 |
| `pm2-inharmonic/matched` | same | template with window 46.44 ms (2048 samples), overlap 8x (hop 5.805 ms), FFT oversampling 4x, Hann; rest unchanged |
| `pm2-harmonic/template` | same, `PM2-Partials-Tracking-Harmonic-20.ptldoc` | as shipped: Hann, window 23.22 ms, overlap 3x (hop 7.755 ms), FFT oversampling 1x, 19 partial tracks driven by a SuperVP F0 track (50–1000 Hz), maximum 20 partials, amplitude threshold 90 |
| `pm2-harmonic/matched` | same | window 46.44 ms, overlap 8x, FFT oversampling 4x, Hann; rest unchanged |

Notes on the rows:

- AudioSpylt v1 has no partial tracker. Its rows contain every spectral peak above `min_db` in every frame, so
  side lobes and noise peaks show up as extra points ("far %") and as false detections. A tracker would prune most
  of them.
- `sms-tools/matched` and `audiospylt-v1/matched` run the same analysis: `np.hanning(2049)` is a periodic Hann
  of 2048 samples followed by one zero, and both use an 8192-point FFT, hop 256, centred frames and parabolic
  interpolation on dB. Their error statistics are therefore identical. sms-tools reports fewer points because
  its tracking drops short and weak peaks.
- PM2 inharmonic output differs from run to run on the same file. Both PM2 inharmonic rows ran five times
  (`r1` … `r5`). `summary.md` shows the mean over the runs and the min–max spread. PM2 harmonic is
  deterministic and runs only on the 8 harmonic files (it needs an F0), so it has no close-pair results.
- `loris/default` uses the true lowest frequency of each file to set the resolution. That follows the
  documented rule of thumb but uses knowledge of the truth.
- The close-pair spacings (5–40 Hz) are below the main-lobe width of every window used here (86 Hz for a
  2048-sample Hann at 44.1 kHz). Family recall stays between 31 % and 63 % for every row.

### Amplitude scales

All points are converted to sine peak amplitude re full scale before scoring. sms-tools reports
20·log10(A/2), so its values get +6.02 dB. A threshold `t` in sms dB is therefore 6.02 dB lower in dBFS: the
default −80 is −74 dBFS. Two offsets stay in the error because they are what the tools report:

- AudioSpylt 0.6 returns `rfft/N` magnitudes without the one-sided ×2 and without the window's coherent gain.
  A Hann-windowed sine of amplitude A reads A/4 (−12.04 dB). v1 corrects this.
- PM2 with the shipped templates reads stationary sines 0.33 dB high (median). With the matched window the
  median error is under 0.001 dB.

## Scoring (`evaluate.py`)

The first and last 0.1 s of each file are skipped. The tolerance for a true partial is min(50 c, half the gap to
its nearest neighbouring partial), so tolerances never overlap.

**Accuracy** compares every found point with the true partials at the point's own time; nothing is resampled.

- Points that share a time (one frame) pair one-to-one, loudest first. Each point takes the nearest true partial
  if that partial lies within tolerance and no louder point of the frame has taken it.
- A point pairs only if its amplitude is at most 20 dB below the true amplitude (`GATE_DB`). Without this gate,
  the side lobe of a partial that a tool missed would count as a hit.
- Frequency error is in cents. Amplitude error is in dB re the true amplitude.
- Unpaired points are counted as `dup` (a louder point of the frame took the partial) or `far` (no partial
  within tolerance, or gated out).

**Detection** is scored on a common 10 ms grid.

- A track covers [first point − hop/2, last point + hop/2] and is interpolated linearly inside its span, with
  values held at its ends. An untracked frame peak covers its own frame ± hop/2.
- At each grid time, true partials and covering points pair one-to-one, closest in cents first, within the
  tolerance and the amplitude gate.
- Recall is the share of true partials that pair. False detections are unpaired points at or above −60 dBFS,
  given per grid frame.
- The hop is the nominal hop for frame rows and sms-tools. For Loris and PM2 it is the median time step within
  tracks (`meta.json` of each row).

### Columns of `results/summary.md`

| column | meaning |
| --- | --- |
| points | found points inside the guard with freq > 0 and amp > 0 |
| paired %, dup %, far % | shares of those points (sum to 100) |
| \|Δf\| p50, p95 (c) | median and 95th percentile of the absolute frequency error of paired points |
| ΔA median (dB) | median signed amplitude error of paired points (bias) |
| \|ΔA\| p95 (dB) | 95th percentile of the absolute amplitude error |
| recall % | paired true partials / all true partials on the 10 ms grid |
| false / frame | unpaired covering points at or above −60 dBFS per grid frame |

`per_signal.csv` and `per_family.csv` hold the same metrics per file and per family, plus the signed median
frequency error (`freq_bias_c`). `repeats.csv` holds the mean, min and max over the PM2 inharmonic runs.

## Headline results

These tables are copied from [`results/summary.md`](results/summary.md) (run of 2026-10-23). Rows marked
"mean of 5" average five PM2 inharmonic runs. "–" means the row has no files in that family.

**|Δf| p95 (cents)** per family:

| row | stationary | glissando | vibrato | close pair | noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| audiospylt-0.6/matched | 37.1 | 31.1 | 26.1 | 47.3 | 37.1 |
| audiospylt-v1/default | 2.44 | 1.43 | 2.81 | 38.5 | 8.53 |
| audiospylt-v1/matched | 0.0594 | 0.0373 | 3.28 | 38.1 | 7.16 |
| loris/default | 0.0294 | 0.0324 | 0.111 | 38 | 26.3 |
| loris/matched | 0.00231 | 0.326 | 7.25 | 33.8 | 44.4 |
| pm2-harmonic/matched | 0.131 | 0.0884 | 3.31 | – | 6.71 |
| pm2-harmonic/template | 43 | 21.4 | 16.3 | – | 34.5 |
| pm2-inharmonic/matched (mean of 5) | 0.126 | 0.0943 | 3.31 | 26.4 | 5.82 |
| pm2-inharmonic/template (mean of 5) | 35 | 20.5 | 15.1 | 36.8 | 32.7 |
| sms-tools/default | 0.921 | 0.43 | 0.815 | 38.2 | 13.8 |
| sms-tools/matched | 0.0594 | 0.0373 | 3.28 | 38.1 | 7.27 |

**Recall %** per family:

| row | stationary | glissando | vibrato | close pair | noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| audiospylt-0.6/matched | 94.1 | 97.9 | 100.0 | 42.4 | 100.0 |
| audiospylt-v1/default | 100.0 | 100.0 | 100.0 | 51.4 | 100.0 |
| audiospylt-v1/matched | 100.0 | 100.0 | 100.0 | 60.5 | 100.0 |
| loris/default | 100.0 | 100.0 | 100.0 | 35.6 | 91.8 |
| loris/matched | 100.0 | 100.0 | 94.4 | 40.7 | 99.9 |
| pm2-harmonic/matched | 100.0 | 100.0 | 100.0 | – | 98.9 |
| pm2-harmonic/template | 99.4 | 99.3 | 100.0 | – | 98.0 |
| pm2-inharmonic/matched (mean of 5) | 76.5 | 85.0 | 90.4 | 31.3 | 72.8 |
| pm2-inharmonic/template (mean of 5) | 96.1 | 91.9 | 95.5 | 36.0 | 70.1 |
| sms-tools/default | 100.0 | 100.0 | 100.0 | 37.4 | 71.5 |
| sms-tools/matched | 100.0 | 100.0 | 100.0 | 62.6 | 98.0 |

On the stationary files, the median amplitude error of every row is within ±0.13 dB, except AudioSpylt 0.6
(−12.5 dB, its uncorrected scale) and the two PM2 templates (+0.33 dB). Read high recall together with
`false / frame` in `summary.md`: the per-frame peak rows reach high recall but also report many extra points.

## Reproducing

The commands are for bash on Linux and run from the repository root. Each tool needs its own environment:
sms-tools 1.2 requires `numpy<2`, and loristrck is not an AudioSpylt dependency.

```sh
OUT=$HOME/asp-bench        # suite, raw rows, Partiels scratch; ~100 MB
VENVS=$HOME/asp-bench-venvs

uv venv --python 3.12 $VENVS/asp
uv pip install --python $VENVS/asp/bin/python '.[plot]'   # the 0.6 shim imports plotly
uv venv --python 3.12 $VENVS/loris
uv pip install --python $VENVS/loris/bin/python loristrck==1.7.2 soundfile
uv venv --python 3.12 $VENVS/sms
uv pip install --python $VENVS/sms/bin/python sms-tools==1.2 'numpy<2' soundfile

A=$VENVS/asp/bin/python
$A benchmarks/accuracy/suite.py $OUT                              # < 1 s
$A benchmarks/accuracy/run_audiospylt.py $OUT                     # ~12 s
$VENVS/loris/bin/python benchmarks/accuracy/run_loris.py $OUT     # ~6 s
$VENVS/sms/bin/python benchmarks/accuracy/run_sms.py $OUT         # ~10 s
$A benchmarks/accuracy/run_partiels.py $OUT TEMPLATE_DIR [PARTIELS]   # ~3 min, optional
$A benchmarks/accuracy/evaluate.py $OUT benchmarks/accuracy/results   # ~1 min with the PM2 rows
```

Every runner writes `OUT/raw/<row>/<signal>.csv` (columns time, freq, amp, track; track −1 for untracked frame
peaks) and `OUT/raw/<row>/meta.json` (parameters, version, hop). `evaluate.py` scores whatever rows are present,
so the PM2 step can be skipped. On 2026-10-23, a run from fresh environments reproduced every non-PM2 value in
`per_family.csv` exactly.

### Partiels and PM2

1. Install Partiels 2.6.0 from the [Partiels releases](https://github.com/Ircam-Partiels/Partiels). By
   default the runner calls `/opt/Partiels/Partiels`; pass another path as the third argument.
2. Install the Ircam Vamp Plugins 2.1.0 from
   [ircam-vamp-plugins](https://github.com/Ircam-Partiels/ircam-vamp-plugins) into a Vamp plugin path
   (`~/vamp` works on Linux).
3. From the same 2.1.0 release, unpack `Ircam-Vamp-Plugins-Manuals.zip`. Pass the folder that holds
   `PM2-Partials-Tracking-Inharmonic-20.ptldoc` and `PM2-Partials-Tracking-Harmonic-20.ptldoc` as `TEMPLATE_DIR`.
   The templates are not redistributed here.

The runner writes the matched templates to `OUT/partiels-templates/`. It changes `windowsize`,
`windowoverlapping`, `fftoversampling` and `windowtype` in every PM2 track. It exports JSON with
`Partiels --export` and reads the export with `audiospylt.read_partiels`, which drops the F0 and spectrogram
tracks and the points duplicated across slots. A missing template makes the Partiels CLI hang, so the runner
checks for both template files first.

## Files

| file | role |
| --- | --- |
| `truth.py` | signal definitions, analytic truth, tolerance rule (numpy only, imported by every runner) |
| `suite.py` | writes `OUT/suite/*.wav` and `truth.json` |
| `points.py` | the shared raw format (numpy only) |
| `run_audiospylt.py`, `run_loris.py`, `run_sms.py`, `run_partiels.py` | one runner per tool |
| `evaluate.py` | scoring, writes `results/` |
| `results/summary.md` | generated tables per family and the repeat spread |
| `results/per_signal.csv`, `per_family.csv`, `repeats.csv` | all metrics |
