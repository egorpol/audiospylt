# Partial-analysis accuracy benchmark

This benchmark measures how closely AudioSpylt and three other analysis tools recover the partials of synthetic
signals whose frequencies and amplitudes are known exactly at every instant. Each tool runs with its documented
defaults (or shipped template) and with a matched setting (Hann window of about 46 ms, hop 5.8 ms). AudioSpylt
0.6 runs with the matched setting only, as a reference for what v1 changed. The scores are frequency error in
cents, amplitude error in dB, recall and false detections.

The external tools are not part of AudioSpylt's dependencies or CI. Results from the run of 2026-11-04 are in
[`results/`](results/).

## Signals

`suite.py` writes 21 files: 44.1 kHz, float32 mono, 2 s, 10 ms raised-cosine fades, peak ≤ 0.9. Starting phases
come from a fixed seed per file. The truth is analytic (`truth.py`), so the true frequency and amplitude of every
partial can be evaluated at any time. Amplitudes are sine peak amplitudes re full scale.

| family | files | content |
| --- | --- | --- |
| stationary | `stat_harm`, `stat_inharm` | 12 harmonics of 220 Hz, amps 0.25/k; 5 inharmonic sines 223.37, 457.81, 911.29, 1843.43, 3299.61 Hz, amps 0.12 … 0.04 |
| glissando | `gliss_600`, `gliss_1200` | 6 harmonics, f0 exponential 220 → 440 Hz (600 c/s) and 220 → 880 Hz (1200 c/s) |
| vibrato | `vib_50`, `vib_100` | 8 harmonics, f0 440 Hz ± 50 c and ± 100 c at 5.5 Hz |
| close pair | `pair_5`, `pair_10`, `pair_20`, `pair_40` | 440 Hz + (440 + Δ) Hz, Δ = 5, 10, 20, 40 Hz, amps 0.4 each |
| noise | `noise_20`, `noise_10`, `noise_0` | 12 harmonics of 220 Hz (amps 0.1/k) + white noise at 20, 10, 0 dB SNR re the total partial power |
| crossing | `cross_fast`, `cross_slow` | two exponential sweeps crossing at 1 s, with fixed partial identities |
| onset/offset | `stagger_harm`, `stagger_inharm` | staggered starts and ends of 6 harmonics or 5 inharmonic sines |
| dropout | `drop_30`, `drop_80` | 6 harmonics with staggered 30 or 80 ms dropouts |
| tremolo | `trem_20`, `trem_40` | 8 harmonics with in-phase 6 Hz tremolo, troughs at −20 or −40 dB |

The separate dev suite has perturbed signals from every family. It is for tuning the tracker. The published
suite never tunes anything. `suite.py OUT --dev` writes only `OUT/dev/suite/`; `OUT/dev` can also be scored as
an ordinary `OUT` when raw rows there list dev names in `meta["signals"]`.

## Rows and parameters

A row is one tool with one setting. Output times are frame centres, or the times a tool reports for its track
points.

| row | tool | parameters |
| --- | --- | --- |
| `audiospylt-v1/default` | AudioSpylt 1.0.0.dev0 (this repository) | `spectrogram` defaults: frame 2048, hop 512, Hann, `zero_pad=1`, centred frames; `find_peaks(min_db=-60)` per frame (QIFFT on dB) |
| `audiospylt-v1/matched` | AudioSpylt 1.0.0.dev0 | frame 2048, hop 256, Hann, `zero_pad=4` (8192-point FFT); `find_peaks(min_db=-80)` |
| `audiospylt-v1/tracked` | AudioSpylt 1.0.0.dev0 | `track_partials(x, sr)` defaults (Hann, frame 2048, hop 256, `zero_pad=4`, `min_db=-80`, prominence 20 dB, 50 c, `max_gap=0`, `min_duration=0.1`), recorded from the signature in `meta.json` |
| `audiospylt-v1/tracked-bh` | AudioSpylt 1.0.0.dev0 | the same with `window="blackmanharris"` |
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

- AudioSpylt v1's default and matched rows are untracked per-frame peaks. They contain every spectral peak
  above `min_db`, so side lobes and noise peaks show up as extra points ("far %") and as false detections.
  The tracked rows use `track_partials`.
- `sms-tools/matched` and `audiospylt-v1/matched` run the same analysis: `np.hanning(2049)` is a periodic Hann
  of 2048 samples followed by one zero, and both use an 8192-point FFT, hop 256, centred frames and parabolic
  interpolation on dB. Their error statistics are therefore identical. sms-tools reports fewer points because
  its tracking drops short and weak peaks.
- PM2 inharmonic output differs from run to run on the same file. Both PM2 inharmonic rows ran five times
  (`r1` … `r5`). `summary.md` shows the mean over the runs and the min–max spread. PM2 harmonic is
  deterministic and runs only on the 13 harmonic files (it needs an F0), so it has no close-pair or
  crossing results.
- `loris/default` uses the true lowest frequency of each file to set the resolution. That follows the
  documented rule of thumb but uses knowledge of the truth.
- The close-pair spacings (5–40 Hz) are below the main-lobe width of every window used here (86 Hz for a
  2048-sample Hann at 44.1 kHz). Close-pair recall runs from 11.6 % (`audiospylt-v1/tracked`) and
  11.8 % (`tracked-bh`) to 62.6 % (`sms-tools/matched`).
- With Hann, side lobes can pass the 20 dB prominence filter. When a partial lies within about 0.03 bin of
  the zero-pad ×4 grid, the spectrum samples exact nulls, which `spectrogram_peaks` clips at its
  `floor_db` of −120 dB, and the side lobes then stand far above those nulls. In a sweep of 200 sine
  frequencies across one bin, 20 % threw side-lobe peaks at −12 dBFS and 16 % at −34 dBFS; with
  Blackman-Harris, 0 %. So on the stationary family `tracked` has 60.7 % paired and 1 false / frame,
  against 100.0 % and 0 for `tracked-bh`; on `stat_inharm` alone, 31.2 % and 2. For stationary or
  synthetic tones use `window="blackmanharris"`. Its main lobe is twice as wide: 172 Hz against 86 Hz at
  2048 samples and 44.1 kHz.
- The tracker defaults were chosen from the development suite, but the published suite is not an untouched
  holdout. Provisional tracked rows had been scored on it before the choice, and the side-lobe finding
  came from its `stat_inharm` file.
- Against the previous published table of 2026-10-23: PM2 inharmonic was re-run. Over the 10 row/family
  pairs and 10 metrics the two tables share, the min–max ranges of the old and new five runs overlap in
  99 of 100 cells. The exception is `pm2-inharmonic/matched`, noise, |Δf| p95: 5.71–5.95 c then,
  5.52–5.67 c now. Loris: the raw tracks of the 13 earlier signals are byte-identical, but recall and
  false / frame moved in 10 per-signal and 6 per-family cells, because `hop_s` is one median over the
  row's files and the 8 new signals moved it (default 2.31 to 2.44 ms, matched 5.18 to 5.23 ms).

### Amplitude scales

All points are converted to sine peak amplitude re full scale before scoring. sms-tools reports
20·log10(A/2), so its values get +6.02 dB. A threshold `t` in sms dB is therefore 6.02 dB lower in dBFS: the
default −80 is −74 dBFS. Two offsets stay in the error because they are what the tools report:

- AudioSpylt 0.6 returns `rfft/N` magnitudes without the one-sided ×2 and without the window's coherent gain.
  A Hann-windowed sine of amplitude A reads A/4 (−12.04 dB). v1 corrects this.
- PM2 with the shipped templates reads stationary sines 0.33 dB high (median). With the matched window the
  median error is under 0.001 dB.

## Scoring (`evaluate.py`)

The first and last 0.1 s of each file are skipped. Each partial has a fixed identity, including through a
crossing. Its state is present, absent, or neutral within ±50 ms of an onset, offset or dropout edge. Gates
use 5 ms raised-cosine edges; file fades are separate. Tremolo troughs stay present. At each time, the tolerance
is min(50 c, half the gap to the nearest non-absent neighbour). Absent partials constrain nothing. Tolerances
never overlap, and shrink to zero at a crossing.

**Accuracy** compares every found point with the true partials at the point's own time; nothing is resampled.

- A point within tolerance of a neutral partial is excluded from the accuracy count, before the amplitude
  gate. An absent partial cannot attract a point.
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
- Pairing includes present and neutral partials. A neutral pair counts as neither a hit nor a false point.
  Absent partials cannot pair. Recall is the share of present (partial, grid time) cells that pair.
  False detections are unpaired points at or above −60 dBFS, given per grid frame.
- The hop is the nominal hop for frame rows and sms-tools. For Loris and PM2 it is the median time step within
  tracks (`meta.json` of each row).

Tracking uses present pairs only. A segment is a maximal run of consecutive present grid times for one
partial. Tracks / segment counts distinct paired track IDs, averaged over segments with a pair. Switch % is
the share of tracks with at least two paired grid times whose pairs cover more than one partial.

Onset and offset errors use the first or last raw point of the track paired nearest the corresponding end of
the adjacent present segment. That pair must be within 0.1 s of the segment end, or the transition is missed.
Raw endpoints include points outside the guard. Dropout edges do not count as onsets or offsets. A dropout is
determinable when both adjacent segment ends have such a pair, and bridged when their track IDs agree.
Family scores pool segments, tracks, transition errors and dropouts across files. Frame rows have no track
metrics. Metrics without applicable transitions are NaN, including their counts.

### Columns of `results/summary.md`

| column | meaning |
| --- | --- |
| points | found points inside the guard with freq > 0 and amp > 0, excluding neutral matches |
| paired %, dup %, far % | shares of those points (sum to 100) |
| \|Δf\| p50, p95 (c) | median and 95th percentile of the absolute frequency error of paired points |
| ΔA median (dB) | median signed amplitude error of paired points (bias) |
| \|ΔA\| p95 (dB) | 95th percentile of the absolute amplitude error |
| recall % | paired present cells / all present cells on the 10 ms grid |
| false / frame | unpaired covering points at or above −60 dBFS per grid frame |

The appended Tracking section has a table per family with tracked rows. Columns that are all “–” are omitted.

| tracking column | meaning |
| --- | --- |
| tracks / segment | distinct track IDs per paired present segment |
| switch % | share of tracks paired at two or more grid times that cover multiple partials |
| onset med (ms), onset \|err\| p95 (ms) | median signed onset error and 95th percentile absolute error |
| onset missed | onsets without a qualifying pair at the adjacent segment start |
| offset med (ms), offset \|err\| p95 (ms), offset missed | the corresponding offset statistics |
| bridged %, bridge undet. | share of determinable dropouts with the same track on both sides, and undeterminable count |

`per_signal.csv` and `per_family.csv` hold the accuracy and detection metrics per file and per family, plus the
signed median frequency error (`freq_bias_c`). They append `segments`, `tracks_per_segment`, `switch_tracks`,
`switch_frac`, `onset_n`, `onset_missed`, `onset_med_ms`, `onset_abs_p95_ms`, `offset_n`, `offset_missed`,
`offset_med_ms`, `offset_abs_p95_ms`, `bridge_n`, `bridge_undet`, `bridged_frac`, in that order. `segments` counts
segments with a pair; `switch_tracks` counts eligible tracks; `onset_n` and `offset_n` count transitions with a
found track; `bridge_n` counts determinable dropouts. `repeats.csv` keeps its existing columns and appends the
mean, min and max of the track metrics after `runs`. Repeat rows in the summary show means.

## Headline results

These tables are copied from [`results/summary.md`](results/summary.md) (run of 2026-11-04). Rows marked
"mean of 5" average five PM2 inharmonic runs. "–" means the row has no files in that family.

**|Δf| p95 (cents)** per family:

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-0.6/matched | 37.1 | 31.1 | 26.1 | 47.3 | 37.1 | 39.3 | 37.1 | 37.1 | 37.1 |
| audiospylt-v1/default | 2.44 | 1.43 | 2.81 | 38.5 | 8.53 | 11.8 | 2.48 | 2.48 | 2.19 |
| audiospylt-v1/matched | 0.0594 | 0.0373 | 3.28 | 38.1 | 7.16 | 9.04 | 0.0974 | 0.0963 | 0.0908 |
| audiospylt-v1/tracked | 0.0594 | 0.0373 | 3.28 | 9.74 | 3.89 | 7.57 | 0.0974 | 0.0963 | 0.0908 |
| audiospylt-v1/tracked-bh | 0.00584 | 0.0168 | 2.14 | 9.75 | 3.95 | 13.4 | 0.0177 | 0.102 | 0.00871 |
| loris/default | 0.0294 | 0.0324 | 0.111 | 38 | 26.3 | 43.8 | 0.0566 | 0.105 | 0.863 |
| loris/matched | 0.00231 | 0.326 | 7.25 | 33.8 | 44.4 | 2.32 | 0.0236 | 0.0424 | 4.34 |
| pm2-harmonic/matched | 0.131 | 0.0884 | 3.31 | – | 6.71 | – | 0.181 | 0.179 | 0.154 |
| pm2-harmonic/template | 43 | 21.4 | 16.3 | – | 34.5 | – | 47.2 | 46 | 38 |
| pm2-inharmonic/matched (mean of 5) | 0.132 | 0.0914 | 3.3 | 28.2 | 5.62 | 8.35 | 0.176 | 0.181 | 0.155 |
| pm2-inharmonic/template (mean of 5) | 34.9 | 20.2 | 14.9 | 36.8 | 34.3 | 33.8 | 46.5 | 45.9 | 38.2 |
| sms-tools/default | 0.921 | 0.43 | 0.815 | 38.2 | 13.8 | 29.7 | 1.57 | 1.28 | 1.31 |
| sms-tools/matched | 0.0594 | 0.0373 | 3.28 | 38.1 | 7.27 | 8.91 | 0.0974 | 0.0963 | 0.0908 |

**Recall %** per family:

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-0.6/matched | 94.1 | 97.9 | 100.0 | 42.4 | 100.0 | 89.6 | 86.7 | 100.0 | 100.0 |
| audiospylt-v1/default | 100.0 | 100.0 | 100.0 | 51.4 | 100.0 | 90.9 | 100.0 | 100.0 | 99.9 |
| audiospylt-v1/matched | 100.0 | 100.0 | 100.0 | 60.5 | 100.0 | 92.5 | 100.0 | 100.0 | 100.0 |
| audiospylt-v1/tracked | 100.0 | 100.0 | 100.0 | 11.6 | 77.0 | 84.8 | 100.0 | 100.0 | 100.0 |
| audiospylt-v1/tracked-bh | 100.0 | 100.0 | 100.0 | 11.8 | 59.1 | 77.8 | 100.0 | 100.0 | 100.0 |
| loris/default | 100.0 | 100.0 | 100.0 | 35.6 | 92.0 | 49.0 | 100.0 | 100.0 | 100.0 |
| loris/matched | 100.0 | 100.0 | 94.5 | 40.7 | 99.9 | 91.7 | 100.0 | 100.0 | 100.0 |
| pm2-harmonic/matched | 100.0 | 100.0 | 100.0 | – | 98.9 | – | 100.0 | 100.0 | 100.0 |
| pm2-harmonic/template | 99.4 | 99.3 | 100.0 | – | 98.0 | – | 98.8 | 98.7 | 100.0 |
| pm2-inharmonic/matched (mean of 5) | 82.4 | 93.3 | 91.6 | 30.5 | 74.4 | 78.6 | 79.8 | 92.6 | 97.5 |
| pm2-inharmonic/template (mean of 5) | 97.3 | 96.7 | 98.0 | 36.0 | 66.1 | 66.7 | 85.2 | 90.8 | 95.8 |
| sms-tools/default | 100.0 | 100.0 | 100.0 | 37.4 | 71.5 | 82.3 | 100.0 | 100.0 | 100.0 |
| sms-tools/matched | 100.0 | 100.0 | 100.0 | 62.6 | 98.0 | 92.4 | 100.0 | 100.0 | 100.0 |

**Tracks / segment**

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/tracked | 1 | 1 | 1 | 7 | 2.28 | 1.5 | 1 | 1 | 1 |
| audiospylt-v1/tracked-bh | 1 | 1 | 1 | 10 | 2.32 | 1.5 | 1 | 1 | 1 |
| loris/default | 1 | 1 | 1 | 8.83 | 17.6 | 1.33 | 1 | 1 | 1 |
| loris/matched | 1 | 1 | 31.1 | 15.9 | 2.33 | 1.5 | 1 | 1 | 1 |
| pm2-harmonic/matched | 1 | 1 | 1 | – | 1 | – | 1 | 1 | 1 |
| pm2-harmonic/template | 1 | 1 | 1 | – | 1.28 | – | 1 | 1 | 1 |
| pm2-inharmonic/matched (mean of 5) | 1 | 1 | 1.16 | 22.5 | 3.17 | 1.45 | 1 | 1.02 | 1 |
| pm2-inharmonic/template (mean of 5) | 1 | 3.41 | 3.53 | 14.2 | 12.6 | 14.5 | 1.1 | 1 | 1.52 |
| sms-tools/default | 1 | 1 | 1 | 20.5 | 2.9 | 3.25 | 1 | 1 | 1 |
| sms-tools/matched | 1 | 1 | 1 | 16.4 | 2.31 | 3.5 | 1 | 1 | 1 |

**Switch %**

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/tracked | 0.0 | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| audiospylt-v1/tracked-bh | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| loris/default | 0.0 | 0.0 | 0.0 | 96.3 | 3.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| loris/matched | 0.0 | 0.0 | 0.0 | 52.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pm2-harmonic/matched | 0.0 | 0.0 | 0.0 | – | 0.0 | – | 0.0 | 0.0 | 0.0 |
| pm2-harmonic/template | 0.0 | 0.0 | 0.0 | – | 0.0 | – | 0.0 | 0.0 | 0.0 |
| pm2-inharmonic/matched (mean of 5) | 0.0 | 0.0 | 0.0 | 7.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pm2-inharmonic/template (mean of 5) | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 3.6 | 0.0 | 0.0 | 0.0 |
| sms-tools/default | 0.0 | 0.0 | 0.0 | 48.5 | 1.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| sms-tools/matched | 0.0 | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Onset/offset and bridging columns are in `summary.md`. On the stationary files, the median amplitude error of
every row is within ±0.13 dB, except AudioSpylt 0.6 (−12.5 dB, its uncorrected scale) and the two PM2
templates (+0.33 dB). Read high recall together with `false / frame` in `summary.md`: the per-frame peak rows
reach high recall but also report many extra points.

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

For a reproducible development search:

```sh
$A benchmarks/accuracy/suite.py $OUT --dev
$A benchmarks/accuracy/tune_tracker.py $OUT DEV_RESULTS
$A benchmarks/accuracy/tune_tracker.py $OUT DEV_RESULTS --max-cents 400   # optional gate
```

The tuner reads only `OUT/dev/suite/`. It fixes Hann, frame 2048, hop 256, zero padding 4, centred frames,
`min_db=-80` and, by default, `max_cents=50` (override with `--max-cents`). It tests prominence None, 10, 15,
20, 25, 30 dB; minimum duration 0.02, 0.05, 0.1 s; and maximum gap 0, 1, 2, 4 frames (72 settings). It reuses
one spectrogram per file, one peak table per prominence and one linking pass per prominence and gap. Duration
filtering happens after linking. Family metrics pool samples; macro metrics average the families, skipping
NaNs. `tuning.csv` records every family and macro row. `tuning.md` lists the Pareto set for maximum recall,
minimum false / frame and minimum tracks / segment; when the gate is the default 50 c it also lists the
`track_partials` default if that setting lies outside the set. The author picked prominence 20 dB, minimum
duration 0.1 s, maximum gap 0 from the Pareto table. That setting is in the Pareto set: macro recall 84.8 %,
0.72 false / frame, 1.18 tracks / segment, switch 0 %.

Every runner writes `OUT/raw/<row>/<signal>.csv` (columns time, freq, amp, track; track −1 for untracked frame
peaks) and `OUT/raw/<row>/meta.json` (parameters, version, hop). `evaluate.py` scores whatever rows are present,
so the PM2 step can be skipped. On 2026-10-23, a run from fresh environments reproduced every non-PM2 value in
`per_family.csv` exactly. The Loris recall and false / frame shifts noted above come from the new signals'
effect on `hop_s`, not from changed tracks of the earlier files.

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
| `suite.py` | writes `OUT/suite/*.wav` and `truth.json`, or the dev suite with `--dev` |
| `tune_tracker.py` | development-only tracker search, writes `tuning.csv` and `tuning.md` |
| `points.py` | the shared raw format (numpy only) |
| `run_audiospylt.py`, `run_loris.py`, `run_sms.py`, `run_partiels.py` | one runner per tool |
| `evaluate.py` | scoring, writes `results/` |
| `results/summary.md` | generated tables per family and the repeat spread |
| `results/per_signal.csv`, `per_family.csv`, `repeats.csv` | all metrics |
