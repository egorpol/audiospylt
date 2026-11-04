# Partial-analysis accuracy benchmark

This benchmark measures how closely AudioSpylt and three other analysis tools recover the partials of synthetic
signals whose frequencies and amplitudes are known exactly at every instant. Each tool runs with its documented
defaults (or shipped template) and with a matched setting (Hann window of about 46 ms, hop 5.8 ms). AudioSpylt
0.6 runs with the matched setting only, as a reference for what v1 changed. The scores are frequency error in
cents, amplitude error in dB, recall and false detections.

The external tools are not part of AudioSpylt's dependencies or CI. Results from the run of 2026-11-04 are in
[`results/`](results/).

## Signals

`suite.py` writes 41 files: 44.1 kHz, float32 mono, 2 s, 10 ms raised-cosine fades, peak ≤ 0.9. Starting phases
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
| turn-back | `turn_320_720`, `turn_280_840`, `turn_450_800`, `turn_260_640`, `turn_550_880`, `turn_380_1140` | two exponential sweeps from ends a and b that meet at √(ab) at 1 s and return to their own ends |
| matched crossing | `cross_320_720`, `cross_280_840`, `cross_450_800`, `cross_260_640`, `cross_550_880`, `cross_380_1140` | the same ends, amps and starting phases as the turn-back of the same name, but the sweeps carry on to each other's end |
| moving dropout | `hole_vib_330`, `hole_vib_262`, `hole_vib_494`, `hole_vib_208`, `hole_gliss_165`, `hole_gliss_147`, `hole_gliss_660`, `hole_gliss_990` | 6 harmonics of a vibrato or an exponential glide, each partial with one dropout of 70 or 110 ms: 48 holes |

The last three families test [prediction](#meetings-and-holes). Their parameters were fixed before any score
and differ from the dev suite's, as a held-out check on settings picked on development signals. A matched crossing
takes its turn-back's seed, so the samples of a pair are identical up to the meeting.

The separate dev suite has perturbed signals from every family. It is for tuning the tracker. The published
suite never tunes anything. `suite.py OUT --dev` writes only `OUT/dev/suite/`: 31 files, the 15 perturbed signals
the tuner reads and 16 moving files (`dev_turn_*`, `dev_x_*`, `dev_hole_*`) it does not. `OUT/dev` can also be
scored as an ordinary `OUT` when raw rows there list dev names in `meta["signals"]`.

## Rows and parameters

A row is one tool with one setting. Output times are frame centres, or the times a tool reports for its track
points.

| row | tool | parameters |
| --- | --- | --- |
| `audiospylt-v1/default` | AudioSpylt 1.0.0.dev0 (this repository) | `spectrogram` defaults: frame 2048, hop 512, Hann, `zero_pad=1`, centred frames; `find_peaks(min_db=-60)` per frame (QIFFT on dB) |
| `audiospylt-v1/matched` | AudioSpylt 1.0.0.dev0 | frame 2048, hop 256, Hann, `zero_pad=4` (8192-point FFT); `find_peaks(min_db=-80)` |
| `audiospylt-v1/tracked` | AudioSpylt 1.0.0.dev0 | `track_partials(x, sr)` defaults (Hann, frame 2048, hop 256, `zero_pad=4`, `min_db=-80`, prominence 20 dB, 50 c, `max_gap=0`, `min_duration=0.1`), recorded from the signature in `meta.json` |
| `audiospylt-v1/tracked-bh` | AudioSpylt 1.0.0.dev0 | the same with `window="blackmanharris"` |
| `audiospylt-v1/tracked-lp` | AudioSpylt 1.0.0.dev0 | the `tracked` defaults with `prediction="lp"` |
| `audiospylt-v1/tracked-gap24`, `tracked-gap60` | AudioSpylt 1.0.0.dev0 | the `tracked` defaults with `max_gap=24` or `60` (139 or 348 ms) |
| `audiospylt-v1/tracked-lp-gap24`, `tracked-lp-gap60` | AudioSpylt 1.0.0.dev0 | the same with `prediction="lp"` |
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
- The rows with a gap are diagnostic settings, not recommendations. A long `max_gap` costs false tracks with
  either prediction: on the noise family, false / frame is 2.12 for `tracked`, 87.3 and 96.6 at gaps 24 and 60,
  and 1.56 for `tracked-lp`, 89.5 and 108 at those gaps.
- `sms-tools/matched` and `audiospylt-v1/matched` run the same analysis: `np.hanning(2049)` is a periodic Hann
  of 2048 samples followed by one zero, and both use an 8192-point FFT, hop 256, centred frames and parabolic
  interpolation on dB. Their error statistics are therefore identical. sms-tools reports fewer points because
  its tracking drops short and weak peaks.
- PM2 inharmonic output differs from run to run on the same file. Both PM2 inharmonic rows ran five times
  (`r1` … `r5`). `summary.md` shows the mean over the runs and the min–max spread. PM2 harmonic is
  deterministic and runs only on the 21 harmonic files (it needs an F0), so it has no close-pair, crossing,
  turn-back or matched-crossing results.
- `loris/default` uses the true lowest frequency of each file to set the resolution. That follows the
  documented rule of thumb but uses knowledge of the truth.
- The close-pair spacings (5–40 Hz) are below the main-lobe width of every window used here (86 Hz for a
  2048-sample Hann at 44.1 kHz). Close-pair recall runs from 11.5 % (`audiospylt-v1/tracked-lp`), 11.6 %
  (`tracked`) and 11.8 % (`tracked-bh`), which find no tracks on `pair_10`, `pair_20` and `pair_40`, to 62.6 %
  (`sms-tools/matched`) among rows without a gap and 68.2 % (`tracked-gap60`) with one.
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
  row's files and the 8 new signals moved it (default 2.31 to 2.44 ms, matched 5.18 to 5.23 ms). The 20 moving
  files moved no earlier value: a row's `hop_s` is now the median over the 21 earlier files alone.

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
determinable when both adjacent segment ends have such a pair, and bridged when their track IDs agree. These
pairs lie on present grid times, so the two ends are read outside the neutral zone.
Family scores pool segments, tracks, transition errors and dropouts across files. Frame rows have no track
metrics. Metrics without applicable transitions are NaN, including their counts.

**Meetings** are scored on the crossing, turn-back and matched crossing files, over a fixed zone of ±0.28 s around
the meeting at 1 s. `switch %` overcounts here, because a merged peak falls within tolerance of the other partial
for a few grid times.

- A track is eligible if its raw points start before the zone and end after it, and it has at least one present
  pair outside the zone.
- Its present pairs outside the zone vote for their partial, on each side. A side has a majority if it holds at
  least 3 votes and one partial has strictly more than any other.
- A track is right if both sides have a majority on the same partial, swap if on different partials, and unscorable
  otherwise, which includes a tie. Eligible = right + swap + unscorable.
- Counts pool over files, with no percentages.

**Holes at the hole edge** (`bridged_edge`) are an exploratory count on every file with dropouts, added after the
published run by re-scoring its raw output. Bridging above is the registered count, declared before that run.

- For each hole, take the raw track frame nearest each edge, within 50 c of the partial and 0.1 s of the edge. Neutral
  frames count, and no amplitude gate applies.
- A hole is determined if both sides have such a frame, and counted if the two frames carry the same track ID.
- The denominator is its own (`bridge_edge_n`), not `bridge_n`. It can differ a lot: on the moving dropouts,
  `pm2-inharmonic/template` has about 30 determined holes for `bridged` and 13 at the edge.

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

The appended Tracking section has a table per family with tracked rows. Columns that are all “–” are omitted. A
last section, Meetings and holes, gives meeting counts per row and family, then holes per row, pooled, by shape and
per file, then the same holes at the hole edge.

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
`offset_med_ms`, `offset_abs_p95_ms`, `bridge_n`, `bridge_undet`, `bridged_frac`, `bridged`, `meet_eligible`,
`meet_right`, `meet_swap`, `meet_unscorable`, `bridged_edge`, `bridge_edge_n`, `bridge_edge_undet`, in that order.
`segments` counts segments with a pair; `switch_tracks` counts eligible tracks; `onset_n` and `offset_n` count
transitions with a found track; `bridge_n` counts determinable dropouts and `bridged` those bridged. `repeats.csv`
keeps its existing columns and appends the mean, min and max of the track metrics after `runs`, then the same three
blocks for the `bridged` and `meet_*` counts, then for the edge counts. Repeat rows in the summary show means, except
the Meetings and holes section, which lists repeat runs one by one.

## Headline results

These tables are copied from [`results/summary.md`](results/summary.md) (run of 2026-11-04). Rows marked
"mean of 5" average five PM2 inharmonic runs. "–" means the row has no files in that family.

**|Δf| p95 (cents)** per family:

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo | turn-back | matched crossing | moving dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-0.6/matched | 37.1 | 31.1 | 26.1 | 47.3 | 37.1 | 39.3 | 37.1 | 37.1 | 37.1 | 38.8 | 38.6 | 32.1 |
| audiospylt-v1/default | 2.44 | 1.43 | 2.81 | 38.5 | 8.53 | 11.8 | 2.48 | 2.48 | 2.19 | 9.27 | 11.9 | 2.61 |
| audiospylt-v1/matched | 0.0594 | 0.0373 | 3.28 | 38.1 | 7.16 | 9.04 | 0.0974 | 0.0963 | 0.0908 | 7.35 | 7.61 | 2.97 |
| audiospylt-v1/tracked | 0.0594 | 0.0373 | 3.28 | 9.74 | 3.89 | 7.57 | 0.0974 | 0.0963 | 0.0908 | 1.04 | 1.18 | 2.97 |
| audiospylt-v1/tracked-bh | 0.00584 | 0.0168 | 2.14 | 9.75 | 3.95 | 13.4 | 0.0177 | 0.102 | 0.00871 | 1.83 | 4.11 | 2.02 |
| audiospylt-v1/tracked-gap24 | 0.0594 | 0.0373 | 3.28 | 38.1 | 5.74 | 7.57 | 0.0974 | 0.0963 | 0.0908 | 3.99 | 3.99 | 2.97 |
| audiospylt-v1/tracked-gap60 | 0.0594 | 0.0373 | 3.28 | 38.1 | 5.75 | 7.58 | 0.0974 | 0.0963 | 0.0908 | 3.99 | 3.99 | 2.97 |
| audiospylt-v1/tracked-lp | 0.0594 | 0.0373 | 3.28 | 9.74 | 3.89 | 7.57 | 0.0974 | 0.0963 | 0.0908 | 1.04 | 1.18 | 2.97 |
| audiospylt-v1/tracked-lp-gap24 | 0.0594 | 0.0373 | 3.28 | 38.1 | 5.75 | 7.57 | 0.0974 | 0.0963 | 0.0908 | 3.59 | 3.97 | 2.97 |
| audiospylt-v1/tracked-lp-gap60 | 0.0594 | 0.0373 | 3.28 | 38.1 | 5.75 | 7.58 | 0.0974 | 0.0963 | 0.0908 | 3.65 | 3.97 | 2.97 |
| loris/default | 0.0294 | 0.0324 | 0.111 | 38 | 26.3 | 43.8 | 0.0566 | 0.105 | 0.863 | 42.2 | 42.3 | 0.259 |
| loris/matched | 0.00231 | 0.326 | 7.25 | 33.8 | 44.4 | 2.32 | 0.0236 | 0.0424 | 4.34 | 0.799 | 0.652 | 6.89 |
| pm2-harmonic/matched | 0.131 | 0.0884 | 3.31 | – | 6.71 | – | 0.181 | 0.179 | 0.154 | – | – | 3 |
| pm2-harmonic/template | 43 | 21.4 | 16.3 | – | 34.5 | – | 47.2 | 46 | 38 | – | – | 24.9 |
| pm2-inharmonic/matched (mean of 5) | 0.132 | 0.0914 | 3.3 | 28.2 | 5.62 | 8.35 | 0.176 | 0.181 | 0.155 | 4.28 | 3.97 | 3.02 |
| pm2-inharmonic/template (mean of 5) | 34.9 | 20.2 | 14.9 | 36.8 | 34.3 | 33.8 | 46.5 | 45.9 | 38.2 | 35.8 | 36.3 | 24.2 |
| sms-tools/default | 0.921 | 0.43 | 0.815 | 38.2 | 13.8 | 29.7 | 1.57 | 1.28 | 1.31 | 20.6 | 21.6 | 1.19 |
| sms-tools/matched | 0.0594 | 0.0373 | 3.28 | 38.1 | 7.27 | 8.91 | 0.0974 | 0.0963 | 0.0908 | 7.35 | 7.61 | 2.97 |

**Recall %** per family:

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo | turn-back | matched crossing | moving dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-0.6/matched | 94.1 | 97.9 | 100.0 | 42.4 | 100.0 | 89.6 | 86.7 | 100.0 | 100.0 | 92.0 | 92.4 | 96.5 |
| audiospylt-v1/default | 100.0 | 100.0 | 100.0 | 51.4 | 100.0 | 90.9 | 100.0 | 100.0 | 99.9 | 93.6 | 94.0 | 100.0 |
| audiospylt-v1/matched | 100.0 | 100.0 | 100.0 | 60.5 | 100.0 | 92.5 | 100.0 | 100.0 | 100.0 | 94.6 | 95.1 | 100.0 |
| audiospylt-v1/tracked | 100.0 | 100.0 | 100.0 | 11.6 | 77.0 | 84.8 | 100.0 | 100.0 | 100.0 | 84.5 | 85.5 | 100.0 |
| audiospylt-v1/tracked-bh | 100.0 | 100.0 | 100.0 | 11.8 | 59.1 | 77.8 | 100.0 | 100.0 | 100.0 | 77.9 | 78.8 | 100.0 |
| audiospylt-v1/tracked-gap24 | 100.0 | 100.0 | 99.9 | 67.8 | 100.0 | 88.3 | 100.0 | 100.0 | 100.0 | 94.0 | 94.1 | 100.0 |
| audiospylt-v1/tracked-gap60 | 100.0 | 100.0 | 99.9 | 68.2 | 100.0 | 91.0 | 100.0 | 100.0 | 100.0 | 95.8 | 94.9 | 100.0 |
| audiospylt-v1/tracked-lp | 100.0 | 100.0 | 100.0 | 11.5 | 77.0 | 84.8 | 100.0 | 100.0 | 100.0 | 84.5 | 85.5 | 100.0 |
| audiospylt-v1/tracked-lp-gap24 | 100.0 | 100.0 | 100.0 | 67.3 | 100.0 | 87.8 | 100.0 | 100.0 | 100.0 | 95.8 | 94.6 | 100.0 |
| audiospylt-v1/tracked-lp-gap60 | 100.0 | 100.0 | 100.0 | 67.4 | 100.0 | 99.4 | 100.0 | 100.0 | 100.0 | 98.7 | 98.9 | 100.0 |
| loris/default | 100.0 | 100.0 | 100.0 | 35.6 | 92.0 | 49.0 | 100.0 | 100.0 | 100.0 | 48.0 | 48.6 | 100.0 |
| loris/matched | 100.0 | 100.0 | 94.5 | 40.7 | 99.9 | 91.7 | 100.0 | 100.0 | 100.0 | 93.6 | 93.8 | 99.4 |
| pm2-harmonic/matched | 100.0 | 100.0 | 100.0 | – | 98.9 | – | 100.0 | 100.0 | 100.0 | – | – | 100.0 |
| pm2-harmonic/template | 99.4 | 99.3 | 100.0 | – | 98.0 | – | 98.8 | 98.7 | 100.0 | – | – | 98.7 |
| pm2-inharmonic/matched (mean of 5) | 82.4 | 93.3 | 91.6 | 30.5 | 74.4 | 78.6 | 79.8 | 92.6 | 97.5 | 63.6 | 71.5 | 89.4 |
| pm2-inharmonic/template (mean of 5) | 97.3 | 96.7 | 98.0 | 36.0 | 66.1 | 66.7 | 85.2 | 90.8 | 95.8 | 80.0 | 78.8 | 89.7 |
| sms-tools/default | 100.0 | 100.0 | 100.0 | 37.4 | 71.5 | 82.3 | 100.0 | 100.0 | 100.0 | 86.0 | 86.2 | 93.7 |
| sms-tools/matched | 100.0 | 100.0 | 100.0 | 62.6 | 98.0 | 92.4 | 100.0 | 100.0 | 100.0 | 94.8 | 95.2 | 100.0 |

**Tracks / segment**

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo | turn-back | matched crossing | moving dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/tracked | 1 | 1 | 1 | 7 | 2.28 | 1.5 | 1 | 1 | 1 | 1.75 | 1.92 | 1 |
| audiospylt-v1/tracked-bh | 1 | 1 | 1 | 10 | 2.32 | 1.5 | 1 | 1 | 1 | 1.92 | 2.17 | 1 |
| audiospylt-v1/tracked-gap24 | 1 | 1 | 2.31 | 2.12 | 1.08 | 1.5 | 1 | 1 | 1 | 1.75 | 2.17 | 1.21 |
| audiospylt-v1/tracked-gap60 | 1 | 1.08 | 2.06 | 1.38 | 1.06 | 2 | 1 | 1 | 1 | 1.92 | 2.75 | 1.33 |
| audiospylt-v1/tracked-lp | 1 | 1 | 1 | 7 | 2.28 | 1.5 | 1 | 1 | 1 | 1.75 | 1.92 | 1 |
| audiospylt-v1/tracked-lp-gap24 | 1 | 1 | 1 | 2.38 | 1.42 | 1.75 | 1 | 1 | 1 | 1.92 | 1.5 | 1 |
| audiospylt-v1/tracked-lp-gap60 | 1 | 1 | 1.06 | 2.38 | 1.92 | 1 | 1 | 1 | 1 | 2.25 | 1.25 | 1.01 |
| loris/default | 1 | 1 | 1 | 8.83 | 17.6 | 1.33 | 1 | 1 | 1 | 15.2 | 15.3 | 1 |
| loris/matched | 1 | 1 | 31.1 | 15.9 | 2.33 | 1.5 | 1 | 1 | 1 | 3.33 | 3.08 | 2.79 |
| pm2-harmonic/matched | 1 | 1 | 1 | – | 1 | – | 1 | 1 | 1 | – | – | 1 |
| pm2-harmonic/template | 1 | 1 | 1 | – | 1.28 | – | 1 | 1 | 1 | – | – | 1 |
| pm2-inharmonic/matched (mean of 5) | 1 | 1 | 1.16 | 22.5 | 3.17 | 1.45 | 1 | 1.02 | 1 | 1.86 | 2.23 | 1 |
| pm2-inharmonic/template (mean of 5) | 1 | 3.41 | 3.53 | 14.2 | 12.6 | 14.5 | 1.1 | 1 | 1.52 | 15.7 | 14.7 | 2.25 |
| sms-tools/default | 1 | 1 | 1 | 20.5 | 2.9 | 3.25 | 1 | 1 | 1 | 7.25 | 6.92 | 1 |
| sms-tools/matched | 1 | 1 | 1 | 16.4 | 2.31 | 3.5 | 1 | 1 | 1 | 3.67 | 3.58 | 1 |

**Switch %**

| row | stationary | glissando | vibrato | close pair | noise | crossing | onset/offset | dropout | tremolo | turn-back | matched crossing | moving dropout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/tracked | 0.0 | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.5 | 0.0 |
| audiospylt-v1/tracked-bh | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.3 | 0.0 |
| audiospylt-v1/tracked-gap24 | 0.0 | 0.0 | 0.0 | 41.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 44.4 | 0.0 |
| audiospylt-v1/tracked-gap60 | 0.0 | 0.0 | 0.0 | 37.5 | 0.0 | 14.3 | 0.0 | 0.0 | 0.0 | 4.5 | 50.0 | 3.2 |
| audiospylt-v1/tracked-lp | 0.0 | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.5 | 0.0 |
| audiospylt-v1/tracked-lp-gap24 | 0.0 | 0.0 | 0.0 | 28.6 | 8.9 | 0.0 | 0.0 | 0.0 | 0.0 | 53.3 | 12.5 | 0.0 |
| audiospylt-v1/tracked-lp-gap60 | 0.0 | 0.0 | 0.0 | 28.6 | 17.0 | 0.0 | 0.0 | 0.0 | 0.0 | 80.0 | 15.4 | 0.0 |
| loris/default | 0.0 | 0.0 | 0.0 | 96.3 | 3.8 | 0.0 | 0.0 | 0.0 | 0.0 | 20.5 | 25.7 | 0.0 |
| loris/matched | 0.0 | 0.0 | 0.0 | 52.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 12.9 | 7.4 | 0.0 |
| pm2-harmonic/matched | 0.0 | 0.0 | 0.0 | – | 0.0 | – | 0.0 | 0.0 | 0.0 | – | – | 0.0 |
| pm2-harmonic/template | 0.0 | 0.0 | 0.0 | – | 0.0 | – | 0.0 | 0.0 | 0.0 | – | – | 0.0 |
| pm2-inharmonic/matched (mean of 5) | 0.0 | 0.0 | 0.0 | 7.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.5 | 0.0 |
| pm2-inharmonic/template (mean of 5) | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 3.6 | 0.0 | 0.0 | 0.0 | 2.3 | 5.7 | 0.0 |
| sms-tools/default | 0.0 | 0.0 | 0.0 | 48.5 | 1.4 | 0.0 | 0.0 | 0.0 | 0.0 | 22.6 | 23.3 | 0.0 |
| sms-tools/matched | 0.0 | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 37.0 | 32.0 | 0.0 |

Onset/offset and bridging columns are in `summary.md`. On the stationary files, the median amplitude error of
every row is within ±0.13 dB, except AudioSpylt 0.6 (−12.5 dB, its uncorrected scale) and the two PM2
templates (+0.33 dB). Read high recall together with `false / frame` in `summary.md`: the per-frame peak rows
reach high recall but also report many extra points.

## Meetings and holes

`link_peaks` and `track_partials` take `prediction`. The default `"hold"` expects a track at its last measured
frequency, and it stays the default, so no earlier result moved. `"lp"` continues a track's motion over its last 64
frames of log frequency: a fitted line plus an order 6 Burg prediction of what the line leaves, fed back through
missed frames. This is not the form of Lagrange, Marchand, Raspaud and Rault (DAFx 2003), which runs Burg on the raw
frequencies: on development signals that form bridged fewer holes in moving partials than `"hold"`.

Prediction acts across a hole only with `max_gap` > 0, so these tables go up to 60 frames. Those rows are diagnostic
settings with a [noise cost](#rows-and-parameters), not recommendations. Counts are copied from
[`results/summary.md`](results/summary.md), which also has every other row and the holes per file.

**Meetings**: right / swap of eligible tracks (the rest are unscorable), 12 partials in 6 files per family.

| row | prediction | `max_gap` | turn-back | matched crossing |
| --- | --- | ---: | ---: | ---: |
| audiospylt-v1/tracked | hold | 0 | 3 / 0 of 3 | 3 / 0 of 3 |
| audiospylt-v1/tracked-lp | lp | 0 | 3 / 0 of 3 | 3 / 0 of 3 |
| audiospylt-v1/tracked-gap24 | hold | 24 | 6 / 0 of 6 | 3 / 5 of 8 |
| audiospylt-v1/tracked-lp-gap24 | lp | 24 | 2 / 8 of 10 | 9 / 0 of 9 |
| audiospylt-v1/tracked-gap60 | hold | 60 | 7 / 0 of 9 | 0 / 7 of 8 |
| audiospylt-v1/tracked-lp-gap60 | lp | 60 | 0 / 12 of 12 | 12 / 0 of 12 |

The two predictions fail in opposite ways. At gap 60, `"hold"` gets 7 of 9 turn-back tracks right with 0 swaps but
swaps 7 of 8 crossing tracks, none right. `"lp"` gets 12 of 12 crossing tracks right but swaps all 12 turn-backs. In
the merged hole the tracker has one peak for two partials, so a prediction states an assumption, not a measurement.

Without a gap, 3 tracks reach across the zone and all 3 stay right. Loris and sms-tools match that in both
families, except `sms-tools/matched` on crossings (3 right, 1 swap of 4). `pm2-inharmonic/matched` has 3 right of 3
on crossings and 1 to 3 right, with no swap, on turn-backs; `pm2-inharmonic/template` has none eligible.

**Holes in moving partials**: bridged of determined holes. The first three count columns are the registered count,
`bridged`; the last three are the exploratory hole-edge count, `bridged_edge`.

| row | prediction | `max_gap` | all | vibrato | glide | edge: all | edge: vibrato | edge: glide |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/tracked | hold | 0 | 0/48 | 0/24 | 0/24 | 0/48 | 0/24 | 0/24 |
| audiospylt-v1/tracked-lp | lp | 0 | 0/48 | 0/24 | 0/24 | 0/48 | 0/24 | 0/24 |
| audiospylt-v1/tracked-gap24 | hold | 24 | 14/48 | 14/24 | 0/24 | 10/48 | 10/24 | 0/24 |
| audiospylt-v1/tracked-lp-gap24 | lp | 24 | 40/48 | 16/24 | 24/24 | 40/48 | 16/24 | 24/24 |
| audiospylt-v1/tracked-gap60 | hold | 60 | 14/48 | 14/24 | 0/24 | 9/48 | 9/24 | 0/24 |
| audiospylt-v1/tracked-lp-gap60 | lp | 60 | 41/48 | 17/24 | 24/24 | 41/48 | 17/24 | 24/24 |
| pm2-harmonic/matched | – | – | 32/48 | 18/24 | 14/24 | 32/48 | 18/24 | 14/24 |

On the registered count, `"lp"` bridges 40 of 48 holes against 14 for `"hold"` at gap 24, and 41 against 14 at gap 60.
The lead comes mostly from glides, 24 of 24 against 0 of 24; on vibrato it is 16 and 17 of 24 against 14. Loris,
sms-tools, PM2 inharmonic and `pm2-harmonic/template` bridge none.

The hole-edge count exists because `bridged` looks past the neutral zone. A held track can lose a vibrato partial at
the hole, leave it to a short new track, and take it back when the vibrato returns near the held frequency; that
scores as bridged. At the hole edge, `"hold"` falls to 10 of 48 at gap 24 and 9 of 48 at gap 60, on vibrato only, and
`"lp"` stays at 40 and 41. This count is exploratory: it was added after the published run, it is not the registered
score, and its denominator is its own. The same ID at the edge does not prove one track through the gap either,
because a track can still leave and come back.

These counts are small samples:

- A pair shares one phase draw, so each meeting family has 6 draws. That buys control, not precision: over 8 seeds
  on the dev pairs, the SD of the turn-back minus crossing total was about 1.1 tracks with shared phases and 1.2 with
  separate ones, on differences of 4 to 9 tracks.
- The fresh pairs are easier than the dev pairs. All are at least 91.6 Hz apart at the zone edge, while 3 of the 6 dev
  pairs are below 86 Hz, the main-lobe width.
- The 48 holes are not 48 phase repeats: they are 6 holes in each of 8 files, one seed per file. On the registered
  count of vibrato holes, both predictions bridge 6 of 6 on `hole_vib_262` and `hole_vib_208` and at most 3 of 6 on
  `hole_vib_330` and `hole_vib_494`. Depth, rate, hole width and partial order vary together across those files, so
  this is an association, not a cause.
- The development means, over 8 seeds, were recorded before the published run. Gap 60, right / swap of 12: turn-back hold
  9.5 / 0, lp 0.4 / 9.6; crossing hold 0.9 / 6.9, lp 8.8 / 0.2. Gap 24, holes of 24: hold 15.8, lp 21.0, of which
  glides of 12 hold 6.0, lp 12.0 and vibrato of 12 hold 9.8, lp 9.0. The meetings kept their pattern. `"hold"` did worse
  on the published holes, with no glide bridged; all four published glides span more than the dev glide's 1 octave
  (1.17 to 1.58 octaves), and no cause was tested.

No pass mark was set: the option is opt-in, and a poor result would have changed this text, not the code.

## Reproducing

The commands are for bash on Linux and run from the repository root. Each tool needs its own environment:
sms-tools 1.2 requires `numpy<2`, and loristrck is not an AudioSpylt dependency.

```sh
OUT=$HOME/asp-bench        # suite, raw rows, Partiels scratch; ~200 MB
VENVS=$HOME/asp-bench-venvs

uv venv --python 3.12 $VENVS/asp
uv pip install --python $VENVS/asp/bin/python '.[plot]'   # the 0.6 shim imports plotly
uv venv --python 3.12 $VENVS/loris
uv pip install --python $VENVS/loris/bin/python loristrck==1.7.2 soundfile
uv venv --python 3.12 $VENVS/sms
uv pip install --python $VENVS/sms/bin/python sms-tools==1.2 'numpy<2' soundfile

A=$VENVS/asp/bin/python
$A benchmarks/accuracy/suite.py $OUT                              # < 1 s
$A benchmarks/accuracy/run_audiospylt.py $OUT                     # ~70 s
$VENVS/loris/bin/python benchmarks/accuracy/run_loris.py $OUT     # ~9 s
$VENVS/sms/bin/python benchmarks/accuracy/run_sms.py $OUT         # ~20 s
$A benchmarks/accuracy/run_partiels.py $OUT TEMPLATE_DIR [PARTIELS]   # ~90 s, optional
$A benchmarks/accuracy/evaluate.py $OUT benchmarks/accuracy/results   # ~30 s with the PM2 rows
```

`run_loris.py`, `run_sms.py` and `run_partiels.py` take `--new`, which analyses only the 20 moving files and extends
the rows of an earlier full run. The published PM2 inharmonic rows were extended this way, because their earlier runs
cannot be reproduced. A full run and a `--new` run agree on `hop_s`, the median over the 21 earlier files alone.

For a reproducible development search:

```sh
$A benchmarks/accuracy/suite.py $OUT --dev
$A benchmarks/accuracy/tune_tracker.py $OUT DEV_RESULTS
$A benchmarks/accuracy/tune_tracker.py $OUT DEV_RESULTS --max-cents 400   # optional gate
```

To score the tracked rows on all 31 dev files:

```sh
$A benchmarks/accuracy/run_audiospylt.py $OUT --dev   # writes OUT/dev/raw
$A benchmarks/accuracy/evaluate.py $OUT/dev $OUT/dev/results
```

The tuner reads only the 15 perturbed signals in `OUT/dev/suite/`. It fixes Hann, frame 2048, hop 256, zero padding 4, centred frames,
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
effect on `hop_s`, not from changed tracks of the earlier files. On 2026-11-04, a second run of the suite and the
runners gave byte-identical raw CSVs for all 14 non-PM2 rows and both PM2 harmonic rows.

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
| `run_audiospylt.py`, `run_loris.py`, `run_sms.py`, `run_partiels.py` | one runner per tool; `run_audiospylt.py --dev` runs the dev files, the other three take `--new` |
| `evaluate.py` | scoring, writes `results/` |
| `results/summary.md` | generated tables per family, the repeat spread, meetings and holes |
| `results/per_signal.csv`, `per_family.csv`, `repeats.csv` | all metrics |
