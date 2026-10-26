# Resynthesis fidelity benchmark

This benchmark measures how close a resynthesis comes to the original sound. Each row analyses a sound and
renders it back to audio. The rows are AudioSpylt v1's per-frame peak resynthesis, AudioSpylt 0.6's tutorial
render path, Loris and sms-tools with their own synthesizers, and the tracks of Loris, sms-tools and PM2 (via
Partiels) played through `audiospylt.synthesize`. The scores are a log-spectral distance per STFT bin, a
log-spectral distance per frequency band, and the level of the render. The evaluator also writes level-matched
WAVs for listening.

There are three sets of sounds: the three case-study excerpts, the 27 sounds of the ICMC 2000
analysis/synthesis panel, and five test sounds from the Loris and sms-tools repositories. For the panel sounds,
the resyntheses that the authors of IRCAM's tools, Loris and SMS made in 2000 are scored as historical rows.

The external tools and the reference sounds are not part of AudioSpylt's dependencies or CI. Results from the
run of 2026-10-26 are in [`results/`](results/).

## Sounds

### Case-study excerpts (`case`)

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

### Reference sounds (`icmc2000`, `tools`)

- **`icmc2000`**: the 27 sounds of the analysis/synthesis panel of ICMC 2000 (Wright, Beauchamp, Fitz, Rodet,
  Röbel, Serra and Wakefield, "Analysis/synthesis comparison", *Organised Sound* 5(3): 173–189, 2000). Six
  systems were each run by their authors on the same sounds; the panel's site kept the inputs, the analyses in
  SDIF and, for IRCAM's tools, Loris and SMS, the authors' resyntheses. The sounds are mono 16-bit AIFF, 82.0 s
  in total: 21 at 44.1 kHz, 5 at 22.05 kHz (`deplus`, `harris`, `leonard`, `piano`, `toby_yodel`) and
  `shaku` at 48 kHz. By the paper's description: trumpet and piano notes; saxophone, shakuhachi, suling,
  reverberant clarinet, flutter-tongued trombone, piano and guitar lines; xylophone, gamelan gong, bass drum,
  berimbau, bongo; tampura, harp glissando; five sung sounds; two speech sounds (one a giggle); an apple bite;
  an angry cat; ten crossing chirps.
- **`tools`**: the test sounds of the tools' own repositories. Loris 1.8 (`test/`, GPL): `loris-clarinet`
  (G♯4, 3.01 s) and `loris-flute` (D4, 2.62 s), used by its `make check`. sms-tools-materials (`sounds/`,
  CC BY 4.0): `sms-bendir` (3.15 s), `sms-vignesh` (3.09 s) and `sms-sax-phrase-short` (3.15 s), the inputs of
  the sine, harmonic and harmonic-plus-residual model examples. All five are mono 16-bit at 44.1 kHz.

PM2 has no public repository with test sounds. Every reference sound gets both PM2 templates, one run each.

[`fetch_reference.py`](fetch_reference.py) downloads these files (122 downloads, 149 files once the two
tarballs are unpacked, 32 MB) to `OUT/reference/` and checks each against the SHA-256 in
[`reference.json`](reference.json), pinned on 2026-10-26. The panel's site is gone; its files come from the
Wayback Machine's copy of `http://www.cnmat.berkeley.edu/SDIF/ICMC2000/`. They carry no licence statement, so
they are fetched, not committed. The Loris renders keep their archived file names (`webern.loris.aiff`,
`angry_cat.d.loris.aiff`); `reference.json` maps each file to its input. The meanings of the `.d` and `.rs`
parts of these names are not stated in the paper.

## Rows and parameters

"Matched" means a Hann window of 46.44 ms and a hop of 5.805 ms for every tool, as in the
[accuracy benchmark](../accuracy/README.md): 2048 and 256 samples at 44.1 kHz, 1024 and 128 at 22.05 kHz, 2229
and 279 at 48 kHz. Each sound is analysed and rendered at its own rate.

| row | analysis | synthesis |
| --- | --- | --- |
| `audiospylt-v1/default` | AudioSpylt 1.0.0.dev0: `spectrogram` defaults (frame 2048, hop 512 samples at every rate, Hann, `zero_pad=1`, centred frames), `find_peaks(min_db=-60)` per frame | `synthesize` defaults: continuous phase, rows linked within `link_cents=100`, `link_ramp=0.005`, `fade=0` |
| `audiospylt-v1/matched` | matched frame and hop, Hann, `zero_pad=4`; `find_peaks(min_db=-80)` | as above |
| `audiospylt-v1/matched-fade` | as `matched` | as above with `fade=0.0025` (the setting of the G1 listening check) |
| `audiospylt-0.6/matched` | 0.6 API (`dft_analysis.analyze_signal`, kept as a deprecated shim) per frame: matched frame and hop, Hann, frame k centred at k·hop, `thresh_amp_low=2.5e-5` (−80 dBFS on 0.6's scale), `thresh_amp_high=1` | the path of `tutorials_tech/resynth.ipynb`: `plot_waves(k=0.00015, phase_mode="chirp", edge_fade_s=0.0025)`, then `render_audio(fs_target_name="44.1kHz", alias_above_nyquist=True)`; see [How the 0.6 row runs](#how-the-06-row-runs-and-what-v1-changes) |
| `loris/matched` | loristrck 1.7.2: `resolution` 43.07 Hz, `windowsize` 86.13 Hz, `hoptime` 5.805 ms, `ampfloor` −80 | `loristrck.synthesize(start=0, end=duration)` |
| `loris/matched+asp` | the same tracks | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `loris/home` | `tools` Loris sounds only: the settings of Loris 1.8's `test/morphtest.py` (see [Home settings](#home-settings)) | `loristrck.synthesize(start=0, end=duration)` |
| `loris/home+asp` | the same tracks | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `sms-tools/matched` | sms-tools 1.2 `sineModelAnal`: `np.hanning(2049)`, N 8192, H 256, t −86.02 (= −80 dBFS), `maxnSines` 100, `minSineDur` 0.01 s (at 44.1 kHz; window and hop scale with the rate, N is the next power of 2 at or above 4 frames) | `sineModelSynth`, Ns = 4 H, H as in the analysis |
| `sms-tools/matched+asp` | the same tracks | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `sms-tools/home` | `tools` sms-tools sounds: the settings of sms-tools' own examples; `icmc2000`: the 21 SMS 2000 analysis files mapped onto sms-tools (see [Home settings](#home-settings)) | `sineModelSynth` |
| `sms-tools/home+asp` | the same tracks | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `pm2-harmonic/matched+asp` | Partiels 2.6.0 with Ircam Vamp Plugins 2.1.0 (PM2 v5), `PM2-Partials-Tracking-Harmonic-20.ptldoc` with window 46.44 ms, overlap 8x (hop 5.805 ms), FFT oversampling 4x, Hann; read with `read_partiels` | `synthesize(tracks_to_events(tracks), fade=0.005)` |
| `pm2-inharmonic/matched+asp` | same with `PM2-Partials-Tracking-Inharmonic-20.ptldoc`; five runs on `case`, one on the reference sets | as above |
| `ircam-2000/author`, `loris-2000/author`, `loris-2000/author-rs`, `sms-2000/author` | `icmc2000` only: the authors' own analyses of 2000 (settings published only for SMS) | the authors' renders of 2000, resampled to the input's rate and shifted; see [Historical rows](#historical-rows) |

Notes on the rows:

- AudioSpylt v1 has no partial tracker. Every peak above `min_db` in every frame becomes an event held over the
  frame centre ± hop/2, and `synthesize` joins events of consecutive frames into one phase-continuous voice
  when their frequencies are within 100 cents. The matched rows keep 283 events per frame on `bayle`, 14 on
  `de_natura_son` and 59 on the flute. For comparison, Loris keeps 245, 11 and 50 track points per frame,
  sms-tools 88, 14 and 55, PM2 inharmonic 10–12 on `bayle` (the mean of each of the five runs; at most 15 in
  a frame) and PM2 harmonic 18 on the flute. A v1 row is a sum of all spectral peaks of every frame, closer to
  an STFT resynthesis than to a compact model of the sound.
- With `fade=0`, an event that does not link starts and stops at full amplitude. On `de_natura_son`, the peaks
  of the 20–80 Hz rumble (bins 1–4 of a 2048-sample frame, 21.5 Hz each) move by more than 100 cents from
  frame to frame, so few of them link, and each one switches on and off every 5.8 ms. The result is broadband
  clicks: in `audiospylt-v1/matched`, every band above 354 Hz has a band LSD between 18.9 and 42.1 dB.
  `fade=0.0025` removes the clicks (`matched-fade`), but it also fades every unlinked event inside its own
  5.8 ms span, which raises the flute's band LSD from 6.46 to 11.71 dB.
- `audiospylt-v1/default` has hop 512, no zero padding and a −60 dB threshold, so it keeps 37, 4 and 9 events
  per frame on the case-study excerpts.
- `loristrck.synthesize` starts its output at the first breakpoint unless `start` is given (83 ms into
  `bayle`, 16 ms into the flute). The runner passes `start=0` and the sound's duration so that the render
  lines up with the input. Every other renderer of today's tools starts at time 0.
- The 0.6 row synthesises on a grid of k = 0.15 ms (6.67 kHz), so partials above 3.33 kHz fold down.
  `render_audio` interpolates linearly up to 44.1 kHz and normalises the peak to full scale. That is why its
  gain is +9.5 to +21 dB on the case-study excerpts, which peak between −24.2 and −10.7 dBFS. See
  [Level](#level) for the scores at matched level.
- Renders by `synthesize` and by the 0.6 path end at the last event, up to 0.96 s before the end of the input
  (`audiospylt-v1/default` on the ICMC `piano`); the evaluator pads them with zeros.
- The `+asp` rows replay frequency and amplitude only. Loris's own synthesizer also uses the analysed phase and
  bandwidth, and sms-tools' uses the analysed phase. Comparing a tool's own row with its `+asp` row therefore
  isolates the synthesizer, not the analysis.
- `sms-tools/matched` keeps at most 100 sines per frame. On `bayle` that leaves the bands above 9 kHz 8–16 dB
  low over the excerpt, and the top band (18–22 kHz) empty in 42 % of the frames.
- PM2 inharmonic output differs from run to run on the same file (as in the accuracy benchmark). On `case` the
  row ran five times (`r1` … `r5`); `summary.md` shows the mean and the min–max spread. On `bayle`, with 10–12
  partials per frame, many frames have no partial in a given band: in run 1, the render is at least 40 dB
  below the excerpt in 1782–2245 Hz in 51 % of the frames and in each band above 5.7 kHz in 45–100 %. On
  `de_natura_son`, it misses most of the sub-bass that carries the excerpt's energy: in run 1 the 0–177 Hz band
  is 22.7 dB low (band LSD 32.3 dB), and the gain is −21.7 dB.
- PM2 harmonic finds 19 tracks on the ICMC `shaku` (PM2 inharmonic 1,564). This comes from the template's f0
  settings, which were not tuned per sound.
- Partiels has no synthesizer, so the PM2 rows exist only as `+asp` rows.

### Home settings

The home rows run today's tools with the settings their authors used on these sounds.

- **`loris/home`**, from Loris 1.8 `test/morphtest.py`: clarinet `Analyzer(390)`, `setFreqDrift(30)`,
  `setAmpFloor(-80)`; flute `Analyzer(270)`, `setFreqDrift(30)`. `Analyzer(r)` sets resolution r, window
  width 2r, hop 1/(2r) and sidelobe level 90 dB, and keeps the amplitude floor at −90 dB unless it is set. The
  runner passes all of these to loristrck 1.7.2. The test script does not channelize or distill, so these
  analyses have many more partials than the matched ones (clarinet 22,432 against 3,832).
- **`sms-tools/home`** on the `tools` sounds: the settings of sms-tools' model examples
  (`interface-models/*_function.py`). `sms-bendir`: sine model, Hamming 2001, N 2048, H 128, t −80,
  `minSineDur` 0.02, `maxnSines` 150, `freqDevOffset` 10, `freqDevSlope` 0.001. `sms-vignesh`: harmonic model,
  Blackman 1201, N 2048, H 128, t −90, `minSineDur` 0.1, `nH` 100, f0 130–300 Hz, `f0et` 7.
  `sms-sax-phrase-short`: the harmonic part of the harmonic-plus-residual example, Blackman 601, N 1024, H 128,
  t −100, `minSineDur` 0.1, `nH` 100, f0 350–700 Hz, `f0et` 5. (`hprModelSynth` adds the subtracted residual
  back, which would return the input itself.)
- **`sms-tools/home`** on `icmc2000`: the SMS 2000 analysis files (`.anl`) of the 21 sounds SMS rendered,
  mapped onto sms-tools 1.2. SMS 2000 is a different program, so this row is an approximation of the 2000
  analysis, not a rerun of it. `SineModel 1` becomes `harmonicModelAnal` with `nH` = `nSines` and the f0 range
  `LowestPitch`–`HighestPitch`; otherwise `sineModelAnal` with `maxnSines` = `nSines`. t = `MagThreshold`.
  `WindowSizeInh` (14 files, in samples) or `WindowSizeHarm` (4 files, in periods of `DefaultPitch`) sets the
  window, `FrameRate` (2 files) the hop (sr / `FrameRate`). Points outside `LowestFreq` (6 files) and
  `HighestFreq` (21 files) are dropped after the analysis. Where a file is silent, the matched values hold:
  `np.hanning`, hop 5.805 ms, N the next power of 2 at or above 4 (M − 1), `minSineDur` 0.01 s; the harmonic
  model uses `f0et` 5 and `harmDevSlope` 0.01. Window types 7 and 8 could not be decoded. The residual,
  attack and tracking-mode keys are not mapped; `OUT/raw/sms-tools/home/meta.json` lists them per sound.

### Historical rows

[`run_historical.py`](run_historical.py) turns the archived renders into rows:

- `ircam-2000/author`: all 27 sounds. 17 `*.synt.aiff`, 8 `*.hmm.synt.aiff`, `applebite2.noise+trans.aiff`
  and `xylo.m6.synth.aiff`; for `bongo_roll`, the plain `.synt.aiff` of the two in the archive.
- `loris-2000/author` (`*.loris.aiff`) and `loris-2000/author-rs` (`*.rs.loris.aiff`): 24 sounds each (no
  `screw`, `suling-phrase` or `trumpet`).
- `sms-2000/author` (`*.sms.wav`): 21 sounds (no `conant-trombone-flutter`, `flute+voice`, `giggle`,
  `shafqat-derbari`, `tatum` or `toby_yodel`).

The four rows have 18 sounds in common; the summary gives each row's mean and median over those 18.

Fifteen renders are not at their input's rate: Loris rendered the five 22.05 kHz inputs and the 48 kHz
`shaku` at 44.1 kHz (both variants), and SMS rendered `deplus` (22.05 kHz) and `shaku` at 44.1 kHz and
`screw` (44.1 kHz) at 22.05 kHz, so the `screw` render has nothing above 11 kHz. Each is resampled to the
input's rate with `scipy.signal.resample_poly`.

Each render is then shifted by the delay within ±0.25 s that gives the lowest bin LSD (a 2 ms grid, then
0.1 ms steps). The median shift is 0.1–0.5 ms per row. Only two are longer than 5 ms: Loris `webern_choral`
(+227 ms, both variants; the render covers 0.23–1.96 s of the 5.33 s input) and IRCAM `xylo` (−46 ms).
Because the shift minimises the score that is reported, it favours these rows. Over all their renders, the
mean bin LSD is 0.00–0.04 dB lower than without the shift. A first version aligned the 5 ms RMS envelopes
instead. On sustained sounds that chose shifts that raised the bin LSD: SMS `shaku` from 8.47 to 9.39 dB
(+14 ms), SMS `trumpet` from 12.70 to 13.33 dB (−43 ms), Loris `piano` from 3.30 to 3.65 dB (−14 ms).

The renders are scored as archived, like every other row:

- Four renders cover less than 90 % of their input: IRCAM `sop6a6` (1.38 of 1.87 s), IRCAM `webern_choral`
  (3.70 of 5.33 s) and Loris `webern_choral` (1.96 of 5.33 s, both variants). The rest is scored as silence.
- Two SMS renders are not level-scaled copies of their input. `trumpet` is 29.7 dB below its input (RMS −48.7
  against −19.1 dBFS) and carries a DC offset of 0.0028. `shaku` is 21.7 dB below its input. Scaled to the
  input's RMS, both score worse (bin LSD 25.23 and 18.24 dB), so they are left as they are. `shaku` is one of
  the 18 common sounds.

`OUT/raw/<row>/meta.json` gives each render's archived file name, rate, shift, length, and its bin LSD before
the shift.

## How the 0.6 row runs, and what v1 changes

`run_audiospylt.py::old_row` calls the 0.6 functions, which v1 keeps as deprecated shims. Against v0.6.3,
`dft_analysis.py`, `plot_wave.py` and `generate_wave_file.py` differ only by the deprecation footer, one
unused import and two assignments that are never read (`git diff --ignore-cr-at-eol --ignore-all-space
v0.6.3 HEAD`), so the row computes what v0.6.3 computes. At 44.1 kHz:

1. The mono excerpt is zero-padded by 1024 samples at each end; frame k is 2048 samples starting at k·256, so
   it is centred at k·256 / 44100 s, the grid of the v1 matched row.
2. `analyze_signal(frame, sr, window_type="hann", thresh_amp_low=2.5e-5, thresh_amp_high=1)`. Its
   `compute_fft` is `|rfft(x·w)| / N`: no zero padding, no one-sided ×2 and no window-gain correction, so a
   sine of amplitude a reads a/4 with Hann (−12 dB). `filter_peaks` is
   `scipy.signal.find_peaks(spec, height=(2.5e-5, 1))`, so peaks sit on bin centres (21.5 Hz apart) without
   interpolation. 2.5e-5 = 10⁻⁴/4, i.e. −80 dBFS for a true sine.
3. Each frame's peaks become rows held over [t − hop/2, t + hop/2], as in the v1 rows, with
   `freq_start = freq_stop` and `amp_min = amp_max`.
4. `plot_waves(table, k=0.00015, phase_mode="chirp", edge_fade_s=0.0025)` synthesises on the grid
   `t = arange(0, n + k, k)`, i.e. at 6.67 kHz (the tutorial's setting). `interval_mode="closed"` lets
   abutting rows share a grid sample. `chirp` starts every row at phase 0 (`sin(2π f τ)`, τ from the row
   start), so the phase resets every 5.8 ms, and every row fades in and out over 2.5 ms (raised cosine) inside
   its 5.8 ms span. Partials above 3.33 kHz fold down on that grid.
5. `render_audio(y, 1/k, fs_target_name="44.1kHz", alias_above_nyquist=True)` interpolates linearly
   (`np.interp`) from 6.67 to 44.1 kHz, normalises the peak to 0 dBFS and writes 24-bit PCM; the runner
   divides the integers by 2³¹. At other rates the row uses the matched frame and hop of that rate and asks
   `render_audio` for the input's rate.

The v1 matched row instead uses `spectrogram(2048, 256, hann, zero_pad=4)` scaled to peak amplitude (a sine of
amplitude a reads a) and `find_peaks(min_db=-80)` with quadratic interpolation on the dB magnitudes. It
synthesises at the output rate directly, with half-open rows and continuous phase (rows within 100 cents link
into one voice that keeps its phase), a 5 ms raised-cosine ramp on amplitude steps at links, no fades
(`fade=0`, or 0.0025 in `matched-fade`), Nyquist gate `drop`, and no normalisation (`write_audio` refuses to
clip).

Which of these changes sets the score? The table crosses {0.6, v1} analysis with {0.6, v1} synthesis on the
case-study excerpts, every render scaled to the excerpt's RMS before scoring (bin / band LSD in dB; probe of
2026-10-26). "Raw gain" is the level before that scaling. The 0.6 / 0.6 cell reproduces the [Level](#level)
numbers.

| analysis / synthesis | `bayle` | `de_natura_son` | `saariaho_flute` | raw gain (dB) |
| --- | --- | --- | --- | --- |
| 0.6 / 0.6 | 17.33 / 11.38 | 3.51 / 9.49 | 10.00 / 8.81 | +9.6 … +21.0 (normalised) |
| v1 / 0.6 | 17.37 / 11.39 | 3.44 / 9.45 | 10.03 / 8.61 | +9.7 … +21.2 (normalised) |
| 0.6 / v1 (`fade=0`) | 6.81 / 1.24 | 17.01 / 23.58 | 6.72 / 7.11 | −12.4 … −14.3 (the a/4 scale) |
| v1 / v1 (`fade=0`) | 6.15 / 1.11 | 22.01 / 29.79 | 6.50 / 6.42 | −0.2 … +0.2 |
| v1 / v1 (`fade=0.0025`) | 6.46 / 1.08 | 3.08 / 9.40 | 7.02 / 11.70 | −1.5 … −0.1 |

The 0.6 synthesis path (6.67 kHz grid, a phase reset per row, 2.5 ms fades inside each row) sets the score.
Swapping the analysis changes it by at most 0.2 dB under 0.6 synthesis, and by 0.2–0.7 dB of bin LSD under v1
synthesis. The analysis changes (interpolated peaks, zero padding, amplitude scale) show in the
[accuracy benchmark](../accuracy/README.md) as frequency and amplitude errors, not in a 2048-bin LSD. The 0.6
amplitude scale is −12 dB (a/4), which `render_audio`'s normalisation hides. The fades inside each row are why
0.6 does not click on `de_natura_son`, like v1 with `fade=0.0025`.

## Metrics (`evaluate.py`)

Each render is cut or zero-padded to the length of its input and compared with the mono input at the input's
rate. Both go through the same STFT, `scipy.signal.stft` with a Hann window of 46.4 ms and a hop of 11.6 ms
(2048 and 512 samples at 44.1 kHz, 1024 and 256 at 22.05 kHz, 2229 and 557 at 48 kHz), `boundary=None`,
`padded=False`.

- **Bin LSD.** Magnitudes are floored at 10⁻⁴ × the loudest reference bin (−80 dB), in the reference and in
  the render. The score is the RMS over all bins and frames of the difference in dB. This is the definition
  of the 2026-10-10 audit and of `tests/test_synth.py`, so the numbers can be compared with the 0.6 ones.
- **Band LSD.** Band powers (sums of |X|²) in third-octave bands with edges at 1 kHz · 2^(n/3 + 1/6). Bands
  are merged upward until each holds at least 8 STFT bins (172 Hz), and the remainder up to Nyquist joins the
  last band. That gives 18 bands at 44.1 kHz, 15 at 22.05 kHz and 19 at 48 kHz. Powers are floored at
  10⁻⁸ × the loudest reference band (−80 dB). The score is the RMS over all bands and frames of the
  difference in dB.
- **Gain.** RMS level of the render re the input, in dB.
- `per_band.csv` gives both LSDs for each band as well (the bin LSD over the bins of that band).

Renders are scored as written, with no gain fit, so a level error is part of both LSDs.

### Noise-pair floor

LSD compares magnitudes frame by frame. Two independent white-noise realisations of the same level (10 s, the
same STFT) score **7.89 dB bin LSD and 1.86 dB band LSD** at 44.1 kHz (7.89 / 2.00 at 22.05 kHz, 7.87 / 1.83
at 48 kHz). Any noise-like part of a sound therefore costs about 8 dB of bin LSD even when the model
reproduces the noise's spectrum exactly. Merging bins into bands averages most of that randomness away, so the
band LSD is the more telling number for noisy material. The floor is listed as the `noise-pair` rows in
`per_excerpt.csv` and `per_band.csv`.

A third metric was tried and dropped: LSD over tonal regions only (reference peaks with a prominence of
≥ 10 dB, present in the neighbouring frames). Its mask still passes 7 % of the bins of white noise, and it
did not separate the rows better than the bin LSD.

### Floor effects

Bins below the bin-LSD floor in both signals count as a difference of 0 dB. On `de_natura_son`, 74.5 % of the
reference bins lie below the floor (40.2 % on the flute, 1.5 % on `bayle`), so its bin LSD mostly reads the
loud low bands, and a render that is silent where the reference is quiet scores well. Its band LSD is less
affected. Compare rows within one sound, not across sounds. Scoring each sound at its own rate keeps a
22.05 kHz input from gaining an empty upper half that would dilute its bin LSD.

### Level

Because the 0.6 renderer normalises its output, the level difference dominates that row's scores. With the
render scaled to the RMS of the excerpt, the 0.6 row scores (bin / band LSD in dB) 17.33 / 11.38 on `bayle`,
3.51 / 9.49 on `de_natura_son` and 10.00 / 8.81 on the flute. These values come from `evaluate.score` on the
rescaled render; they are not in `results/`. Every other row is within −3.6 to +0.2 dB of the excerpt, except
PM2 inharmonic on `de_natura_son` (see above).

## Headline results

These tables are copied from [`results/summary.md`](results/summary.md) (run of 2026-10-26). Lower is better.
"–" means the row has no render of that sound.

### Case-study excerpts

The PM2 inharmonic row shows the mean of five runs and the min–max spread. Its runs differ from those of the
first run of this benchmark (step 4.2), so its cells differ too; every other case-study cell is unchanged.
Noise-pair floor: bin LSD 7.89, band LSD 1.86.

**Bin LSD (dB)**

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
| pm2-inharmonic/matched+asp (5 runs) | 28.10 (27.69–28.53) | 4.04 (3.96–4.15) | – |

**Band LSD (dB)**

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
| pm2-inharmonic/matched+asp (5 runs) | 33.62 (29.01–37.50) | 12.13 (12.01–12.28) | – |

**Gain (dB re excerpt RMS)**

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
| pm2-inharmonic/matched+asp (5 runs) | -0.96 (-1.33–-0.79) | -21.80 (-21.89–-21.75) | – |

### ICMC 2000 panel sounds

Mean and median over the 18 sounds that all four historical rows rendered; n is the number of the 27 sounds
the row rendered. Per-sound values are in `per_excerpt.csv`, and `summary.md` also gives each row's mean and
median over all the sounds it rendered.

| row | n | bin LSD mean | bin LSD median | band LSD mean | band LSD median | gain median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/default | 27 | 9.06 | 8.55 | 10.83 | 10.39 | -0.92 |
| audiospylt-v1/matched | 27 | 5.39 | 5.19 | 5.53 | 5.21 | -0.48 |
| audiospylt-v1/matched-fade | 27 | 4.94 | 4.90 | 5.30 | 4.77 | -0.59 |
| audiospylt-0.6/matched | 27 | 12.66 | 13.10 | 12.73 | 12.67 | 6.23 |
| loris/matched | 27 | 6.18 | 6.55 | 7.67 | 6.82 | -2.00 |
| loris/matched+asp | 27 | 7.43 | 7.38 | 9.37 | 8.52 | -2.03 |
| sms-tools/matched | 27 | 8.37 | 7.44 | 7.89 | 8.25 | -1.24 |
| sms-tools/matched+asp | 27 | 8.99 | 8.31 | 8.52 | 8.77 | -0.95 |
| sms-tools/home | 21 | 6.97 | 5.65 | 7.73 | 7.08 | -1.04 |
| sms-tools/home+asp | 21 | 7.76 | 6.84 | 8.70 | 7.41 | -1.03 |
| pm2-harmonic/matched+asp | 27 | 11.69 | 10.71 | 15.65 | 14.56 | -1.82 |
| pm2-inharmonic/matched+asp | 27 | 13.90 | 14.03 | 20.48 | 21.60 | -2.47 |
| ircam-2000/author | 27 | 5.51 | 5.77 | 6.32 | 4.98 | -0.27 |
| loris-2000/author | 24 | 5.30 | 5.32 | 7.00 | 4.93 | -0.66 |
| loris-2000/author-rs | 24 | 5.71 | 5.83 | 7.38 | 5.34 | -0.72 |
| sms-2000/author | 21 | 4.09 | 3.53 | 3.62 | 3.11 | -0.51 |

### Test sounds of Loris and sms-tools

Bin LSD / band LSD in dB.

| row | `loris-clarinet` | `loris-flute` | `sms-bendir` | `sms-vignesh` | `sms-sax-phrase-short` |
| --- | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/default | 14.27 / 22.68 | 7.98 / 7.74 | 2.07 / 5.36 | 8.74 / 8.51 | 5.59 / 12.64 |
| audiospylt-v1/matched | 8.19 / 8.22 | 4.56 / 3.48 | 1.68 / 5.02 | 4.47 / 2.97 | 3.36 / 6.85 |
| audiospylt-v1/matched-fade | 8.42 / 10.49 | 4.51 / 6.39 | 1.42 / 2.46 | 4.45 / 4.80 | 3.26 / 7.18 |
| audiospylt-0.6/matched | 15.10 / 21.53 | 12.59 / 10.14 | 3.31 / 5.12 | 12.89 / 8.47 | 14.64 / 21.92 |
| loris/matched | 8.46 / 12.90 | 5.40 / 8.09 | 1.93 / 3.63 | 6.25 / 5.91 | 3.90 / 7.58 |
| loris/matched+asp | 11.66 / 14.43 | 6.31 / 8.73 | 2.09 / 4.28 | 6.79 / 7.38 | 4.35 / 8.57 |
| loris/home | 6.30 / 11.87 | 5.10 / 8.47 | – | – | – |
| loris/home+asp | 13.10 / 18.86 | 8.39 / 11.53 | – | – | – |
| sms-tools/matched | 10.83 / 11.55 | 7.18 / 8.93 | 1.26 / 2.52 | 8.99 / 7.85 | 2.85 / 6.88 |
| sms-tools/matched+asp | 11.24 / 12.26 | 7.63 / 9.46 | 1.59 / 2.68 | 9.52 / 8.11 | 3.36 / 7.37 |
| sms-tools/home | – | – | 1.93 / 3.62 | 5.63 / 5.53 | 4.16 / 5.04 |
| sms-tools/home+asp | – | – | 1.99 / 3.79 | 6.19 / 5.70 | 4.45 / 5.39 |
| pm2-harmonic/matched+asp | 13.52 / 18.48 | 10.13 / 15.90 | 2.59 / 4.70 | 13.84 / 22.42 | 5.18 / 8.14 |
| pm2-inharmonic/matched+asp | 14.49 / 23.31 | 12.66 / 22.33 | 3.02 / 5.79 | 15.31 / 25.04 | 6.55 / 15.14 |

In short:

- With the matched analysis, v1's per-frame peaks give the lowest LSDs on `bayle` and the flute, because they
  keep every peak above −80 dBFS. On `de_natura_son` they need `fade=0.0025` to avoid clicks, which costs the
  flute 5.3 dB of band LSD.
- Loris is the closest tracker on `bayle` (band LSD 2.02 dB, 0.9 dB above v1 with 14 % fewer points) and on
  the flute's bin LSD. sms-tools is the closest on `de_natura_son`, where it also beats every v1 row. On
  `bayle`, the sms-tools and PM2 limits on the number of partials show up as empty bands.
- Played through `asp.synthesize` (frequency and amplitude only), the same tracks score 0.1–3.1 dB higher
  than with the tool's own synthesizer on the case-study excerpts.
- On the 18 common ICMC sounds, the authors' own renders of 2000 have mean bin LSDs of 4.09 (SMS) to 5.71 dB
  (Loris `rs`) and median band LSDs of 3.11 to 5.34 dB. v1 `matched-fade` lies within that range (4.94 / 4.77);
  today's tools at matched settings lie above it (Loris 6.18 / 6.82, sms-tools 8.37 / 8.25). The SMS 2000
  settings mapped onto sms-tools (`sms-tools/home`, 6.97 / 7.08) do not reach SMS's own 2000 renders; the
  mapping leaves out SMS 2000's residual, attack and synthesis settings.
- On their own test sounds, the tools' home settings give the lowest bin LSD for Loris on the clarinet (6.30
  against 8.19 for v1 `matched`); played through `asp.synthesize`, the same Loris tracks score 13.10. sms-tools
  at matched settings has the lowest bin LSD on `sms-bendir` and `sms-sax-phrase-short`, and its home settings
  the lowest band LSD on `sms-sax-phrase-short` (5.04). On the other cells a v1 matched row is lowest.

## Listening

`evaluate.py` writes `OUT/listening/<sound>/NN_<row>.wav`: `00_original` (the mono input) and one file per row,
using the first run of the PM2 inharmonic row. Every file is scaled to the RMS of the input, then one factor
per sound puts the loudest peak of the set at −1 dBFS, so the files of one sound can be compared at equal
loudness. They are 16-bit WAVs at the input's rate, 35 folders and about 145 MB in total.

## Reproducing

The commands are for bash on Linux and run from the repository root. Each tool needs its own environment:
sms-tools 1.2 requires `numpy<2`, and loristrck is not an AudioSpylt dependency. `run_audiospylt.py` runs
after the other runners because it also replays the track tables that they write. Every runner but
`run_historical.py` takes `--sets case,icmc2000,tools` (default: every set whose files are present).

```sh
OUT=$HOME/asp-fidelity     # reference sounds, renders, track tables, listening WAVs; ~660 MB
VENVS=$HOME/asp-bench-venvs

uv venv --python 3.12 $VENVS/asp
uv pip install --python $VENVS/asp/bin/python '.[plot]'   # the 0.6 shims import plotly
uv venv --python 3.12 $VENVS/loris
uv pip install --python $VENVS/loris/bin/python loristrck==1.7.2 soundfile
uv venv --python 3.12 $VENVS/sms
uv pip install --python $VENVS/sms/bin/python sms-tools==1.2 'numpy<2' soundfile

A=$VENVS/asp/bin/python
python3 benchmarks/fidelity/fetch_reference.py $OUT               # standard library; 32 MB from the Wayback Machine and others
$VENVS/loris/bin/python benchmarks/fidelity/run_loris.py $OUT     # ~30 s
$VENVS/sms/bin/python benchmarks/fidelity/run_sms.py $OUT         # ~35 s
$A benchmarks/fidelity/run_partiels.py $OUT TEMPLATE_DIR [PARTIELS]   # optional, see below
$A benchmarks/fidelity/run_audiospylt.py $OUT                     # ~11 min
$A benchmarks/fidelity/run_historical.py $OUT                     # ~2.5 min
$A benchmarks/fidelity/evaluate.py $OUT benchmarks/fidelity/results   # ~15 s
```

Without the fetch step, only the `case` set runs, and `run_historical.py` has nothing to do. Every runner writes `OUT/raw/<row>/<sound>.wav` (float32, at
the level the tool renders, at the input's rate) and `OUT/raw/<row>/meta.json` (version, parameters,
synthesizer, and per sound its set, rate and counts). The Loris, sms-tools and PM2 runners also write their
tracks to `OUT/tracks/<row>/<sound>.csv` in the format of the accuracy benchmark (time, freq, amp, track).
`evaluate.py` scores the rows that have a `meta.json`, so the PM2 step can be skipped. The first run (step
4.2, `case` only, 2026-10-26) was reproduced exactly from fresh environments, with the PM2 track tables copied
in. The run with the reference sets reproduced its 25 deterministic `case` renders sample for sample; PM2
inharmonic differs from run to run.

### Partiels and PM2

Partiels 2.6.0 and the Ircam Vamp Plugins 2.1.0 are installed as for the
[accuracy benchmark](../accuracy/README.md#partiels-and-pm2); `TEMPLATE_DIR` is the folder with the two PM2
`.ptldoc` templates from the plugins' manuals. The runner reuses that benchmark's template editing and export
code: it writes the matched templates to `OUT/partiels-templates/` and the mono inputs it analyses to
`OUT/mono/`. It runs the inharmonic template five times per case-study excerpt and once per reference sound.

## Files

| file | role |
| --- | --- |
| `excerpts.py` | sets and sounds, mono loading, render and metadata writers (numpy and soundfile only, imported by every runner) |
| `fetch_reference.py`, `reference.json` | download and SHA-256 check of the reference sounds and the authors' renders; the manifest |
| `run_audiospylt.py` | the v1 and 0.6 rows, and the `+asp` replays of every track table in `OUT/tracks/` |
| `run_loris.py`, `run_sms.py`, `run_partiels.py` | one runner per external tool |
| `run_historical.py` | the authors' renders of 2000: resampling, shift, meta.json |
| `evaluate.py` | scoring, listening WAVs, writes `results/` (no AudioSpylt import) |
| `results/summary.md` | generated tables per set and the noise-pair floors |
| `results/per_excerpt.csv` | bin LSD, band LSD and gain per set, row and sound, every PM2 run separately |
| `results/per_band.csv` | both LSDs per set, row, sound and band |
| `results/repeats.csv` | mean, min and max over the PM2 inharmonic runs |
