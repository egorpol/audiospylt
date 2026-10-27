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
run of 2026-10-27 are in [`results/`](results/).

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
| `audiospylt-v1/default` | AudioSpylt 1.0.0.dev0: `spectrogram` defaults (frame 2048, hop 512 samples at every rate, Hann, `zero_pad=1`, centred frames), `find_peaks(min_db=-60)` per frame | `synthesize` defaults: continuous phase, rows linked within `link_cents=100`, `link_ramp=0.005`, `fade=0.005` centred on voice ends |
| `audiospylt-v1/matched` | matched frame and hop, Hann, `zero_pad=4`; `find_peaks(min_db=-80)` | as above |
| `audiospylt-v1/matched-nofade` | as `matched` | as above with `fade=0` (kept to show the clicks) |
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
  clicks: in `audiospylt-v1/matched-nofade`, every band above 354 Hz has a band LSD between 18.9 and 42.1 dB.
  The library default (centred 5 ms fades) removes the clicks: on `de_natura_son` the band excess falls from
  29.98 to 1.94 dB, and the band LSD from 29.98 to 9.55 dB. On the flute the fade-0 row still scores a lower
  band LSD (6.46 against 11.67 dB) because its click noise fills the high bands where the peaks leave out the
  breath noise, which LSD counts as closer. The split shows it: the band excess is 0.88 dB without fades and
  0.77 dB with them, and the band deficit 6.40 and 11.65 dB.
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
  below the excerpt in 1782–2245 Hz in 50 % of the frames and in each band above 5.7 kHz in 43–97 %. On
  `de_natura_son`, it misses much of the sub-bass that carries the excerpt's energy: in run 1 the 0–177 Hz
  band is 11.5 dB low (band LSD 24.4 dB), and the gain is −11.9 dB.
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
into one voice that keeps its phase), a 5 ms raised-cosine ramp on amplitude steps at links, the library
default 5 ms voice fades centred on voice ends (`fade=0` in `matched-nofade`), Nyquist gate `drop`, and no
normalisation (`write_audio` refuses to clip).

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
- **Excess / deficit.** With the same floored dB values, `d = render_dB − reference_dB` per bin or band, per
  frame; excess is the RMS of `max(d, 0)` and deficit the RMS of `min(d, 0)` over the same cells as the LSD,
  so `LSD² = excess² + deficit²`. Excess is sound the render adds (such as clicks); deficit is sound it
  misses (such as noise a sinusoidal model leaves out).
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

These tables are copied from [`results/summary.md`](results/summary.md) (run of 2026-10-27). Lower is better.
"–" means the row has no render of that sound. The excess / deficit tables split the band LSD into the
part where the render is louder than the sound and the part where it is quieter (see
[Metrics](#metrics-evaluatepy)); `summary.md` gives the same split of the bin LSD.

### Case-study excerpts

The PM2 inharmonic row shows the mean of five runs and the min–max spread; its excess / deficit cells show
the means. Noise-pair floor: bin LSD 7.89, band LSD 1.86.

**Bin LSD (dB)**

| row | bayle | de_natura_son | saariaho_flute |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | 21.13 | 3.24 | 12.00 |
| audiospylt-v1/matched | 6.45 | 2.56 | 7.05 |
| audiospylt-v1/matched-nofade | 6.19 | 22.20 | 6.51 |
| audiospylt-0.6/matched | 16.76 | 5.33 | 17.34 |
| loris/matched | 7.48 | 3.11 | 7.08 |
| loris/matched+asp | 10.12 | 3.17 | 8.81 |
| sms-tools/matched | 20.32 | 2.17 | 7.31 |
| sms-tools/matched+asp | 20.63 | 2.40 | 7.48 |
| pm2-harmonic/matched+asp | – | – | 10.26 |
| pm2-inharmonic/matched+asp (5 runs) | 28.08 (27.67–28.51) | 3.58 (3.50–3.70) | – |

**Band LSD (dB)**

| row | bayle | de_natura_son | saariaho_flute |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | 8.49 | 10.88 | 22.61 |
| audiospylt-v1/matched | 1.11 | 9.55 | 11.67 |
| audiospylt-v1/matched-nofade | 1.14 | 29.98 | 6.46 |
| audiospylt-0.6/matched | 12.82 | 13.17 | 21.88 |
| loris/matched | 2.02 | 10.11 | 12.39 |
| loris/matched+asp | 2.20 | 10.42 | 13.72 |
| sms-tools/matched | 11.67 | 8.71 | 12.23 |
| sms-tools/matched+asp | 11.83 | 9.64 | 12.43 |
| pm2-harmonic/matched+asp | – | – | 16.90 |
| pm2-inharmonic/matched+asp (5 runs) | 33.52 (28.85–37.44) | 10.86 (10.72–11.04) | – |

**Gain (dB re excerpt RMS)**

| row | bayle | de_natura_son | saariaho_flute |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | -0.16 | -1.79 | -0.19 |
| audiospylt-v1/matched | -0.18 | -0.45 | -0.09 |
| audiospylt-v1/matched-nofade | -0.17 | 0.21 | -0.08 |
| audiospylt-0.6/matched | 9.55 | 10.61 | 20.99 |
| loris/matched | -0.47 | -3.62 | -0.44 |
| loris/matched+asp | -0.48 | -3.62 | -0.45 |
| sms-tools/matched | -0.65 | -2.03 | -0.44 |
| sms-tools/matched+asp | -0.36 | -1.30 | -0.16 |
| pm2-harmonic/matched+asp | – | – | -0.40 |
| pm2-inharmonic/matched+asp (5 runs) | -0.91 (-1.28–-0.75) | -11.86 (-11.90–-11.81) | – |

**Band excess / deficit (dB)**

| row | bayle | de_natura_son | saariaho_flute |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | 0.72 / 8.46 | 1.05 / 10.83 | 0.23 / 22.61 |
| audiospylt-v1/matched | 0.79 / 0.78 | 1.94 / 9.35 | 0.77 / 11.65 |
| audiospylt-v1/matched-nofade | 0.85 / 0.76 | 29.98 / 0.07 | 0.88 / 6.40 |
| audiospylt-0.6/matched | 11.87 / 4.86 | 11.68 / 6.08 | 21.88 / 0.21 |
| loris/matched | 0.47 / 1.97 | 0.15 / 10.11 | 0.57 / 12.38 |
| loris/matched+asp | 0.22 / 2.19 | 0.06 / 10.42 | 0.20 / 13.72 |
| sms-tools/matched | 0.71 / 11.65 | 0.45 / 8.70 | 0.13 / 12.23 |
| sms-tools/matched+asp | 0.71 / 11.81 | 0.86 / 9.61 | 0.29 / 12.43 |
| pm2-harmonic/matched+asp | – | – | 1.31 / 16.85 |
| pm2-inharmonic/matched+asp (5 runs) | 0.59 / 33.51 | 0.67 / 10.84 | – |

### ICMC 2000 panel sounds

Mean and median over the 18 sounds that all four historical rows rendered; n is the number of the 27 sounds
the row rendered. Per-sound values are in `per_excerpt.csv`, and `summary.md` also gives each row's mean and
median over all the sounds it rendered.

| row | n | bin LSD mean | bin LSD median | band LSD mean | band LSD median | gain median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/default | 27 | 9.82 | 9.72 | 14.36 | 14.76 | -1.03 |
| audiospylt-v1/matched | 27 | 4.82 | 4.76 | 5.16 | 4.69 | -0.53 |
| audiospylt-v1/matched-nofade | 27 | 5.39 | 5.19 | 5.53 | 5.21 | -0.48 |
| audiospylt-0.6/matched | 27 | 12.66 | 13.10 | 12.73 | 12.67 | 6.23 |
| loris/matched | 27 | 6.18 | 6.55 | 7.67 | 6.82 | -2.00 |
| loris/matched+asp | 27 | 6.97 | 6.89 | 8.75 | 7.52 | -1.91 |
| sms-tools/matched | 27 | 8.37 | 7.44 | 7.89 | 8.25 | -1.24 |
| sms-tools/matched+asp | 27 | 8.68 | 7.91 | 8.14 | 8.42 | -0.86 |
| sms-tools/home | 21 | 6.97 | 5.65 | 7.73 | 7.08 | -1.04 |
| sms-tools/home+asp | 21 | 7.47 | 6.36 | 8.11 | 6.84 | -0.95 |
| pm2-harmonic/matched+asp | 27 | 11.68 | 10.71 | 15.58 | 14.45 | -1.82 |
| pm2-inharmonic/matched+asp | 27 | 13.72 | 13.91 | 19.88 | 20.98 | -2.35 |
| ircam-2000/author | 27 | 5.51 | 5.77 | 6.32 | 4.98 | -0.27 |
| loris-2000/author | 24 | 5.30 | 5.32 | 7.00 | 4.93 | -0.66 |
| loris-2000/author-rs | 24 | 5.71 | 5.83 | 7.38 | 5.34 | -0.72 |
| sms-2000/author | 21 | 4.09 | 3.53 | 3.62 | 3.11 | -0.51 |

**Band excess / deficit (dB)**, mean and median over the 18 sounds

| row | n | mean | median |
| --- | ---: | ---: | ---: |
| audiospylt-v1/default | 27 | 0.97 / 14.30 | 0.69 / 14.70 |
| audiospylt-v1/matched | 27 | 1.16 / 4.96 | 0.89 / 4.46 |
| audiospylt-v1/matched-nofade | 27 | 4.02 / 2.88 | 3.51 / 2.95 |
| audiospylt-0.6/matched | 27 | 10.75 / 4.84 | 10.33 / 3.47 |
| loris/matched | 27 | 0.82 / 7.56 | 0.50 / 6.75 |
| loris/matched+asp | 27 | 0.40 / 8.73 | 0.16 / 7.49 |
| sms-tools/matched | 27 | 0.59 / 7.84 | 0.32 / 7.98 |
| sms-tools/matched+asp | 27 | 0.67 / 8.08 | 0.38 / 8.17 |
| sms-tools/home | 21 | 0.80 / 7.59 | 0.31 / 6.53 |
| sms-tools/home+asp | 21 | 0.76 / 8.01 | 0.35 / 6.67 |
| pm2-harmonic/matched+asp | 27 | 1.76 / 15.30 | 0.87 / 13.99 |
| pm2-inharmonic/matched+asp | 27 | 0.51 / 19.86 | 0.23 / 20.96 |
| ircam-2000/author | 27 | 1.68 / 5.79 | 1.48 / 4.69 |
| loris-2000/author | 24 | 0.81 / 6.82 | 0.43 / 4.87 |
| loris-2000/author-rs | 24 | 0.67 / 7.25 | 0.34 / 5.15 |
| sms-2000/author | 21 | 1.88 / 2.37 | 1.66 / 1.09 |

### Test sounds of Loris and sms-tools

Bin LSD / band LSD in dB.

| row | `loris-clarinet` | `loris-flute` | `sms-bendir` | `sms-vignesh` | `sms-sax-phrase-short` |
| --- | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/default | 14.40 / 23.79 | 8.89 / 14.23 | 2.28 / 5.87 | 9.65 / 13.01 | 5.64 / 13.33 |
| audiospylt-v1/matched | 8.40 / 10.56 | 4.42 / 6.31 | 1.37 / 2.50 | 4.37 / 4.73 | 3.18 / 7.06 |
| audiospylt-v1/matched-nofade | 8.19 / 8.22 | 4.56 / 3.48 | 1.68 / 5.02 | 4.47 / 2.97 | 3.36 / 6.85 |
| audiospylt-0.6/matched | 15.10 / 21.53 | 12.59 / 10.14 | 3.31 / 5.12 | 12.89 / 8.47 | 14.64 / 21.92 |
| loris/matched | 8.46 / 12.90 | 5.40 / 8.09 | 1.93 / 3.63 | 6.25 / 5.91 | 3.90 / 7.58 |
| loris/matched+asp | 11.39 / 13.91 | 6.08 / 8.50 | 2.00 / 3.88 | 6.34 / 6.76 | 4.24 / 8.31 |
| loris/home | 6.30 / 11.87 | 5.10 / 8.47 | – | – | – |
| loris/home+asp | 9.85 / 13.41 | 6.38 / 9.21 | – | – | – |
| sms-tools/matched | 10.83 / 11.55 | 7.18 / 8.93 | 1.26 / 2.52 | 8.99 / 7.85 | 2.85 / 6.88 |
| sms-tools/matched+asp | 10.98 / 11.82 | 7.45 / 9.14 | 1.48 / 2.61 | 9.30 / 7.92 | 3.26 / 7.23 |
| sms-tools/home | – | – | 1.93 / 3.62 | 5.63 / 5.53 | 4.16 / 5.04 |
| sms-tools/home+asp | – | – | 1.97 / 3.68 | 6.17 / 5.64 | 4.45 / 5.37 |
| pm2-harmonic/matched+asp | 13.51 / 18.41 | 10.13 / 15.88 | 2.59 / 4.64 | 13.84 / 22.42 | 5.18 / 8.11 |
| pm2-inharmonic/matched+asp | 14.46 / 23.08 | 12.63 / 22.20 | 2.87 / 5.33 | 15.20 / 24.75 | 6.48 / 14.86 |

**Band excess / deficit (dB)**

| row | `loris-clarinet` | `loris-flute` | `sms-bendir` | `sms-vignesh` | `sms-sax-phrase-short` |
| --- | ---: | ---: | ---: | ---: | ---: |
| audiospylt-v1/default | 0.82 / 23.77 | 0.52 / 14.22 | 1.06 / 5.77 | 0.51 / 13.00 | 1.71 / 13.22 |
| audiospylt-v1/matched | 1.66 / 10.43 | 0.56 / 6.28 | 1.19 / 2.20 | 0.63 / 4.69 | 2.03 / 6.76 |
| audiospylt-v1/matched-nofade | 1.73 / 8.04 | 2.66 / 2.23 | 4.78 / 1.55 | 0.94 / 2.82 | 3.33 / 5.98 |
| audiospylt-0.6/matched | 20.27 / 7.25 | 9.79 / 2.64 | 4.30 / 2.78 | 7.66 / 3.62 | 21.87 / 1.48 |
| loris/matched | 0.48 / 12.89 | 0.29 / 8.08 | 0.78 / 3.55 | 0.70 / 5.86 | 1.46 / 7.44 |
| loris/matched+asp | 0.35 / 13.91 | 0.10 / 8.50 | 0.09 / 3.88 | 0.23 / 6.75 | 0.85 / 8.27 |
| loris/home | 0.10 / 11.87 | 0.37 / 8.47 | – | – | – |
| loris/home+asp | 0.05 / 13.41 | 0.05 / 9.21 | – | – | – |
| sms-tools/matched | 0.38 / 11.54 | 0.22 / 8.92 | 1.05 / 2.29 | 0.34 / 7.84 | 1.32 / 6.75 |
| sms-tools/matched+asp | 1.58 / 11.72 | 0.29 / 9.14 | 1.02 / 2.40 | 0.42 / 7.91 | 1.80 / 7.00 |
| sms-tools/home | – | – | 0.76 / 3.54 | 0.19 / 5.52 | 0.69 / 5.00 |
| sms-tools/home+asp | – | – | 0.71 / 3.61 | 0.55 / 5.61 | 0.46 / 5.35 |
| pm2-harmonic/matched+asp | 0.33 / 18.40 | 0.39 / 15.88 | 1.31 / 4.45 | 0.24 / 22.42 | 0.93 / 8.06 |
| pm2-inharmonic/matched+asp | 1.41 / 23.03 | 0.14 / 22.20 | 0.63 / 5.29 | 0.38 / 24.75 | 1.43 / 14.80 |

In short:

- With the matched analysis and the library's default fades (`audiospylt-v1/matched`), v1's per-frame peaks
  score lower than every tracker on `bayle` (bin / band LSD 6.45 / 1.11 dB) and the flute (7.05 / 11.67 dB;
  Loris 7.08 / 12.39, sms-tools 7.31 / 12.23), because they keep every peak above −80 dBFS.
- Without fades (`matched-nofade`), the unlinked rumble peaks of `de_natura_son` click (band LSD 29.98 dB
  against 9.55 dB with fades). On the flute the same row scores a lower band LSD (6.46 dB), because its clicks
  fill the high bands where the peaks miss the breath noise: its band deficit is 6.40 dB, against 11.65 dB with
  fades.
- Loris is the closest tracker on `bayle` (band LSD 2.02 dB, 0.9 dB above v1 with 14 % fewer points) and on
  the flute's bin LSD. sms-tools is the closest on `de_natura_son`, where it also beats every v1 row (bin /
  band LSD 2.17 / 8.71 dB, v1 `matched` 2.56 / 9.55). On `bayle`, the sms-tools and PM2 limits on the number
  of partials show up as empty bands.
- Played through `asp.synthesize` (frequency and amplitude only), the same tracks score 0.1–2.6 dB higher
  than with the tool's own synthesizer on the case-study excerpts.
- On the 18 common ICMC sounds, the authors' own renders of 2000 have mean bin LSDs of 4.09 (SMS) to 5.71 dB
  (Loris `rs`) and median band LSDs of 3.11 to 5.34 dB. v1 `matched` lies within that range (4.82 / 4.69);
  today's tools at matched settings lie above it (Loris 6.18 / 6.82, sms-tools 8.37 / 8.25). The SMS 2000
  settings mapped onto sms-tools (`sms-tools/home`, 6.97 / 7.08) do not reach SMS's own 2000 renders; the
  mapping leaves out SMS 2000's residual, attack and synthesis settings.
- On their own test sounds, the tools' home settings give the lowest bin LSD for Loris on the clarinet (6.30
  against 8.40 for v1 `matched`); played through `asp.synthesize`, the same Loris tracks score 9.85. sms-tools
  at matched settings has the lowest bin LSD on `sms-bendir` and `sms-sax-phrase-short`, and its home settings
  the lowest band LSD on `sms-sax-phrase-short` (5.04). On the other cells v1 `matched` is lowest, except that
  `matched-nofade` has a lower band LSD on `loris-clarinet`, `loris-flute` and `sms-vignesh` (8.22, 3.48 and
  2.97 against 10.56, 6.31 and 4.73 dB), with a lower band deficit, as on the case-study flute.

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
| `results/per_excerpt.csv` | bin LSD, band LSD, gain, and bin/band excess and deficit per set, row and sound, every PM2 run separately |
| `results/per_band.csv` | both LSDs per set, row, sound and band |
| `results/repeats.csv` | mean, min and max over the PM2 inharmonic runs |
