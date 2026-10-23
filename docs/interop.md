# Importing partial tracks

AudioSpylt reads partial tracks made by other analysis tools into a `tracks`
table (`track_id`, `time`, `freq`, `amp`, optional `phase`; see
[table-format.md](table-format.md)). From there, `tracks_to_events()` and `synthesize()`
resynthesise them, `viz.plot_spectrogram(overlay=...)` and `viz.plot_events()`
draw them, and `tracks_to_chords()` reduces them to chords for notation
([below](#notation-a-chord-per-time-window)). `write_sdif()` writes them back out as SDIF
([SDIF](#sdif-audiosculpt-pm2-loris-spear)).

## Partiels (Ircam PM2)

[Partiels](https://github.com/Ircam-Partiels/Partiels) runs Vamp plugins on a
sound and exports their results. The Ircam Vamp Plugins include PM2 partial
tracking, with document templates "PM2 Partials Tracking Harmonic/Inharmonic/
Chord" of 20 tracks: one track ("slot") per partial. Export the partial tracks
as CSV or JSON, from the application or the command line (one file per track),
and read the folder:

```python
import audiospylt as asp

tracks = asp.read_partiels("exports/flute")  # a folder of Group_1_Partial_*.json files
low = tracks[tracks["partial_id"] <= 4]  # harmonics 1-4 (JSON exported with --description)

x, sr = asp.load_audio("samples/saariaho_flute.wav", mono=True)
fig = asp.viz.plot_spectrogram(asp.spectrogram(x, sr), overlay=tracks)
resynthesis = asp.synthesize(asp.tracks_to_events(tracks), sr, fade=0.005)
```

On the Saariaho flute excerpt (harmonic template, 19 partials, channel 0) this
gives 398 tracks with 17,310 points, and the resynthesis comes within 0.03 dB of
the recording's RMS level.

From the command line (Partiels 2.6; `--description` keeps the plugin and each
track's PM2 partial ID in the JSON):

```sh
Partiels --export -i flute.wav -t PM2-Partials-Tracking-Harmonic-20.ptldoc -o exports/flute -f json --description
```

`-f csv` works too, with any `--header` (`none`, `generic`, `specific`). CSV
keeps 6 significant digits and loses the partial IDs, so prefer JSON. The
templates also hold an FFT track (13 MB as CSV for an 8-second sound) and, for
harmonic tracking, the SuperVP f0 track; `read_partiels` skips both when it
reads a folder. Export each format to its own folder: a folder with both CSV
and JSON raises. Exports from the application read the same as from the
command line (checked on the flute: identical points and partial IDs).

### What `read_partiels` does

- **Tracks.** Each file is one PM2 slot. Every frame holds a point or nothing,
  and each run of frames between gaps becomes one `track_id`. Tracks are
  numbered by start time, then start frequency. `max_step_cents=` also ends a
  track where the frequency jumps by more than that from one frame to the next.
  The default (None) splits at gaps only, so a harmonic stays one track when
  the f0 jumps an octave.
- **Duplicates.** Inharmonic PM2 in Partiels can write the same point (time,
  frequency, amplitude) into several slots. `dedupe=True` (default) keeps each
  point once, in the longest run that holds it; shorter runs lose those points
  and split around them. `attrs["meta"]["duplicates_dropped"]` counts them.
- **Channels.** Partiels analyses every channel separately. `channel=0`
  (default) reads the first. `attrs["meta"]["channels"]` says how many there
  were.
- **Values.** `amp` is PM2's amplitude: a sine's peak amplitude re full scale,
  the convention of `spectrum()` (a 0.5 sine reads 0.519, +0.33 dB). `phase` is
  PM2's phase in radians, the phase of a sine at the frame time (the reference
  of `synthesize()`). `partial_id` is PM2's partial ID (the harmonic number for
  harmonic tracking) from JSON with a description, and missing otherwise.
  Times are PM2's frame times; a first frame half a sample before 0 is read at 0.

### Known limits of PM2 in Partiels

Measured with Partiels 2.6.0, Ircam Vamp Plugins 2.1.0 (PM2 5) and the
templates' settings (23.2 ms Hann window, threshold -90 dB, 20 partials):

- **Harmonic tracking** is deterministic: the same export every run. Each
  harmonic follows the SuperVP f0, so its frequency jumps whenever the f0 makes
  an octave error.
- **Inharmonic tracking** is not reproducible. About a quarter of the points of
  one run are missing from a second run on the same file, and the same holds
  for the two channels of a dual-mono file within one run. Slots hold copies of each other's points: on a cello
  note, 15,666 points of which 10,642 are distinct. The slots also miss most
  frames of stable partials: with all 20 slots pooled, the 148 Hz partial is
  present in 37 % of the frames and the fundamental in none. No import setting
  can recover what the export does not contain.
- **Low notes.** With the 23 ms window, stationary sines read sharp: 220 Hz
  +46 cents, 440 Hz +13 cents, 880 Hz +1.4 cents. A longer window helps.

## SDIF (AudioSculpt, PM2, Loris, SPEAR)

SDIF files hold analyses as frames of matrices. Three matrix types carry
partial tracks: `1TRC` (sinusoidal tracks: AudioSculpt and PM2, the CNMAT
tools, loristrck), `1HRM` (PM2 harmonic tracks) and `RBEP` (Loris and SPEAR
breakpoints). Each row holds an index, a frequency, an amplitude and a phase.
AudioSculpt's chord-sequence analysis writes `1MRK` marker frames instead.

```python
import audiospylt as asp

tracks = asp.read_sdif("snare-trc.sdif")        # 1TRC, 1HRM or RBEP: the one the file holds
right = asp.read_sdif("drum-trc.sdif", stream=1)  # PM2 writes a stereo analysis as streams 0 and 1
chords = asp.read_chord_seq("snare-cs.sdif")      # AudioSculpt chord sequence -> events table
asp.write_sdif(tracks, "out.sdif")               # 1TRC, laid out as AudioSculpt writes it
```

### What `read_sdif` does

The SDIF index becomes the Int64 `partial_id` column. What it means, and how
it makes tracks, depends on the matrix type:

| Matrix | Index | A track is |
| --- | --- | --- |
| `1TRC` | partial number (PM2: from 1, never reused; CNMAT pm 1.2.2: a slot that later partials reuse) | a run of consecutive frames that hold the index |
| `1HRM` | harmonic number | the same: a harmonic that drops out and returns is a new track |
| `RBEP` | partial number (Loris: from 0) | all breakpoints of the index, at frame time + offset |

- **Gaps.** A `1TRC` or `1HRM` frame without an index ends that index's
  track. PM2's `1TRC` files have none (12 files checked). Harmonic files do:
  in an IRCAM example analysis (PM2 1.3.2, 20 harmonics), 14 harmonics drop
  out and return 178 times in all, so it reads as 198 tracks. `RBEP`
  breakpoints are spread over frames of irregular times, so a missing frame
  says nothing and an index is never split.
- **Values.** `amp` is the writer's amplitude. For PM2 and Loris this is a
  sine's peak amplitude, like `spectrum()`: a resynthesis of a Loris analysis
  of the flute excerpt comes within 0.4 dB of the recording's RMS level.
  `phase` is in radians. `RBEP` adds `bandwidth`, the Loris noise share
  (0 to 1). Times just below 0 (PM2 places its first frame half a sample
  early) are read as 0.
- **Options.** `max_step_cents=` also ends a track where the frequency jumps,
  as in `read_partiels`. `matrix=` picks a type when a file holds several.
  `stream=` picks a stream (default: the lowest that holds the matrix);
  `attrs["meta"]["streams"]` lists them all.
- **Metadata.** `attrs["meta"]` keeps the file's name-value tables (`nvt`),
  with `table_name` (`SinusoidalTracks`, `HarmonicTracks`) and `written_by`
  (`Pm2_Version_...`) when present. Other frame types in the file (f0,
  markers, FFT) are ignored.

The reader needs only numpy. It matched pysdif3 0.9.1 frame for frame on 36
test files (AudioSculpt and PM2, the CNMAT tools, CHANT, OpenMusic, Loris and
the IRCAM SDIF test set). pysdif3 itself is not a dependency: its current
release (1.0.1) cannot open files without a `1TYP` declaration, and none of
the AudioSculpt and PM2 `1TRC` and `1HRM` files tested has one; 0.9.1 opens
them but has no wheels beyond Python 3.11.

### What `write_sdif` writes

One `1NVT` frame (`TableName SinusoidalTracks`, `WrittenBy audiospylt
<version>`, then any `nvt=` entries) and one `1TRC` frame per distinct time
of the table, on stream 0. Each row is index `track_id + 1`, frequency,
amplitude and phase as float32; frame times stay float64. A track without
phases gets them integrated from its frequency. pysdif3 0.9.1 reads the
written flute PM2 import (398 tracks, 950 frames) bit for bit, and so does
the IRCAM library inside Partiels (`Partiels --sdif2json -f 1TRC -m 1TRC -c 1`
returns all 17,310 frequencies). `read_sdif` reads it back with
`partial_id = track_id + 1`.

SDIF readers end a track at a frame that lacks its index. When tracks do not
share one time grid, a track therefore gets a point at every frame time
inside its span: frequency and amplitude interpolated linearly, phase
integrated from the previous point. Tables on one grid (Partiels, `pyin`,
`1TRC` files) are written point for point. `RBEP` breakpoints all have
different times, so the fill multiplies them: the Loris flute analysis
(12,431 partials, 62,368 breakpoints) becomes 3.7 million rows and 62 MB,
against 3.1 MB as `RBEP`. Its resynthesis keeps the level and the spectrum
(within 0.14 dB); 1,849 of the 9,224 partials start at another phase,
because `synthesize` starts a voice at its first row that covers a sample.

### Chord sequences

`read_chord_seq` reads AudioSculpt's chord-sequence analysis: at each marker,
a `1TRC` matrix lists the partials that start there and a `1END` matrix the
ones that end, each holding one frequency and amplitude in between. The
result is an `events` table with one row per partial and a `partial_id`;
`attrs["meta"]["markers"]` keeps the marker times. `events_to_mei(chords)`
writes one measure per segment and `synthesize(chords, sr)` plays it. The
snare-drum example in the tests holds 235 partials in 18 segments.

## Notation: a chord per time window

A `tracks` table holds a point per analysis frame (every 7.8 ms for PM2 on the
flute), so `events_to_mei(tracks_to_events(tracks))` would write a chord per
frame. `tracks_to_chords(tracks, windows)` reduces the tracks to one chord per
time window, and `tracks_at(tracks, time)` reads every track at one time as a
snapshot:

```python
tracks = asp.read_partiels("exports/flute")
chords = asp.tracks_to_chords(tracks, 0.5, max_notes=8)  # 0.5 s windows, the 8 loudest notes
asp.nb.show_mei(asp.events_to_mei(chords, measure_seconds=0.5, labels=True))
asp.nb.show_mei(asp.snapshot_to_mei(asp.tracks_at(tracks, 2.9)))  # 19 partials at 2.9 s
```

- **Notes.** Each track that sounds in a window gives one note held over the
  window. Its frequency is the amplitude-weighted median of the track's
  frequencies there, so a few frames of an octave error do not move it (a mean
  would write a pitch in between). Its amplitude is the track's average over
  the window: a track heard in a fifth of the window comes out 14 dB quieter.
- **Duplicates.** Analyses split one partial over several tracks: harmonic PM2
  at every octave jump of the f0, inharmonic PM2 in copies across slots. In
  each window the loudest note is kept first, then each next loudest unless it
  lies within `merge_cents` (50) of a note already kept. `merge_cents=None`
  keeps one note per track.
- **Thinning** is off by default. `min_db=-40` drops notes more than 40 dB
  below the window's loudest note, `max_notes=8` keeps the 8 loudest.
- **Windows** are a step in seconds (from 0 up to the last point) or your own
  boundary times, such as onsets. A window where nothing sounds gives no chord;
  the score shows empty measures there.

On the flute excerpt (398 tracks, 7.37 s), 0.5 s windows give 15 chords:

| Options | Notes per chord, median (range) | Highest note |
| --- | --- | --- |
| `merge_cents=None` | 40 (23–88) | 12.3 kHz |
| defaults | 21 (19–28) | 12.3 kHz |
| `min_db=-40` | 16 (5–21) | 6.5 kHz |
| `max_notes=8` | 8 | 4.5 kHz |

The inharmonic cello export (3,755 tracks, 17 chords) gives medians of 220,
57, 31 and 8 notes. With the defaults, the notes of a chord lie more than
50 cents apart, so no two share a written quarter-tone pitch.

The chords keep the `track_id` of the track each note comes from, so
`synthesize(chords, sr, fade=0.005)` plays the reduction. A track kept in
consecutive windows continues as one voice where its note moves by at most
`link_cents` (100 cents). In the flute reduction with `max_notes=8`, 85 of the
120 notes continue into the next window this way.
