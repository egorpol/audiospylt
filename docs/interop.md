# Importing partial tracks

AudioSpylt reads partial tracks made by other analysis tools into a `tracks`
table (`track_id`, `time`, `freq`, `amp`, optional `phase`; see
[table-format.md](table-format.md)). From there, `tracks_to_events()` and `synthesize()`
resynthesise them, and `viz.plot_spectrogram(overlay=...)` and `viz.plot_events()`
draw them.

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
