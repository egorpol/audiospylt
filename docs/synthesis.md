# Synthesis modes

`synthesize(events, sr)` renders an events table as a sum of sinusoids, one per
row, directly at `sr`. Within a row, frequency and amplitude move in straight
lines from their `_start` to their `_stop` values. What differs between the
modes is what happens where one row ends and the next begins, which matters
most for frame-by-frame tables (one row per partial and analysis frame, such as
`data/tables/conga_multidft.tsv`).

## Choosing a mode

| `phase` | at a row boundary | use it for | listen for |
| --- | --- | --- | --- |
| `"continuous"` (default) | a row that starts where another stops, within `link_cents` (100) of its frequency, continues its phase; an amplitude step there becomes a raised-cosine ramp of `link_ramp` (5 ms) | resynthesis of analysed frames; morphs chained between held snapshots | smooth joins; the ramp blurs attacks by up to 2.5 ms on either side of a frame edge |
| `"reset"` | every row restarts at phase 0 | the 0.6 sound (0.6 `reset` and `chirp`); always together with `fade`, e.g. `fade=0.0025` | a soft pulse at each frame edge (each row fades in and out); without `fade`, clicks |
| `"absolute"` | the phase follows the clock: a row at frequency *f* starting at *t* has phase 2π·*f*·*t* (0.6 `global`) | tables whose partials repeat the same frequency across frames | joins are seamless only where frequency and amplitude stay the same and `fade=0`; with `fade`, every row edge dips like `"reset"` |

The other options:

- `fade` (seconds, default 0.005): raised-cosine fade at the start and end of
  each voice. In `"continuous"` each ramp is centred on the voice end (half
  before, half after); a side that would leave the output stays inside instead.
  In `"reset"` and `"absolute"` every row is its own voice and the ramps stay
  inside. `fade=0` leaves unfaded ends, which click on frame tables.
- `link_ramp` (seconds, default 0.005): only in `"continuous"`, only at links
  where `amp_stop` of one row differs from `amp_start` of the next. The ramp is
  centred on the link and shortened so it covers at most half of either row.
  `link_ramp=0` keeps hard steps.
- `link_cents` (default 100): the largest frequency step a link bridges;
  `None` links any frequencies. An integer `track_id` column limits links to
  rows with the same id.
- `nyquist`: `"drop"` (default) silences samples at or above `sr / 2` and
  warns; `"alias"` lets them fold.
- `duration`: output length in seconds; the default ends at the last
  `time_stop`.

## What the ramp does

`conga_multidft.tsv` holds 87 constant rows in 4 frames (edges at 0.1, 0.2 and
0.4 s). 27 rows continue a partial from the previous frame; the amplitude steps
at those links are 14 dB (median) and up to 24 dB. Energy above 6 kHz within
2 ms of each frame edge, relative to the render's RMS, all with `fade=0.0025`:

| render | 0.1 s | 0.2 s | 0.4 s |
| --- | --- | --- | --- |
| `"continuous"`, `link_ramp=0` (hard steps) | −32 dB | −47 dB | −73 dB |
| `"continuous"`, `link_ramp=0.0025` | −77 dB | −103 dB | −112 dB |
| `"continuous"`, default `link_ramp=0.005` | −77 dB | −102 dB | −124 dB |
| `"reset"` | −81 dB | −101 dB | −119 dB |
| `"absolute"` | −87 dB | −102 dB | −120 dB |

From 5 ms on, what is left comes from the fades of the rows that start or end at
the frame edges, not from the links.

## Tracks

`tracks_to_events(tracks)` turns a `tracks` table (points of partials over time,
such as `f0_to_tracks` returns) into events: each pair of consecutive points of a
track becomes one row, so frequency and amplitude move in straight lines from
point to point. The rows keep their `track_id`, and neighbouring rows of a track
share their boundary values, so `"continuous"` plays each track as one voice,
with no phase jumps and no link ramps. Two tracks never join, even where one
starts on the frequency at which another stops. A track with a single point
gives no row, and a `phase` column is not used.

A track starts and stops at the amplitude of its first and last point; the
default `fade` covers those ends unless they are already 0. On a 223.3 Hz sine
tracked by `pyin`, whose last point has amplitude 0.055, energy above 6 kHz
within 2 ms of the track ends is −51 dB re the render's RMS with `fade=0` and
−102 dB with the default `fade`.

```python
import audiospylt as asp

x, sr = asp.load_audio("samples/cello_arco_d_sul_c_01.wav", mono=True)
contour = asp.pyin(x, sr, fmin=65.41, fmax=2093.0)
audio = asp.synthesize(asp.tracks_to_events(asp.f0_to_tracks(contour)), sr)
asp.write_audio("cello_f0.wav", audio, sr, normalize=-1.0)
```

## Morphs

`morph(source, target, duration, method)` returns an events table, so it renders
the same way. When several rows share one partial, `duplicates="full"` (the
default for `equal`, `power` and `sigmoid`, as in 0.6) gives every copy the full
amplitude: copies that leave one partial start in phase and add up, copies that
arrive at one partial beat slowly against each other, and that end is louder
than the snapshot. `duplicates="primary"` (the default for `transport`) keeps
both ends at the snapshots' levels.

## Example

Run from the repository root. It writes three files to compare.

```python
import warnings

from audiospylt import read_table, synthesize, write_audio

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the file uses 0.6 column names
    events = read_table("data/tables/conga_multidft.tsv")

for name, options in {
    "continuous": {},
    "continuous_hard": {"link_ramp": 0},
    "reset": {"phase": "reset"},
}.items():
    audio = synthesize(events, 44100, fade=0.0025, **options)
    write_audio(f"conga_{name}.wav", audio, 44100, normalize=-1.0)
```
