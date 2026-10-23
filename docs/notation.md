# Notation of events on a time axis

`events_to_mei(events)` writes each onset of an events table (each distinct
`time_start`) as a chord, followed by empty measures for the gap to the next
onset (`measure_seconds`, default 0.5 s per measure). Verovio engraves those
measures with readable spacing, not at true time scale: a chord with many
accidentals takes more room than an empty measure. Rendered, every system
therefore carries a time ruler underneath it, which shows where each onset
really lies.

```python
import audiospylt as asp

events = asp.read_table("data/tables/conga_multidft.tsv")
slow = asp.stretch(events, 4.0)
top5 = slow.sort_values("amp_start", ascending=False).groupby("time_start").head(5)
mei = asp.events_to_mei(top5, time_labels=True)
asp.nb.show_mei(mei)          # in a notebook: the score with its ruler and hover
svg = asp.render_svg(mei)     # the same as an SVG string, e.g. for a figure
```

## The ruler

Under each system, a horizontal line covers the system's time span at true
scale. It starts below the system's first chord at that chord's onset and ends
at the right edge of the staff, at the next system's first onset (or at the
latest `time_stop` for the last system), so consecutive rulers continue one
another, across pages too. Each onset gets a dot on the ruler, and a thin line
runs from below its chord to the dot. Scale ticks are labelled in seconds with
one step (1, 2, 2.5 or 5 × 10ⁿ) per page, the smallest whose labels do not
collide on any of the page's systems.

The engraving itself is unchanged; Verovio only leaves more room between
systems (`spacingSystem` grows by 16 MEI units) and the page grows at the
bottom. `render_svg(..., timeline=False)` gives the score without the ruler.
`timeline=None` (the default) draws it whenever the MEI carries onset times,
`timeline=True` raises a `ValueError` if it does not (snapshots have none).

## Colours and hover

`events_to_mei(..., colors=True)` colours each onset's chords, cycling through
six colours of the `viz` light theme; the ruler uses the same colour for that
onset's line and dot. A list of `'#rrggbb'` or `'#rrggbbaa'` strings sets your
own cycle. Colours are part of the MEI (`chord@color`), so other MEI viewers
show them too.

In a notebook, `nb.show_mei` adds hover (CSS only, like the cents labels):

| under the mouse | coloured |
| --- | --- |
| a notehead | the note, its cents label, the onset label, the onset's line and dot |
| a dot, its line or an onset label | every note of the onset, their cents labels, the onset label, the line and the dot |

`highlight=None` turns hover off.

## How the times are stored in MEI

The onset times are part of the MEI document, using MEI's own timeline, so
they survive export and other software can read them:

```xml
<music>
  <performance>
    <recording begin="00:00:00" end="00:00:02.91" betype="time">
      <when xml:id="w1" absolute="00:00:00.03" />
      <when xml:id="w2" absolute="00:00:00.31" />
    </recording>
  </performance>
  <body> … <measure n="1" xml:id="m1" when="#w1"> … </measure>
         <measure n="2" xml:id="m2" when="#w2"> … </measure> … </body>
</music>
```

- `recording@end` is the latest `time_stop` of the table; `begin` is 0.
- One `<when>` per onset, in ISO time (`HH:MM:SS.ffffff`), rounded to whole
  microseconds. Times of 24 h or more raise a `ValueError` (the format stops
  at 23:59:59.999999).
- The measure that holds an onset's chord points at its `<when>` with
  `@when`; padding measures have none. All measures have ids `m1`, `m2`, …

`render_svg` draws a ruler for any MEI document whose measures point this way
at `<when absolute>` elements in ISO time (in a `<recording betype="time">` or
with `abstype="time"`), not only for documents written by `events_to_mei`.
