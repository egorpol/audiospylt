# Partiels PM2 exports (test fixtures)

Cuts of real Partiels exports for `tests/test_interop_partiels.py`. Made 2026-10-23 with Partiels 2.6.0 and
Ircam Vamp Plugins 2.1.0 (PM2 5) from the Ircam templates `PM2-Partials-Tracking-Harmonic-20.ptldoc` and
`PM2-Partials-Tracking-Inharmonic-20.ptldoc` (release asset `Ircam-Vamp-Plugins-Manuals.zip`, template
parameters unchanged):

    Partiels --new -i SOUND.wav -t TEMPLATE.ptldoc -o DOC.ptldoc
    Partiels --export -d DOC.ptldoc -o EXPORT/csv -f csv --header=specific
    Partiels --export -d DOC.ptldoc -o EXPORT/json -f json --description

| Folder | Sound | Template | Files | Frames | Channels |
|---|---|---|---|---|---|
| `flute_csv/`, `flute_json/` | `samples/saariaho_flute.wav` (stereo) | Harmonic | `Partial_1`, `_2`, `_3`, `_20` (PM2 partial ID 19) and the SuperVP f0 track `Feature_Scoring` | 140-179 (1.097-1.400 s) | both |
| `cello_json/` | `samples/cello_arco_d_sul_c_01.wav` (dual mono) | Inharmonic | `Partial_8`, `_9`, `_10`, `_12`, `_15`, `_18` | 570-609 (3.320-3.547 s) | 0 |

`trim.py` cut them from the full exports (`python tests/data/partiels/trim.py FLUTE_EXPORT CELLO_EXPORT`);
frames, values and the JSON `track` descriptions are as exported. Harmonic PM2 is deterministic, so the
flute CSV and JSON come from identical analyses. Inharmonic PM2 in Partiels is not: a new export of the
cello differs, and its slots hold copies of each other's points (the reason for `dedupe`).
