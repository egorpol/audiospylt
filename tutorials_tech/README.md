# Technical tutorials

Twelve notebooks on the AudioSpylt v1 API. Open them in Jupyter or VS Code from
this folder (paths are relative to the notebook). Install with
`pip install "audiospylt[all]"` (`pyin` needs the `[librosa]` extra).

From the repository root, `python scripts/run_notebooks.py` executes the
tutorials headless and reports pass or fail per notebook (`--offline` skips the
one that needs the network; see `python scripts/run_notebooks.py --help`).

1. `waveforms_sampling_dft` — sampling, Nyquist, aliasing vs resampling, window
   and zero-padding, and AM/FM: tones built with NumPy/SciPy.
2. `audio_load_dft_single` — load a viola recording from the network, trim and
   fade it, then compare spectrogram frame sizes, a mel view, and DFT peaks.
3. `visual_tsv` — table kinds under `data/tables/`, snapshot and event plots, a
   0.6 → v1 upgrade, and a schema failure.
4. `df_pitch_stretch` — freeze, split, stretch and transpose `conga_multidft` and
   `bayle1`.
5. `resynth` — synthesise `conga_multidft`, compare phase modes and fades, then
   alias on purpose at a low rate.
6. `search_f0` — `harmonic_f0` on five synthetic peaks and on cello peaks from
   `samples/cello_arco_d_sul_c_01.wav`.
7. `symbolic_mei` — `snapshot_to_mei` and `events_to_mei` on tables under
   `data/tables/`, with SVG pages, cents labels and a coloured time ruler.
8. `2df_merge` — listen to `speech1` and `conga3` as held frames, then pair and
   morph one into the other with equal, power, sigmoid and transport maps (each
   one played), including `duplicates`, non-default warps and curved glides.
9. `pyin_f0` — `pyin` (hop 512 vs 256) and `f0_to_tracks` on the cello arco
   sample; refine vs raw pYIN; sonify the contour.
10. `interop_partiels_sdif` — `read_partiels`, `read_sdif`, `read_chord_seq` and
    `write_sdif` on `data/interop/`, with `to_mono` and PM2 frame markers.
11. `tracks_chords_mei` — `tracks_at` and `tracks_to_chords` (`merge_cents`,
    `min_db`) on `data/interop/saariaho_flute_pm2.sdif`, then MEI next to
    `samples/saariaho_sheet.png`.
12. `partial_tracking` — `spectrogram_peaks`, `link_peaks` and `track_partials`:
    test-tone failures, noise and gaps, `prediction` on crossing and turn-back
    glides, then the Saariaho flute next to PM2, with settings, point agreement
    and resynthesis.
