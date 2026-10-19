# Example tables

Small tab-separated tables read by the notebooks in `tutorials_tech/`. They use the column names of
AudioSpylt 0.6 (the v1 format and the upgrade rules for these files are specified in
[`docs/table-format.md`](../../docs/table-format.md)):

- **peak list** (peaks of one DFT frame): `Frequency (Hz)`, `Amplitude`
- **event table** (input for additive resynthesis): `freq_start`, `freq_stop`, `time_start`, `time_stop`,
  `amp_min`, `amp_max`, where `amp_min` and `amp_max` are the amplitudes at the start and at the stop of each event

| File | Layout | Rows | Used by |
|---|---|---|---|
| `speech1.tsv` | peak list | 41 | `2df_merge`, `symbolic_mei`, `visual_tsv` |
| `conga3.tsv` | peak list | 10 | `2df_merge`, `symbolic_mei`, `visual_tsv` |
| `bayle1.tsv` | peak list | 64 | `df_pitch_stretch` |
| `cello_single.tsv` | peak list | 14 | `symbolic_mei` |
| `conga_multidft.tsv` | event table | 87 | `df_pitch_stretch`, `resynth` |
| `conga_multidft2_dropped.tsv` | event table | 70 | `symbolic_mei` |
| `sigmoid_high.tsv` | event table | 41 | `visual_tsv` |
| `sigmoid_low.tsv` | event table | 41 | `visual_tsv` |
