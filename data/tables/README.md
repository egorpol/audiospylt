# Example tables

Small tab-separated tables read by the notebooks in `tutorials_tech/`, in the v1 table format specified in
[`docs/table-format.md`](../../docs/table-format.md). The first line names the kind
(`# audiospylt-table: snapshot v1` or `# audiospylt-table: events v1`), the next the columns:

- **snapshot** (the peaks of one DFT frame): `freq`, `amp`
- **events** (input for additive resynthesis): `time_start`, `time_stop`, `freq_start`, `freq_stop`,
  `amp_start`, `amp_stop`, where `amp_start` and `amp_stop` are the amplitudes at the start and at the stop of
  each event

| File | Kind | Rows | Used by |
|---|---|---|---|
| `speech1.tsv` | snapshot | 41 | `2df_merge`, `symbolic_mei`, `visual_tsv` |
| `conga3.tsv` | snapshot | 10 | `2df_merge`, `symbolic_mei`, `visual_tsv` |
| `bayle1.tsv` | snapshot | 64 | `df_pitch_stretch` |
| `cello_single.tsv` | snapshot | 14 | `symbolic_mei` |
| `conga_multidft.tsv` | events | 87 | `df_pitch_stretch`, `resynth` |
| `conga_multidft2_dropped.tsv` | events | 70 | `symbolic_mei` |
| `sigmoid_high.tsv` | events | 41 | `visual_tsv` |
| `sigmoid_low.tsv` | events | 41 | `visual_tsv` |

These files were written by AudioSpylt 0.6 and upgraded with `write_table(read_table(old), new)`; the values are
unchanged. The 0.6 originals (no header line, `Frequency (Hz)` / `Amplitude` and `amp_min` / `amp_max` column
names) are kept as test fixtures in [`tests/data/legacy_tables/`](../../tests/data/legacy_tables/).
