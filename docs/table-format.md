# AudioSpylt table interchange format, version 1

This is the public interchange contract for AudioSpylt and sibling tools such as
FFTimbre. A table is a collection of rows of one declared kind: `snapshot`,
`events`, or `tracks`. The major version in the file is the **format version**,
independent of the installed AudioSpylt package version. Writers emit `v1`;
readers must reject every other version and report the version they found. New
incompatible interpretations require a new format major version.

## File structure

Writers MUST emit UTF-8 without a byte-order mark (BOM), LF (`0A`) record endings,
a final LF, the tab character (`09`) as the field separator, and no DataFrame
index column. Writers MUST NOT emit CRLF record endings, a leading BOM, or extra
empty records at EOF. CR and LF inside quoted cells remain cell content.

Readers MUST accept LF or CRLF (`0D 0A`) record endings, including on the type
line and metadata lines, an optional single leading UTF-8 BOM (`EF BB BF`), and
zero or more empty physical lines at EOF. Mixed LF/CRLF endings are accepted.
These tolerances apply to both versioned and headerless files, and to paths and
text streams. They do not permit interior empty records or alter CR/LF embedded
in quoted cells. Here an empty line contains no characters before its line
ending; a line containing spaces or tabs is not an empty line. The structure is:

```text
# audiospylt-table: <kind> v1
# <key>: <value>
<column names separated by tabs>
<data fields separated by tabs>
...
```

The first physical line is mandatory and is exactly
`# audiospylt-table: <kind> v<major>`: single ASCII spaces, no indentation,
trailing whitespace, or additional tokens. `<kind>` is one of the three
case-sensitive tokens above. For this version, `<major>` is exactly `1` (neither
`01` nor `1.0`). Unknown kind tokens and malformed type lines are errors.

Zero or more metadata lines follow immediately. Each is exactly
`# <key>: <value>`, with one ASCII space after `#` and after `:`. Keys match the
entire regular expression `[a-z][a-z0-9_]*` and must be unique. All keys are
application-defined; there are no required metadata keys. Values are strings
without LF, CR, or tab. An empty string is allowed: the line then ends with the
space after the colon. Strip exactly that one separator space after the colon;
preserve all remaining leading/trailing spaces, punctuation, and Unicode in the
value. The CR in a CRLF line ending is not part of the metadata value; metadata
values never end with a retained CR. Metadata is not CSV-quoted or escaped. Its iteration order is preserved
by the Python writer; its order has no semantic meaning.

The next logical record is the column-name row; all subsequent records are data.
There is exactly one column-name row, even for an empty table. Column labels are
case-sensitive and must be unique. Each data record must have exactly the same
number of cells as this row. Interior empty records and arbitrary comment
records are errors; only empty lines at EOF are ignored by readers. Metadata
cannot appear after the column-name row. There is no general
comment syntax or inline-comment stripping: `#` inside a cell is ordinary data,
including at the beginning of a cell. An extra column name may also contain `#`.
During preamble parsing, a physical line beginning with
`# [a-z][a-z0-9_]*: ` is always a metadata line: a tab anywhere in its value is a
metadata error, not a column-name row. Other `#` lines containing a tab begin
the column-name row, allowing a first column name such as `# label`.

Column names and data use CSV-style quoting with tab as delimiter: a field
containing a tab, LF, CR, or double quote is enclosed in double quotes (`"`), and an
embedded double quote is doubled (`""`). Other fields need not be quoted. There
is no backslash escape. A quoted field may span physical lines; its embedded LF
is part of the value; quoted CR is also preserved as cell content. For example,
a cell containing `a`, a tab, and `"b"` is
encoded as `"a<TAB>""b"""`, where `<TAB>` here denotes an actual tab. Parse logical
records with these quoting rules before checking field counts. Metadata and the
type line are physical lines outside this quoted TSV payload.

## Kinds and canonical columns

The order below is the canonical order: all required columns, then any present
optional columns in the listed order, then extras in their original relative
order. Optional columns may be omitted entirely. Writers normalize to this
order. Readers identify columns by name and normalize their output; reordering
input columns does not change their meaning. An extra column cannot substitute
for a required column. Names such as `amplitude` do not mean `amp`.

### `snapshot`

Peaks from a single analysis frame:

- Required: `freq`, `amp`.
- Optional: `phase`, `bin`, `prominence`.
- `freq`: frequency in Hz, non-negative.
- `amp`: linear amplitude, non-negative.
- `phase`: phase in radians, unrestricted sign.
- `bin`: non-negative integer analysis-bin index, or unknown.
- `prominence`: non-negative peak prominence, or unknown; this format assigns
  no additional analysis-specific scaling convention to it.

### `events`

Additive-synthesis input. Every row describes a linear ramp from the start
frequency/amplitude to the stop frequency/amplitude over its time interval:

- Required: `time_start`, `time_stop`, `freq_start`, `freq_stop`, `amp_start`,
  `amp_stop`, in that order.
- No optional columns.
- Times are in seconds; frequencies are in Hz; amplitudes are linear.
- Every value is non-negative, and `time_stop >= time_start` in each row.
- A zero-length row (`time_stop == time_start`) is valid and is a synthesis
  no-op, including an all-zero row.
- Amplitudes may rise, fall, or remain constant. There is no requirement that
  `amp_start <= amp_stop`, nor any row-sorting requirement.

Sample-level endpoint conventions belong to the synthesis API, not this table
format. The format does not impose an upper frequency limit or a sample rate.

### `tracks`

Partial tracks, including targets of future Partiels/SPEAR/SDIF interchange:

- Required: `track_id`, `time`, `freq`, `amp`.
- Optional: `phase`.
- `track_id` is a signed 64-bit integer identifier, including negative IDs:
  `-9223372036854775808 <= track_id <= 9223372036854775807`.
- `time` is in seconds, `freq` in Hz, and `amp` is linear amplitude; all three
  are non-negative. `phase` is in radians with unrestricted sign.
- Time must be **strictly increasing within each track in row order**. A
  repeated or decreasing time in one track is an error; readers must not sort
  the rows to repair it. Rows of different tracks may interleave, and a track
  may have just one row.

Zero Hz is valid in every frequency column. Amplitude scaling is intentionally
not fixed in v1's table schema; do not assume a specific measurement convention.

## Numbers, missing values, and exact round trips

All required cells must contain real numeric values: booleans, missing values,
NaN, and positive/negative infinity are errors. Optional cells may be unknown,
encoded by an empty cell, but must otherwise contain real numeric values;
booleans and infinities are errors. Empty `bin` is a missing integer, not zero.
Zero rows are valid whenever all required column names are present.

Float columns are IEEE 754 binary64 (`float64`). Use decimal point `.` and a
base-10 exponent if needed (`e` or `E`, optional exponent sign), without digit
grouping or a locale-specific decimal separator. Examples: `0`, `-0.0`, `1.5`,
`1e-300`, `0.30000000000000004`, `1.0000000000000002`,
`71.42857142857142`. Parse to binary64 with correct rounding to nearest,
ties-to-even. Writers must emit enough digits to recover the identical binary64
bit pattern, including signed zero and subnormal values. A shortest
round-trippable decimal representation (Python `repr(float(value))`) or 17
significant decimal digits with a correctly rounded parser meets this rule.
Fixed decimal precision such as six digits does not. Optional NaN is written as
an empty cell; its NaN payload/sign is not part of the interchange contract.
Readers may also recognize textual NaN in optional numeric cells as unknown;
writers always emit the empty representation.

`track_id` and `bin` must be integer-valued. Integer-valued decimal/exponent
representations such as `2.0` or `2e0` are accepted, but writers emit decimal
integers without a fractional part. Parse these columns without routing through
binary64, which would round IDs above `2**53`. Both columns must fit signed
int64; `bin` additionally requires `0 <= bin <= 9223372036854775807` because its
in-memory representation is pandas nullable `Int64`.

Extras are untyped extension columns. Their labels, relative order, and cell
contents are passed through by validation; their values do not satisfy or
override any required/optional field. The file has no dtype declarations for
extras. The Python reader infers their types (numeric and boolean where
possible, otherwise text); only an empty cell is treated as missing in v1 extra
columns, so text such as `NA` remains text. Arbitrary Python objects, categorical
types, all-text numeric columns, and extra-column dtypes in an empty file are
not guaranteed to round trip as the original Python dtype. For reliable
interchange, use extra columns containing plain text or numeric scalars. Numeric
float64 extras use the same exact float printing/parsing as canonical columns.
Only the recognized schema columns carry portable units and dtype semantics.

## Python API and validation

Use absolute imports:

```python
from audiospylt.model.schema import infer_kind, upgrade, validate
from audiospylt.io.tables import read_table, write_table
```

`TABLE_CONTRACTS` maps kind tokens to frozen `TableContract` objects. Each has
`kind`, `required`, and `optional`; the latter two are ordered tuples of frozen
`ColumnContract` objects with `name`, `dtype`, and `nonnegative` fields.

Tables in memory are plain pandas DataFrames. `validate(df, kind)` returns a
copy in canonical column order, with a fresh zero-based RangeIndex, float64
float columns, int64 `track_id`, and nullable `Int64` `bin`. It preserves extras
and unrelated attrs, and sets `df.attrs["audiospylt_table"]` on the result to the
validated kind. Numeric in-memory cells must be numbers (not numeric strings
or booleans). Integral floats in integer columns are accepted. Validation raises
one `SchemaError(ValueError)` listing every detected problem across usable
columns, including counts and first offending zero-based row positions. A
duplicate label is an error; its duplicated column cannot be value-validated
unambiguously. Validation neither sorts rows nor mutates the input. Numeric
NumPy/pandas dtypes, including nullable numeric dtypes, use vectorised checks;
object columns retain scalar checks to preserve exact numbers and reject
booleans among mixed values. Within-track time checks are also vectorised.

`infer_kind(df)` uses required column sets, ignoring attrs. Complete `tracks`
columns take precedence over `snapshot` because tracks contain `freq` and
`amp`; `track_id` plus `time` distinguish them. Independent matches (for example
complete `events` and `snapshot` columns together) or no complete match raise
`SchemaError`.

`write_table(df, path, kind=None, *, meta=None) -> Path` selects the explicit
kind, otherwise `attrs["audiospylt_table"]`, otherwise inference, and always
validates. Metadata defaults to `df.attrs.get("meta", {})`; an explicit mapping
overrides it, including an empty mapping. Metadata values are converted with
`str()`, then checked for prohibited characters. Table validation, metadata
validation, and TSV serialization finish before the destination is opened;
invalid input leaves a missing or existing destination untouched. No parent
directories are created. The function returns the supplied path as a `Path` and
does not modify the source frame. An OS error during writing may leave a partial
file; the writer does not promise an atomic filesystem transaction.

`read_table(path_or_buffer, kind=None) -> DataFrame` accepts a string path,
`os.PathLike`, or a text file object. It opens paths as UTF-8. A caller-owned
stream is read from its current position and is left open; binary streams are
rejected. Open caller-owned text streams with `newline=""` to preserve embedded
CR/LF characters verbatim (default universal-newline translation happens before
the reader receives the text). For v1, the kind comes from the type line, and a
disagreeing explicit kind is an error. The output is validated and always has
`attrs["audiospylt_table"]` and `attrs["meta"]`; the latter is a `dict[str, str]`,
empty when there is no metadata. Other attrs are not stored in files. A table's
attrs are only hints because frames are mutable and pandas operations may drop
attrs; table-receiving APIs validate rather than trusting a hint. No new table
API prints, displays, or writes another file.

## Headerless 0.6 compatibility

Here “headerless” means without the version/type line, **not** without the
column-name row. The following are the only legacy renames:

- `Frequency (Hz)` → `freq`.
- `Amplitude` → `amp`.
- `amp_min` → `amp_start`.
- `amp_max` → `amp_stop`.

The 0.6 event time/frequency names already match v1. In 0.6, `amp_min` and
`amp_max` meant the start and stop amplitude of a ramp, respectively. Never
compare, swap, take a minimum/maximum, or reorder these values during upgrade:
decaying ramps are normal.

If the first line has no version marker, the Python reader parses the TSV,
applies the renames, infers the kind (unless explicitly supplied), and validates.
Both headerless and v1 files use pandas' native float parser with
`float_precision="round_trip"`. Headerless fixture values therefore match
`pd.read_csv(path, sep="\t", float_precision="round_trip")` exactly, rather than
the less precise default parser. Integer columns retain decimal-token parsing
because nullable bins and decimal/exponent identifiers can otherwise round
through binary64. Non-numeric cells cause their column to remain text in pandas;
these exceptional float columns are parsed cell by cell before validation so
valid numeric tokens and invalid tokens are distinguished in aggregated errors.
A malformed version/type line is an error, not a request for legacy fallback.
Any occurrence of both a legacy name and its canonical counterpart is ambiguous
and raises `SchemaError`, even if one would otherwise be considered an extra.
Versioned v1 files must already use canonical names; they are never upgraded.

Each successful read that renames columns emits exactly one
`LegacyFormatWarning`, a `FutureWarning` subclass. The message identifies the
path (a stream's name if available, otherwise `<buffer>`) and every applied
old → new mapping. Its warning location is the caller of `read_table`.
Headerless files already using canonical names load silently, with empty
metadata. `upgrade(df)` performs the same rename and validation for an in-memory
DataFrame; its warning names `DataFrame` and points at the caller of `upgrade`.
It is idempotent: upgrading a canonical, normalized frame preserves its values,
column order, dtypes, index, and attrs without a warning, while returning a copy.

## Complete examples

The separators in these examples are literal tabs. An optional unknown phase
appears as an empty cell between two tabs:

```tsv
# audiospylt-table: snapshot v1
# source: FFTimbre
freq	amp	phase	bin	prominence	label
440.0	0.30000000000000004		10	0.2	fundamental
```

A decaying event followed by a valid zero-length no-op:

```tsv
# audiospylt-table: events v1
time_start	time_stop	freq_start	freq_stop	amp_start	amp_stop
0.0	1.0	440.0	442.0	0.8	0.1
1.0	1.0	0.0	0.0	0.0	0.0
```

An empty tracks table without metadata or optional columns:

```tsv
# audiospylt-table: tracks v1
track_id	time	freq	amp
```
