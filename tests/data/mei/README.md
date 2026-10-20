# MEI 5.1 schema (vendored for tests)

`mei-all.rng` is the RELAX NG schema of the Music Encoding Initiative (MEI) 5.1, unchanged.
The tests validate every MEI document that `audiospylt.notation` writes against it. It is not
part of the installed package.

- Source: `MEI_Schemata_v5.1.zip` from <https://github.com/music-encoding/music-encoding/releases/tag/v5.1>
  (asset URL `https://github.com/music-encoding/music-encoding/releases/download/v5.1/MEI_Schemata_v5.1.zip`),
  generated 2025-01-22.
- SHA-256 of the zip: `2586509c350d67b16bdeaae8e8e33e626dee615df88d49c87058484bb6334877`
- SHA-256 of `mei-all.rng`: `24f0d6e53eebdaf6e85f91f38fccc479bf91a200e15177d5284070cafd3b29bc`
- Licence: Educational Community License 2.0, text in `LICENSE-ECL-2.0.txt`
  (from the SPDX licence list, <https://github.com/spdx/license-list-data>).

Notice from the MEI 5.1 source:

> NOTICE: Copyright (c) by the Music Encoding Initiative (MEI) Board (formerly known as "MEI Council").
>
> Licensed under the Educational Community License, Version 2.0 (the "License"); you may
> not use this file except in compliance with the License. You may obtain a copy of the License
> at https://opensource.org/licenses/ECL-2.0.
>
> Unless required by applicable law or agreed to in writing, software distributed under the
> License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
> OF ANY KIND, either express or implied. See the License for the specific language
> governing permissions and limitations under the License.
>
> This is a derivative work based on earlier versions of the schema © 2001-2006 Perry Roland
> and the Rector and Visitors of the University of Virginia; licensed under the Educational
> Community License version 1.0.

Validate a file by hand: `xmllint --noout --relaxng tests/data/mei/mei-all.rng FILE.mei`
