# Self-hosted fonts

Two families, self-hosted instead of loaded from Google Fonts (see the
"fonts" comment block in `../custom.css` for why):

- **Inter** (body copy) — `inter-latin.woff2`, `inter-latin-ext.woff2`
- **Sora** (headings) — `sora-latin.woff2`, `sora-latin-ext.woff2`

Both are variable-font woff2 files covering weights 400-800 (Inter) /
600-800 (Sora) in a single file per subset, extracted from Google Fonts'
`css2` API (`https://fonts.googleapis.com/css2?family=Inter:wght@...&family=Sora:wght@...`)
with a modern-Chrome user agent, keeping only the `latin` and `latin-ext`
subsets (this is an English-language technical docs site; the other
subsets — cyrillic, greek, vietnamese, etc. — aren't needed).

Both are licensed under the SIL Open Font License 1.1 (full text in
`Inter-OFL.txt` / `Sora-OFL.txt`, bundled per the license's redistribution
requirement).

To update to a newer version, re-fetch the same URL, extract the `latin`/
`latin-ext` `@font-face` blocks' `src` URLs (unicode-range starting
`U+0000-00FF` and `U+0100-02BA` respectively), and re-download.
