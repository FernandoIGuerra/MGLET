# Manuscript

LaTeX template for the compound-channel journal article. Two columns, Times,
journal-neutral so it compiles on any TeX Live.

```bash
./build.sh          # build main.pdf
./build.sh watch    # rebuild on save
./build.sh todo     # count the \TODO left, per file
./build.sh clean
```

## Layout

| file | contents |
|---|---|
| `main.tex` | document class, front matter, section order |
| `preamble.tex` | packages, page style, and **all notation macros** |
| `sections/` | one file per section, in the order they appear |
| `refs.bib` | bibliography (natbib author–year, `plainnat`) |
| `figures/` | figures; `\graphicspath` already points here |

Section files map to the structure:

1. `01_introduction.tex`
2. `02_configuration.tex` — flow configuration and LES
3. `03_spod.tex`
4. `04_linear_resolvent.tex` — theory, with **4.3 Solver**
5. `05_results.tex` — numerical results, validation first
6. `06_conclusions.tex`, `A1_appendix.tex`

## Conventions

- **Notation lives in `preamble.tex`.** Change a symbol there, not in the text.
  `\Retau`, `\utau`, `\Dr`, `\spodmode`, `\Rop`, `\gaink` and the rest are
  already defined.
- **`\TODO{...}`** marks what is unwritten; it renders as grey italics. Set
  `\draftfalse` in `preamble.tex` at submission — that hides every `\TODO` and
  drops the line numbers. Get `./build.sh todo` to zero first.
- Columns are narrow: keep display equations short, use `split`/`multline`, and
  put anything wider than a column in `figure*`/`table*`.
- British spelling throughout (JFM/JHR house style).

## Switching to a publisher class

Replace the `\documentclass` line and the front-matter block in `main.tex`;
nothing in `sections/` changes. Candidates are listed in the header comment
(`elsarticle`, `jfm`, `revtex4-2`) — none is installed here, so install the
class before switching.

Turn off `\linenumbers` at submission.
