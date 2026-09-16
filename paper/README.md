# ZKFly paper

The paper is intentionally claim-bounded: full MaleCNS numeric execution is
measured, the complete circuit size is estimated from implemented relations,
and only the small fixed circuit is reported as an end-to-end Nova proof.

## Files

- `main.tex`: three-page, two-column manuscript.
- `references.bib`: primary cryptography and MaleCNS references.
- `../output/pdf/zkfly-paper.pdf`: generated submission PDF.

The manuscript separates four categories: proved relations, measured fixture
performance, full-artifact capacity estimates, and explicit non-claims. In
particular, the 4.770 ms full-graph CUDA execution result is not presented as
proof latency, and the current implementation does not claim a distributed or
full MaleCNS proof.

## Build

Build from this directory:

```sh
mkdir -p ../tmp/pdfs/zkfly-paper ../output/pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../tmp/pdfs/zkfly-paper main.tex
cp ../tmp/pdfs/zkfly-paper/main.pdf ../output/pdf/zkfly-paper.pdf
```

Required LaTeX packages include `amsmath`, `booktabs`, `algorithmicx`,
`pgfplots`, `natbib`, and `cleveref`. The source tables use the checked JSON
receipts under `../receipts/`; generated files under `../tmp/` and
`../output/` are build artifacts rather than manuscript sources.

## Validation

After compilation, check that the log has no undefined citations/references or
overfull boxes, render all pages with Poppler, and inspect every rendered page.
The capacity figures are covered by the
`estimates_registered_topology_and_one_thousand_ticks` unit test.
