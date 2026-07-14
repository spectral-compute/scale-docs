# How to Cite SCALE

!!! danger "DRAFT - do not deploy"

    This page is a draft. The values below are *proposals*, pending answers from
    the team. Each open decision is marked with a `TODO(cite-N)` comment in the
    Markdown source, numbered to match the questions asked internally:

    1. **Which paper is primary?** Proposed: the HCDS '25 paper.
    2. **Software DOI?** Proposed: none - we have paper DOIs, but SCALE itself is
       not archived (no Zenodo record).
    3. **Software author?** Proposed: `Spectral Compute Ltd` as a corporate author.
    4. **Title wording?** Proposed: "SCALE: a CUDA-compatible GPU programming toolkit".
    5. **Canonical URL?** Proposed: `https://scale-lang.com`.
    6. **Version/year policy?** Proposed: pin the current release, bump by hand.

    Remove this admonition, resolve every `TODO(cite-N)`, and re-enable the nav
    entry in `mkdocs.yml` before this page ships.

If SCALE contributed to work you are publishing - a paper, a preprint, a poster,
a thesis, a technical report, or the artifact description that accompanies them -
please cite it. Citations help other researchers reproduce your results,
and they help us justify continued investment in SCALE.

As a rule of thumb, cite SCALE whenever you would have cited a compiler or a GPU
runtime: if your results were produced by CUDA code that SCALE compiled or ran,
it belongs in your references. Mentioning the *version* you used matters too,
since compiler and library behaviour changes between releases.

## Which reference should I use?

There are two kinds of reference on this page, and they are not interchangeable:

- **The papers** describe how SCALE works. Cite one of these when you discuss
  SCALE's approach, compare against it, or credit the ideas behind it.
- **The software** is what actually produced your numbers. Cite this - with the
  version - when SCALE compiled or ran the code behind your results.

If SCALE both inspired a point you make *and* produced your results, cite both.
That is the usual case for an experimental paper.

## Citing the papers

<!-- TODO(cite-1): Confirm which paper leads. Proposed: HCDS '25, on the grounds
     that it is the fuller (7pp vs 2pp) and more recent of the two. If a newer
     paper, a preprint, or an arXiv version should be primary instead, swap the
     order of the two entries below and update this sentence. -->

The most complete description of SCALE is the HCDS '25 paper. Prefer it unless
you specifically need the earlier one:

{% raw %}
```bibtex
@inproceedings{pavlidakis2025crossvendor,
  author    = {Pavlidakis, Manos and Kitching, Chris and Tomlinson, Nicholas
               and S{\o}ndergaard, Michael},
  title     = {Cross-Vendor {GPU} Programming: Extending {CUDA} Beyond {NVIDIA}},
  booktitle = {Proceedings of the 4th Workshop on Heterogeneous Composable
               and Disaggregated Systems (HCDS '25)},
  publisher = {ACM},
  address   = {New York, NY, USA},
  year      = {2025},
  month     = mar,
  pages     = {45--51},
  doi       = {10.1145/3723851.3723860},
}
```
{% endraw %}

The earlier Middleware '24 demo paper is a shorter, two-page introduction to
ahead-of-time compilation of CUDA for AMD GPUs:

{% raw %}
```bibtex
@inproceedings{pavlidakis2024scale,
  author    = {Pavlidakis, Manos and Kitching, Chris and Tomlinson, Nicholas
               and S{\o}ndergaard, Michael},
  title     = {{SCALE}-Ahead-Of-Time Compilation of {CUDA} for {AMD} {GPU}s},
  booktitle = {Proceedings of the 25th International Middleware Conference:
               Demos, Posters and Doctoral Symposium (Middleware '24)},
  publisher = {ACM},
  address   = {New York, NY, USA},
  year      = {2024},
  month     = dec,
  pages     = {5--6},
  doi       = {10.1145/3704440.3704782},
}
```
{% endraw %}

Both entries were generated from the publishers' Crossref metadata. The page
ranges use LaTeX `--` rather than the en-dashes in ACM's own BibTeX export, and
`S{\o}ndergaard` is escaped so the entries work in bibliographies that are not
compiled as UTF-8. If you use BibLaTeX with `biber`, you can safely write
`Søndergaard` instead.

The `doi` field requires a `doi = {...}`-aware bibliography style (most modern
ACM, IEEE and BibLaTeX styles are). If yours ignores it, add the resolver link
by hand: <https://doi.org/10.1145/3723851.3723860>.

## Citing the software

<!-- TODO(cite-2): Does SCALE get its own software DOI (e.g. a Zenodo deposit,
     ideally auto-archiving each release)? The sentence below asserts that it
     does not. If we mint one, add a `doi` field to BOTH entries below, and
     rewrite this sentence - a software DOI is the thing reviewers increasingly
     ask for, so it should be stated prominently rather than buried. -->

SCALE itself is not archived under its own DOI, so cite it as software, naming
the version you used.

<!-- TODO(cite-3): Confirm the author string. Proposed: the company as a
     corporate author, rather than a list of individuals (which we would then
     have to maintain). Also settle "Spectral Compute Ltd" vs "Spectral Compute".
     Whatever we pick must stay double-braced - see the note below. -->
<!-- TODO(cite-4): Confirm the title wording. Proposed text is adapted from the
     docs homepage. -->
<!-- TODO(cite-5): Confirm the canonical URL. Proposed: the marketing site
     rather than the docs subdomain. -->
<!-- TODO(cite-6): Confirm version/year. Currently pinned to 1.7.1 (2026-06-01),
     bumped by hand at each release. If we would rather it tracked releases
     automatically, replace the literals with a macros variable (see main.py)
     so the page cannot go stale. -->

Use the `@software` entry if your bibliography is processed by BibLaTeX, which is
the case for most modern LaTeX templates:

{% raw %}
```bibtex
@software{scale,
  author  = {{Spectral Compute Ltd}},
  title   = {{SCALE}: a {CUDA}-compatible {GPU} programming toolkit},
  version = {1.7.1},
  date    = {2026-06-01},
  url     = {https://scale-lang.com},
}
```
{% endraw %}

If you are using legacy BibTeX (for example, an older conference template that
still runs `bibtex` rather than `biber`), `@software` is not a recognised entry
type. Use this instead:

{% raw %}
```bibtex
@misc{scale,
  author       = {{Spectral Compute Ltd}},
  title        = {{SCALE}: a {CUDA}-compatible {GPU} programming toolkit},
  year         = {2026},
  note         = {Version 1.7.1},
  howpublished = {\url{https://scale-lang.com}},
}
```
{% endraw %}

The braces around `{% raw %}{{Spectral Compute Ltd}}{% endraw %}` tell BibTeX that the author is an
organisation rather than a person, and stop it from being reformatted as
"Ltd, S. C.". The inner braces in the title preserve the capitalisation of
`SCALE`, `CUDA` and `GPU` in styles that would otherwise lowercase them.

`\url{}` requires the `url` or `hyperref` package. If your template loads
neither, replace it with the bare address.

## Plain text

For bibliography styles you fill in by hand, or for acknowledgements sections.

The paper:

```text
Manos Pavlidakis, Chris Kitching, Nicholas Tomlinson, and Michael Søndergaard.
Cross-Vendor GPU Programming: Extending CUDA Beyond NVIDIA. In Proceedings of
the 4th Workshop on Heterogeneous Composable and Disaggregated Systems (HCDS
'25), pages 45-51. ACM, 2025. https://doi.org/10.1145/3723851.3723860
```

The software:

```text
Spectral Compute Ltd. SCALE: a CUDA-compatible GPU programming toolkit,
version 1.7.1, 2026. https://scale-lang.com
```

An in-text mention might read:

```text
CUDA sources were compiled for AMD GPUs with SCALE 1.7.1 (Spectral Compute Ltd).
```

## Citing a specific version

The software entries above name the release that was current when this page was
last updated. If you used a different one, change `version` (and the `date` or `year`)
to match what you actually ran, so that others can reproduce your results against
the same toolchain.

You can print the version of an installed SCALE with:

```shell
nvcc --version
```

Release dates for every version are listed in the
[changelog](./manual/changelogs/whats-new.md).

## Tell us about your work

We would love to hear what you built. If you publish something that uses SCALE,
tell us on [Discord](./contact/join-our-discord.md) or email
[hello@spectralcompute.com](mailto:hello@spectralcompute.com).

Note that citing SCALE does not imply that we endorse your work, and that SCALE
is a registered trademark of Spectral Compute Ltd - see
[Use of Trademarks](./use_of_trademarks.md).
