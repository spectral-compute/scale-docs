# Acknowledging or Citing SCALE

If SCALE is part of your application, your product, or the work behind a
publication, we would appreciate a mention. It helps other people discover
SCALE, and it helps us understand how it is being used.

## In Presentations or Apps

If your application, product, demo, or talk uses SCALE, please say so - a
simple mention linking to [scale-lang.com](https://scale-lang.com) is perfect:

```text
Powered by SCALE (https://scale-lang.com)
```

or, in running text:

```text
This application is built with SCALE, a CUDA-compatible GPU programming
toolkit by Spectral Compute (https://scale-lang.com).
```

Our logos and usage guidelines are available at
[scale-lang.com/brand-assets](https://scale-lang.com/brand-assets). If in
doubt, reach out on [Discord](./contact/join-our-discord.md) or at
[hello@spectralcompute.com](mailto:hello@spectralcompute.com).

## In Publications

If SCALE contributed to work you are publishing - a paper, a preprint, a
poster, or a thesis - please acknowledge it. A sentence like this works well,
with a link to [scale-lang.com](https://scale-lang.com) if the venue permits:

```text
This work made use of SCALE, a CUDA-compatible GPU programming toolkit by
Spectral Compute (https://scale-lang.com).
```

Consider naming the SCALE version you used (printed by `nvcc --version`;
release dates are in the [changelog](./manual/changelogs/whats-new.md)), so
that others can reproduce your results against the same toolchain.

For a formal citation, please cite this paper:

```text
Manos Pavlidakis, Chris Kitching, Nicholas Tomlinson, and Michael Søndergaard.
Cross-Vendor GPU Programming: Extending CUDA Beyond NVIDIA. In Proceedings of
the 4th Workshop on Heterogeneous Composable and Disaggregated Systems (HCDS
'25), pages 45-51. ACM, 2025. https://doi.org/10.1145/3723851.3723860
```

Or, in BibTeX:

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

The entry was generated from the publisher's Crossref metadata. `S{\o}ndergaard`
is escaped so it works in bibliographies that are not compiled as UTF-8; if you
use BibLaTeX with `biber`, you can safely write `Søndergaard` instead.

## Tell us about your work

We would love to hear what you built. If you publish or ship something that
uses SCALE, tell us on [Discord](./contact/join-our-discord.md) or email
[hello@spectralcompute.com](mailto:hello@spectralcompute.com).

Note that SCALE is a registered trademark of Spectral Compute Ltd - see
[Use of Trademarks](./use_of_trademarks.md).
