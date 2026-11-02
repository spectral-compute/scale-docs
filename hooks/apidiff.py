"""
Generate the API coverage pages from the apidiff output in the SCALE source tree.

Each API's `<api>-<version>.diff` files in `apidiff/apidiff_out/diffs` become a page,
`manual/library/apidiff/<api>-api.md`, with a tab per CUDA version. The pages are added to the nav
after `apis.md`, which gets a table of how complete each one is.
"""

import logging
import os
import re

from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.apidiff")

DIFFS_DIR = os.path.join("..", "..", "apidiff", "apidiff_out", "diffs") # Relative to mkdocs.yml.
OUT_DIR = "manual/library/apidiff"
INDEX = OUT_DIR + "/apis.md"
INDEX_MARKER = "<!-- apidiff-index -->"


class Api:
    def __init__(self, title, scale_side, nv_side, versioned = True):
        self.title = title
        self.scale_side = scale_side # What SCALE provides it with.
        self.nv_side = nv_side # What it's diffed against.
        self.versioned = versioned # Whether NVIDIA's docs are versioned with CUDA.


# In nav order. APIs not listed here are appended under their own name.
APIS = {
    "runtime": Api("Runtime API", "the SCALE runtime library", "the CUDA Runtime API"),
    "driver": Api("Driver API", "the SCALE runtime library", "the CUDA Driver API"),
    "math": Api("Math API", "the SCALE device library", "the CUDA Math API"),
    "nvrtc": Api("NVRTC", "SCALE's NVRTC", "NVRTC"),
    "cublas": Api("cuBLAS", "SCALE's cuBLAS", "cuBLAS"),
    "cufft": Api("cuFFT", "SCALE's cuFFT", "cuFFT"),
    "curand": Api("cuRAND", "SCALE's cuRAND", "cuRAND"),
    "cusolver": Api("cuSOLVER", "SCALE's cuSOLVER", "cuSOLVER"),
    "cusparse": Api("cuSPARSE", "SCALE's cuSPARSE", "cuSPARSE"),
    "nvml": Api("NVML", "SCALE's NVML", "the NVML API", versioned = False),
}


def version_key(v):
    return tuple(int(x) for x in v.split("."))


class Diff:
    def __init__(self, path):
        with open(path) as f:
            self.body = f.read()

        # The declarations listed: those the diff marks missing, and the rest. An enum's or struct's
        # members are indented under it, and aren't counted.
        self.present = self.missing = 0
        in_block = False
        for line in self.body.splitlines():
            if line.startswith("```"):
                in_block = not in_block
            elif in_block and line[1:5] != "    " and line[1:].strip():
                if line[0] == "-":
                    self.missing += 1
                else:
                    self.present += 1

    def percent(self):
        return 100 * self.present / max(self.present + self.missing, 1)

    def meter(self, label = True):
        p = self.percent()
        text = f"{self.present} of {self.present + self.missing} declarations available"
        bar = f'<span class="apidiff-meter" role="meter" aria-valuenow="{p:.0f}" aria-valuemin="0" ' \
              f'aria-valuemax="100" title="{text}"><span style="width: {p:.1f}%"></span></span>'
        return f'{bar} {p:.0f}%' + (f" ({text})" if label else "")


# {api: {version: Diff}}, newest version first.
diffs = {}


def page_uri(api):
    return f"{OUT_DIR}/{api}-api.md"


def on_config(config):
    diffs.clear()
    diffs_dir = os.path.normpath(os.path.join(os.path.dirname(config.config_file_path), DIFFS_DIR))
    if not os.path.isdir(diffs_dir):
        log.warning(f"No apidiff output at {diffs_dir}: omitting the API coverage pages")
        return config

    found = {}
    for f in os.listdir(diffs_dir):
        api, _, version = f.removesuffix(".diff").rpartition("-")
        if f.endswith(".diff") and api:
            found.setdefault(api, {})[version] = os.path.join(diffs_dir, f)
    for api in sorted(found, key = lambda a: list(APIS).index(a) if a in APIS else len(APIS)):
        APIS.setdefault(api, Api(api, "SCALE", api))
        versions = sorted(found[api], key = version_key, reverse = True)
        if not APIS[api].versioned:
            versions = versions[:1]
        diffs[api] = {v: Diff(found[api][v]) for v in versions}

    # Splice the pages into the nav after the index.
    def find(nav):
        for i, item in enumerate(nav):
            if item == INDEX:
                return nav, i
            for child in (item.values() if isinstance(item, dict) else []):
                if isinstance(child, list) and (r := find(child)):
                    return r
        return None
    if not (r := find(config.nav or [])):
        log.warning(f"{INDEX} is not in the nav: omitting the API coverage pages")
        diffs.clear()
        return config
    nav, i = r
    nav[i + 1:i + 1] = [page_uri(api) for api in diffs]

    config.watch.append(diffs_dir)
    return config


def indent(s):
    return re.sub(r"^(?=.)", "    ", s, flags = re.M)


def on_files(files, config):
    for api, versions in diffs.items():
        a = APIS[api]
        md = f"# {a.title}\n\n"
        if a.versioned:
            md += f"This page shows how {a.scale_side} corresponds to each version of {a.nv_side}.\n"
        else:
            md += f"This page shows how {a.scale_side} corresponds to the current version of " \
                  f"{a.nv_side}.\n"
        md += "\n\n"
        if a.versioned:
            # Oldest first, with the newest selected.
            for v, d in reversed(versions.items()):
                select = "+" if v == next(iter(versions)) else ""
                md += f'==={select} "CUDA {v}"\n\n{indent(d.meter())}\n\n{indent(d.body)}\n'
        else:
            d = next(iter(versions.values()))
            md += f"{d.meter()}\n\n{d.body}"
        files.append(File.generated(config, page_uri(api), content = md))
    return files


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != INDEX or not diffs:
        return markdown
    versions = sorted({v for vs in diffs.values() for v in vs}, key = version_key)
    rows = ["| API | " + " | ".join(f"CUDA {v}" for v in versions) + " |",
            "|---|" + "---|" * len(versions)]
    for api, vs in diffs.items():
        if APIS[api].versioned:
            cells = [vs[v].meter(label = False) if v in vs else "" for v in versions]
        else:
            cells = [""] * (len(versions) - 1) + [next(iter(vs.values())).meter(label = False)]
        rows.append(f"| [{APIS[api].title}]({api}-api.md) | " + " | ".join(cells) + " |")
    for api in diffs:
        if not APIS[api].versioned:
            rows += ["", f"{APIS[api].title}'s documentation is not versioned with CUDA: its row shows the "
                         "current version."]
    return markdown.replace(INDEX_MARKER, "\n".join(rows))


def on_page_content(html, page, config, files):
    """Keep all but the newest version out of the search index, so a search finds a declaration once.
    Material drops a heading's section if the heading has data-search-exclude."""
    if page.file.src_uri not in map(page_uri, diffs):
        return html
    blocks = html.split('<div class="tabbed-block">')
    for i in range(1, len(blocks) - 1): # The newest is last.
        blocks[i] = re.sub(r"<(h[1-6]) ", r"<\1 data-search-exclude ", blocks[i])
    return '<div class="tabbed-block">'.join(blocks)
