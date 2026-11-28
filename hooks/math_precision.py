"""
Generate the tables of the maths functions' precision from MathPrecision.def in the SCALE source tree.

That file lists, for each function and type, SCALE's bound on the error in ulps, and whether it has been
checked on every argument. The maths tests, and the nightly exhaustive tests, take their bounds from the
same file, so the page states what they check. To keep it that way, the build fails if an entry with a
bound is not used by any test.

Beside it, MathPrecisionDistribution.json has the number of arguments at each error, in ulps, from an
exhaustive run of the nightly tests (as libs/util/specmath-ulp-distribution extracts them from
its log). Each section's table is followed by those of its functions that have any error. The build fails if they contradict
MathPrecision.def.
"""

import json
import logging
import math
import os
import re

from mkdocs.exceptions import PluginError

log = logging.getLogger("mkdocs.hooks.math_precision")

TESTS_DIR = os.path.join("..", "..", "test", "cuda_tests", "cuda_gtest") # Relative to mkdocs.yml.
DEF_FILE = os.path.join(TESTS_DIR, "math_variants", "MathPrecision.def")
DIST_FILE = os.path.join(TESTS_DIR, "math_variants", "MathPrecisionDistribution.json")
PAGE = "manual/library/math-precision.md"
MARKER = "<!-- math-precision-tables -->"

TYPES = {"float": "`float`", "double": "`double`", "half": "`__half`", "bf16": "`__nv_bfloat16`"}

# A C string literal, or adjacent ones, which C++ concatenates.
STRING = r'((?:"(?:[^"\\]|\\.)*"\s*)+)'
SECTION_RE = re.compile(r'^\s*MATH_PRECISION_SECTION\(\s*' + STRING + r'\)', re.M)
ENTRY_RE = re.compile(r'^\s*MATH_PRECISION\(\s*(\w+)\s*,\s*' + STRING + r',\s*(\w+)\s*,\s*(-?\d+)\s*,\s*'
                      r'(true|false)\s*,\s*' + STRING + r',\s*' + STRING + r'\)', re.M)


def unquote(s):
    return "".join(bytes(m, "utf-8").decode("unicode_escape") for m in re.findall(r'"((?:[^"\\]|\\.)*)"', s))


class Entry:
    def __init__(self, m):
        self.id = m.group(1)
        self.function = unquote(m.group(2))
        self.type = m.group(3)
        self.ulps = int(m.group(4))
        self.exhaustive = m.group(5) == "true"
        self.notes = unquote(m.group(7))
        if self.type not in TYPES:
            raise PluginError(f"{DEF_FILE}: {self.id} has unknown type {self.type}")

    def cell(self):
        if self.ulps < 0:
            return "see notes"
        # Bold where every argument has been tried.
        return f"**{self.ulps}**" if self.exhaustive else str(self.ulps)


# [(section title, [Entry])], in the file's order.
sections = []

# {(function, type): [number of arguments at each ulp error]}.
distributions = {}


def load_distributions(path, entries):
    with open(path) as f:
        data = json.load(f)
    by_key = {(e.function, e.type): e for e in entries}
    for d in data:
        key = (d["function"], d["type"])
        e = by_key.get(key)
        if e is None or not e.exhaustive:
            raise PluginError(f"{path}: {key} is not an exhaustively tested entry of {DEF_FILE}")
        if len(d["counts"]) - 1 > e.ulps:
            raise PluginError(f"{path}: {key} has errors beyond its bound of {e.ulps} ulps")
        distributions[key] = d["counts"]


def on_config(config):
    sections.clear()
    distributions.clear()
    root = os.path.dirname(config.config_file_path)
    path = os.path.normpath(os.path.join(root, DEF_FILE))
    if not os.path.isfile(path):
        log.warning(f"No {path}: omitting the maths precision tables")
        return config
    with open(path) as f:
        text = f.read()

    # Sections and entries in file order.
    items = sorted([(m.start(), "section", m) for m in SECTION_RE.finditer(text)] +
                   [(m.start(), "entry", m) for m in ENTRY_RE.finditer(text)], key = lambda t: t[0])
    for _, kind, m in items:
        if kind == "section":
            sections.append((unquote(m.group(1)), []))
        elif not sections:
            raise PluginError(f"{path}: {m.group(1)} comes before any MATH_PRECISION_SECTION")
        else:
            sections[-1][1].append(Entry(m))
    entries = [e for _, es in sections for e in es]
    if len({e.id for e in entries}) != len(entries):
        raise PluginError(f"{path}: duplicate ids")

    # Every bound must be one a test applies.
    tests_dir = os.path.normpath(os.path.join(root, TESTS_DIR))
    used = set()
    for dirpath, _, filenames in os.walk(tests_dir):
        for name in filenames:
            if name.endswith((".cu", ".cpp", ".h", ".hpp")):
                with open(os.path.join(dirpath, name), errors = "replace") as f:
                    used.update(re.findall(r"MathPrecision::(\w+)", f.read()))
    unused = [e.id for e in entries if e.ulps >= 0 and e.id not in used]
    if unused:
        raise PluginError(f"{path}: no test uses the bounds of " + ", ".join(unused))

    config.watch.append(path)

    dist_path = os.path.normpath(os.path.join(root, DIST_FILE))
    if os.path.isfile(dist_path):
        load_distributions(dist_path, entries)
        config.watch.append(dist_path)
    else:
        log.warning(f"No {dist_path}: omitting the distributions of the maths functions' errors")
    return config


def table(entries):
    types = [t for t in TYPES if any(e.type == t for e in entries)]
    rows = {}
    for e in entries:
        rows.setdefault(e.function, {})[e.type] = e
    has_notes = any(e.notes for e in entries)
    header = ["Function"] + [TYPES[t] for t in types] + (["Notes"] if has_notes else [])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for function, by_type in rows.items():
        cells = [by_type[t].cell() if t in by_type else "" for t in types]
        # The same note for every type stands alone, and others say which type they're for.
        noted = [t for t in types if t in by_type and by_type[t].notes]
        if noted and len(noted) == len(by_type) and len({by_type[t].notes for t in noted}) == 1:
            notes = by_type[noted[0]].notes
        else:
            notes = "<br>".join(f"{TYPES[t]}: {by_type[t].notes}" for t in noted)
        row = [f"`{function}`"] + cells + ([notes] if has_notes else [])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def percentage(count, total):
    """A percentage to 3 significant figures, but neither 0 nor 100 unless exactly so."""
    p = 100 * count / total
    if count in (0, total):
        return f"{p:.0f}"
    places = max(0, 2 - math.floor(math.log10(p)))
    while float(f"{p:.{places}f}") in (0, 100):
        places += 1
    return f"{p:.{places}f}"


def distribution_table(entries):
    # Correctly rounded functions' rows would say nothing the bounds don't.
    rows = [(e, counts) for e in entries if len(counts := distributions.get((e.function, e.type), [])) > 1]
    if not rows:
        return ""
    width = max(len(counts) for _, counts in rows)
    # A single type goes in the title rather than a column.
    types = {e.type for e, _ in rows}
    title = f"Distribution of {TYPES[next(iter(types))]} errors" if len(types) == 1 else "Distribution of errors"
    header = ["Function"] + (["Type"] if len(types) > 1 else []) + [f"{u} ULP" for u in range(width)] + ["Mean"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for e, counts in rows:
        total = sum(counts)
        cells = [f'<span title="{c:,} of {total:,} arguments">{percentage(c, total)}%</span>'
                 for c in counts] + [""] * (width - len(counts))
        mean = sum(u * c for u, c in enumerate(counts)) / total
        row = [f"`{e.function}`"] + ([TYPES[e.type]] if len(types) > 1 else []) + cells + [f"{mean:.3g}"]
        lines.append("| " + " | ".join(row) + " |")
    return (f'\n\n??? info "{title}"\n\n' +
            "\n".join("    " + l for l in lines))


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != PAGE:
        return markdown
    if not sections:
        return markdown.replace(MARKER, "*The tables are generated from the SCALE source tree, which this "
                                        "build of the documentation lacks.*")
    md = "\n\n".join(f"## {title}\n\n{table(entries)}{distribution_table(entries)}"
                     for title, entries in sections)
    return markdown.replace(MARKER, md)
