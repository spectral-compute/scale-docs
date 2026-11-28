"""
Give content tabs IDs from their titles, so links can select a tab (Material selects the tab whose
input is the URL fragment). A nested tab's ID is prefixed with its enclosing tab's, e.g.
`rockyrhel-8`.

pymdownx.tabbed's own slugify option can't nest IDs, and assigns them after MkDocs collects each
page's anchors, so MkDocs's anchor validation would reject links to them.
"""

import html

from markdown.extensions import Extension, toc
from markdown.treeprocessors import Treeprocessor
from pymdownx.slugs import slugify

slug = slugify(case = "lower")


class TabIdsTreeprocessor(Treeprocessor):
    def run(self, doc):
        used_ids = {el.attrib["id"] for el in doc.iter() if "id" in el.attrib}
        parents = {c: p for p in doc.iter() for c in p}
        tab_ids = {} # {tabbed-block: its tab's ID}

        # Document order, so an enclosing tab set is done before the ones it contains.
        for el in doc.iter("div"):
            if "tabbed-set" not in el.attrib.get("class", "").split():
                continue
            prefix = ""
            p = parents.get(el)
            while p is not None and p not in tab_ids:
                p = parents.get(p)
            if p is not None:
                prefix = tab_ids[p] + "-"

            inputs = el.findall("input")
            labels = el.findall("div[@class='tabbed-labels']/label")
            blocks = el.findall("div[@class='tabbed-content']/div")
            for inp, label, block in zip(inputs, labels, blocks, strict = True):
                text = html.unescape(toc.strip_tags(toc.render_inner_html(label, self.md)))
                inp.attrib["id"] = label.attrib["for"] = tab_ids[block] = \
                    toc.unique(prefix + slug(text, "-"), used_ids)


class TabIdsExtension(Extension):
    def extendMarkdown(self, md):
        # Before toc and MkDocs's anchor collection (both 5), after attr_list (8).
        md.treeprocessors.register(TabIdsTreeprocessor(md), "tab_ids", 6)


def on_config(config):
    config.markdown_extensions.append(TabIdsExtension())
    return config
