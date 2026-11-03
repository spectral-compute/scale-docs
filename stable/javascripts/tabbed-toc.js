// A heading in a content tab is listed in the table of contents whichever tab is selected, so pages
// with a tab per version would list each section once per version. List only the selected tabs'.
document$.subscribe(() => {
  const entries = [];
  for (const a of document.querySelectorAll(".md-nav--secondary a.md-nav__link")) {
    // Instant navigation makes the links absolute.
    const block = a.pathname === location.pathname && a.hash &&
      document.getElementById(decodeURIComponent(a.hash.slice(1)))?.closest(".tabbed-block");
    if (block) {
      const set = block.closest(".tabbed-set");
      const i = Array.prototype.indexOf.call(block.parentElement.children, block);
      entries.push([a.closest("li"), set.querySelectorAll(":scope > input")[i]]);
    }
  }
  if (!entries.length)
    return;

  const update = () => {
    for (const [li, input] of entries)
      li.hidden = !input.checked;
  };
  for (const input of new Set(entries.map(([, input]) => input)))
    input.closest(".tabbed-set").addEventListener("change", update);
  update();
});
