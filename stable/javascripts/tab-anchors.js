// Material selects the tab whose input is the URL fragment, but not the tabs enclosing it, so a
// nested tab would stay hidden. Select those too. style.css puts a nested tab's input at the top of
// the outermost tab set, so scrolling to it shows all the enclosing tabs' labels.
function selectEnclosingTabs() {
  const id = decodeURIComponent(location.hash.slice(1));
  const input = id && document.getElementById(id);
  if (!input?.matches(".tabbed-set > input"))
    return;

  let revealed = false;
  for (let block = input.closest(".tabbed-block"); block;
       block = block.parentElement.closest(".tabbed-block")) {
    const i = Array.prototype.indexOf.call(block.parentElement.children, block);
    const outer = block.closest(".tabbed-set").querySelectorAll(":scope > input")[i];
    if (!outer.checked) {
      outer.click();
      revealed = true;
    }
  }
  if (revealed)
    input.scrollIntoView(); // The browser can't scroll to it while it's hidden.
}

document$.subscribe(selectEnclosingTabs);
addEventListener("hashchange", selectEnclosingTabs);
