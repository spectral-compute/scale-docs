/*
 * "Register for SCALE updates" popup on the install page.
 *
 * - Opens after 5 seconds on the page, or as soon as the visitor clicks a code
 *   block's copy button (the copy still happens), whichever comes first.
 * - Shown once per browser: after it is submitted or closed it does not come back.
 *   Add ?register to the URL to forget that (for testing); the usual rules then apply.
 * - Does not wait for the cookie banner: the popup sets no cookies. It opens on
 *   top of the banner, which the visitor can answer once the popup is closed.
 * - Sends the answers as JSON to config.extra.registration.endpoint. With no
 *   endpoint configured it only logs them to the console.
 */
(function () {
  var STORAGE_KEY = "scale-register-seen";

  function seen() {
    try { return localStorage.getItem(STORAGE_KEY) !== null; } catch (e) { return false; }
  }

  function markSeen(how) {
    try { localStorage.setItem(STORAGE_KEY, how + " " + new Date().toISOString()); } catch (e) {}
  }

  function setup(dialog) {
    if (dialog.dataset.ready) return;
    dialog.dataset.ready = "1";

    var form = dialog.querySelector("form");
    var thanks = dialog.querySelector("[data-register-thanks]");
    var run = form.elements.run_target;
    var projectField = dialog.querySelector("[data-register-project]");
    var error = dialog.querySelector("[data-register-error]");
    var endpoint = dialog.dataset.endpoint;

    run.addEventListener("change", function () {
      var opt = run.options[run.selectedIndex];
      var asks = opt && opt.hasAttribute("data-asks-project");
      projectField.hidden = !asks;
      if (!asks) form.elements.project.value = "";
    });

    dialog.querySelectorAll("[data-register-close]").forEach(function (btn) {
      btn.addEventListener("click", function () { dialog.close(); });
    });

    // Closing by the X, Esc or a click on the backdrop all count as "seen".
    dialog.addEventListener("close", function () {
      if (!seen()) markSeen("closed");
    });
    dialog.addEventListener("click", function (e) {
      if (e.target === dialog) dialog.close();
    });

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var email = form.elements.email;
      if (!email.value.trim() || !email.checkValidity()) {
        error.textContent = "Please enter a valid email address.";
        error.hidden = false;
        email.focus();
        return;
      }
      error.hidden = true;

      var payload = {
        email: email.value.trim(),
        heard_from: form.elements.heard_from.value || null,
        run_target: form.elements.run_target.value || null,
        project: form.elements.project.value.trim() || null,
        discord: form.elements.discord.value.trim() || null,
        submitted_at: new Date().toISOString(),
        page: location.pathname,
        website: form.elements.website.value
      };

      if (endpoint) {
        // Fire and forget: the visitor never waits on our server.
        fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          keepalive: true
        }).catch(function () {});
      } else {
        console.info("[scale-register] no endpoint configured; would send:", payload);
      }

      markSeen("registered");
      form.hidden = true;
      thanks.hidden = false;
      thanks.querySelector("button").focus();
    });
  }

  var timer = null;

  function open(dialog) {
    clearTimeout(timer);
    if (document.body.contains(dialog) && !dialog.open && !seen()) dialog.showModal();
  }

  function maybeShow() {
    clearTimeout(timer);
    var dialog = document.getElementById("scale-register");
    if (!dialog || typeof dialog.showModal !== "function") return;
    setup(dialog);

    if (new URLSearchParams(location.search).has("register")) {
      try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    }
    if (seen()) return;

    timer = setTimeout(function () { open(dialog); }, 5000);
  }

  // Copying an install command is the moment of intent: let the copy finish, then ask.
  document.addEventListener("click", function (e) {
    if (!e.target.closest || !e.target.closest(".md-clipboard")) return;
    var dialog = document.getElementById("scale-register");
    if (!dialog || seen()) return;
    setTimeout(function () { open(dialog); }, 300);
  });

  // Material's instant navigation swaps pages without a reload.
  if (typeof document$ !== "undefined") {
    document$.subscribe(maybeShow);
  } else {
    document.addEventListener("DOMContentLoaded", maybeShow);
  }
})();
