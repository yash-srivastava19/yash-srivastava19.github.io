// The header's mode slider: light, dark and reading. Click a stop, drag the
// thumb, or use the arrow keys. With nothing saved the page follows the
// system's light or dark setting, and the thumb shows which one that is. The
// saved choice is applied before first paint by the inline script in
// _includes/head.html.
(function () {
  var modes = ["light", "dark", "reading"];
  var root = document.documentElement;
  var slider = document.querySelector(".theme-slider");
  if (!slider) return;
  var stops = slider.querySelectorAll(".theme-stop");
  var system = window.matchMedia("(prefers-color-scheme: dark)");

  function saved() {
    var t = root.getAttribute("data-theme");
    return modes.indexOf(t) >= 0 ? t : null;
  }

  function show() {
    var mode = saved() || (system.matches ? "dark" : "light");
    slider.setAttribute("data-mode", mode);
    stops.forEach(function (s) {
      var on = s.getAttribute("data-mode") === mode;
      s.setAttribute("aria-checked", on ? "true" : "false");
      s.tabIndex = on ? 0 : -1;
    });
  }

  function choose(mode) {
    root.setAttribute("data-theme", mode);
    try { localStorage.setItem("theme", mode); } catch (e) {}
    show();
  }

  stops.forEach(function (s) {
    s.addEventListener("click", function () { choose(s.getAttribute("data-mode")); });
  });

  slider.addEventListener("keydown", function (e) {
    var step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    var i = modes.indexOf(slider.getAttribute("data-mode"));
    var next = modes[(i + step + modes.length) % modes.length];
    choose(next);
    slider.querySelector('[data-mode="' + next + '"]').focus();
  });

  // Dragging: the stop under the pointer wins.
  var dragging = false;
  function stopAt(x) {
    var box = slider.getBoundingClientRect();
    var i = Math.floor(((x - box.left) / box.width) * modes.length);
    return modes[Math.max(0, Math.min(modes.length - 1, i))];
  }
  slider.addEventListener("pointerdown", function (e) {
    dragging = true;
    slider.setPointerCapture(e.pointerId);
  });
  slider.addEventListener("pointermove", function (e) {
    if (dragging) slider.setAttribute("data-mode", stopAt(e.clientX));
  });
  slider.addEventListener("pointerup", function (e) {
    if (!dragging) return;
    dragging = false;
    choose(stopAt(e.clientX));
  });
  slider.addEventListener("pointercancel", function () { dragging = false; show(); });

  system.addEventListener("change", show);
  window.addEventListener("storage", function (e) {
    if (e.key !== "theme") return;
    if (modes.indexOf(e.newValue) >= 0) root.setAttribute("data-theme", e.newValue);
    else root.removeAttribute("data-theme");
    show();
  });

  show();
})();
