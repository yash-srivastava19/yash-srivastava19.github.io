// The reading rail on blog and diary pages: builds the section tree from the
// post's headings, highlights the section in view, and tracks progress.
(function () {
  var rail = document.querySelector(".reading-rail");
  var article = document.querySelector(".reading article");
  if (!rail || !article) return;

  // Posts differ in heading levels (some open with one h1 and use h2 for
  // sections, some use h1 throughout). The tree lists the shallowest level
  // that appears at least twice, so a lone title heading is skipped.
  var headings = [];
  for (var level = 1; level <= 3 && headings.length < 2; level++) {
    headings = Array.prototype.slice.call(article.querySelectorAll("#content h" + level));
  }
  var links = [];
  if (headings.length >= 2) {
    var list = rail.querySelector(".rail-tree ol");
    headings.forEach(function (h) {
      if (!h.id) return;
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      var li = document.createElement("li");
      li.appendChild(a);
      list.appendChild(li);
      links.push({ heading: h, link: a });
    });
    if (links.length) rail.querySelector(".rail-tree").hidden = false;
  }

  var fill = rail.querySelector(".rail-fill");
  var pct = rail.querySelector(".rail-pct");
  var ticking = false;

  function update() {
    ticking = false;
    var box = article.getBoundingClientRect();
    var span = box.height - window.innerHeight;
    var done = span > 0 ? Math.min(1, Math.max(0, -box.top / span)) : 1;
    fill.style.width = (done * 100).toFixed(1) + "%";
    pct.textContent = ("0" + Math.round(done * 100)).slice(-2) + "%";
    if (done === 1) pct.textContent = "100%";

    // The current section is the last heading above the top third of the screen.
    var current = null;
    links.forEach(function (l) {
      if (l.heading.getBoundingClientRect().top < window.innerHeight / 3) current = l;
    });
    links.forEach(function (l) {
      if (l === current) l.link.setAttribute("aria-current", "true");
      else l.link.removeAttribute("aria-current");
    });
  }

  window.addEventListener("scroll", function () {
    if (!ticking) { ticking = true; requestAnimationFrame(update); }
  }, { passive: true });
  window.addEventListener("resize", update);
  update();

  var copy = rail.querySelector(".rail-copy");
  copy.addEventListener("click", function () {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(copy.dataset.url).then(function () {
      copy.textContent = "copied";
      setTimeout(function () { copy.textContent = "copy link"; }, 1500);
    });
  });
})();
