(function () {
  "use strict";

  // Mobile menu
  var menuBtn = document.querySelector(".menu");
  var mobileNav = document.getElementById("mobile-nav");
  if (menuBtn && mobileNav) {
    menuBtn.addEventListener("click", function () {
      var open = mobileNav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
    mobileNav.addEventListener("click", function (e) {
      if (e.target.tagName === "A") {
        mobileNav.classList.remove("open");
        menuBtn.setAttribute("aria-expanded", "false");
      }
    });
  }

  // Tabs
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (t) {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
        var pane = document.getElementById(t.getAttribute("aria-controls"));
        if (pane) pane.hidden = true;
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      var target = document.getElementById(tab.getAttribute("aria-controls"));
      if (target) target.hidden = false;
    });
  });

  // Copy buttons
  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var el = document.getElementById(btn.getAttribute("data-copy"));
      if (!el) return;
      var text = el.innerText;
      var done = function () {
        btn.textContent = "Copied";
        btn.classList.add("ok");
        setTimeout(function () {
          btn.textContent = "Copy";
          btn.classList.remove("ok");
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  });

  // Lazy-load the Space iframe
  var loadBtn = document.getElementById("loadDemo");
  var frame = document.getElementById("demoFrame");
  if (loadBtn && frame) {
    loadBtn.addEventListener("click", function () {
      var iframe = document.createElement("iframe");
      iframe.src = "https://itriedcoding-satari-code-demo.hf.space?__theme=dark";
      iframe.title = "Satari Code live demo";
      iframe.loading = "lazy";
      iframe.allow = "clipboard-write";
      frame.innerHTML = "";
      frame.appendChild(iframe);
    });
  }

  // Reveal on scroll
  var prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var items = document.querySelectorAll(".reveal");
  if (prefersReduced || !("IntersectionObserver" in window)) {
    items.forEach(function (el) { el.classList.add("in"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.1 });
    items.forEach(function (el) { io.observe(el); });
  }
})();
