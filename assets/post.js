// Renders the Markdown embedded in a post page (see tools/build_blog.py):
// math with KaTeX, code with highlight.js. blog-lang.js decides the language.
(function () {
  const art = document.getElementById('article');
  const sources = {};
  document.querySelectorAll('script[type="text/markdown"]').forEach(s => { sources[s.dataset.lang] = s.textContent.replace(/<\\\/script/g, '</script'); });

  // Pull math out before marked sees it (it would eat _ and \), put it back rendered.
  function render(src) {
    const math = [];
    const keep = (tex, display) => { math.push([tex, display]); return `MATHPH${math.length - 1}ENDPH`; };
    const text = src.split(/(```[\s\S]*?```|`[^`\n]*`)/g).map((part, i) => i % 2 ? part : part
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => keep(t, true))
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => keep(t, true))
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => keep(t, false))
      .replace(/(^|[^\\$])\$([^\n$]+?)\$/g, (_, pre, t) => pre + keep(t, false))).join('');
    return marked.parse(text, { gfm: true }).replace(/MATHPH(\d+)ENDPH/g, (_, n) => {
      const [tex, display] = math[n];
      try { return katex.renderToString(tex.trim(), { displayMode: display, throwOnError: false }); }
      catch (e) { return tex; }
    });
  }

  let shown = null;
  function show(lang) {
    if (!sources[lang]) lang = Object.keys(sources)[0];
    if (lang === shown) return;
    shown = lang;
    art.innerHTML = render(sources[lang]);
    art.querySelectorAll('pre code').forEach(el => { try { hljs.highlightElement(el); } catch (e) {} });
    art.querySelectorAll('a[href^="http"]').forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });
    art.querySelectorAll('h2, h3').forEach((h, i) => { h.id = h.id || `s${i + 1}`; });
    art.lang = lang === 'cn' ? 'zh-CN' : 'en';
  }

  document.addEventListener('langchange', e => show(e.detail));
  show(document.body.dataset.lang || 'en');
})();
