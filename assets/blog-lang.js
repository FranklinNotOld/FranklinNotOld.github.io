// Blog language: English by default, 中文 on request. The choice is remembered
// and shared by the blog index and every post (?lang=cn in a link also works).
(function () {
  const body = document.body;
  const set = lang => {
    body.dataset.lang = lang;
    document.documentElement.lang = lang === 'cn' ? 'zh-CN' : 'en';
    document.querySelectorAll('.langseg button').forEach(b => b.setAttribute('aria-pressed', b.dataset.lang === lang));
    try { localStorage.setItem('blog-lang', lang); } catch (e) {}
    document.dispatchEvent(new CustomEvent('langchange', { detail: lang }));
  };
  document.querySelectorAll('.langseg button').forEach(b => b.addEventListener('click', () => set(b.dataset.lang)));
  let lang = new URLSearchParams(location.search).get('lang');
  if (!lang) { try { lang = localStorage.getItem('blog-lang'); } catch (e) {} }
  set(lang === 'cn' ? 'cn' : 'en');
})();
