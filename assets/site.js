// Shared page shell: the right menu rail, light/dark switch and scroll spy.
// Add a page or a section to the menu by adding one line to NAV.

const NAV = [
  { group: 'Pages' },
  { id: 'top', label: 'Home', href: 'index.html#top', icon: 'home' },
  { id: 'projects', label: 'Projects', href: 'index.html#projects', icon: 'layers', page: 'projects' },
  { id: 'publications', label: 'Publications', href: 'index.html#publications', icon: 'file' },
  { id: 'blog', label: 'Blog', href: 'blog.html', icon: 'pen', page: 'blog' },
  { id: 'experience', label: 'Experience', href: 'index.html#experience', icon: 'cap' },
  { id: 'contact', label: 'Contact', href: 'index.html#contact', icon: 'mail' },
  { group: 'Elsewhere' },
  { label: 'Google Scholar', href: 'https://scholar.google.com/citations?user=XocWx_UAAAAJ', icon: 'book' },
  { label: 'GitHub', href: 'https://github.com/FranklinNotOld', icon: 'code' },
  { label: 'Zhihu', href: 'https://www.zhihu.com/people/li-yong-kang-69-58', icon: 'globe' },
];

const ICONS = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M10 21v-6h4v6"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/>',
  file: '<path d="M14 3H6v18h12V7z"/><path d="M14 3v4h4"/><path d="M9 13h6M9 17h6"/>',
  pen: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13 7 4 4"/>',
  cap: '<path d="M2 9l10-5 10 5-10 5z"/><path d="M6 11v5c3 2 9 2 12 0v-5"/>',
  mail: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/>',
  book: '<path d="M2 5h7a3 3 0 0 1 3 3v12a2 2 0 0 0-2-2H2z"/><path d="M22 5h-7a3 3 0 0 0-3 3v12a2 2 0 0 1 2-2h8z"/>',
  code: '<path d="m8 8-4 4 4 4M16 8l4 4-4 4M14 5l-4 14"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/>',
  moon: '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
};
const svg = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

(function shell() {
  const root = document.documentElement;
  const base = document.body.dataset.root || '';
  const page = document.body.dataset.page || 'home';
  const resolve = href => /^https?:/.test(href) ? href : base + href;

  // theme: explicit choice wins, else follow the system
  try { const t = localStorage.getItem('theme'); if (t) root.dataset.theme = t; } catch (e) {}
  const isDark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;

  const items = NAV.map(n => n.group
    ? `<div class="grp">${n.group}</div>`
    : `<a class="it" href="${resolve(n.href)}" ${n.id ? `data-id="${n.id}"` : 'target="_blank" rel="noopener"'} title="${n.label}">${svg(n.icon)}<span>${n.label}</span></a>`).join('');

  const rail = document.createElement('aside');
  rail.className = 'rail glass';
  rail.setAttribute('aria-label', 'Site menu');
  rail.innerHTML = `
    <a class="me" href="${resolve('index.html#top')}"><span class="orb" aria-hidden="true"></span><div><b>Yongkang Li</b><span>Ph.D. student · Purdue</span></div></a>
    <nav>${items}</nav>
    <div class="foot"><small>© 2026</small>
      <div class="themebtn" role="group" aria-label="Appearance">
        <button type="button" data-t="light" title="Light">${svg('sun')}</button>
        <button type="button" data-t="dark" title="Dark">${svg('moon')}</button>
      </div>
    </div>`;
  const btn = document.createElement('button');
  btn.className = 'menubtn glass'; btn.type = 'button'; btn.setAttribute('aria-label', 'Open menu'); btn.innerHTML = svg('menu');
  const scrim = document.createElement('div');
  scrim.className = 'scrim';
  document.body.prepend(btn, rail, scrim);

  const toggle = open => { rail.classList.toggle('open', open); btn.setAttribute('aria-expanded', open); };
  btn.addEventListener('click', () => toggle(!rail.classList.contains('open')));
  scrim.addEventListener('click', () => toggle(false));
  rail.addEventListener('click', e => { if (e.target.closest('a')) toggle(false); });

  const tbtns = rail.querySelectorAll('.themebtn button');
  const paint = () => tbtns.forEach(b => b.setAttribute('aria-pressed', (b.dataset.t === 'dark') === isDark()));
  tbtns.forEach(b => b.addEventListener('click', () => {
    root.dataset.theme = b.dataset.t;
    try { localStorage.setItem('theme', b.dataset.t); } catch (e) {}
    paint();
    document.dispatchEvent(new CustomEvent('themechange'));
  }));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { paint(); document.dispatchEvent(new CustomEvent('themechange')); });
  paint();

  // which menu item is lit: the page itself, or on the home page the section in view
  const links = [...rail.querySelectorAll('a.it[data-id]')];
  const light = id => links.forEach(a => a.classList.toggle('on', a.dataset.id === id));
  if (page !== 'home') { light(NAV.find(n => n.page === page)?.id); return; }
  const sections = links.map(a => document.getElementById(a.dataset.id)).filter(Boolean);
  const spy = () => {
    let cur = sections[0]?.id;
    for (const s of sections) if (s.getBoundingClientRect().top < innerHeight * 0.35) cur = s.id;
    light(cur);
  };
  addEventListener('scroll', spy, { passive: true });
  spy();
})();
