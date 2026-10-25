#!/usr/bin/env python3
"""Build the blog from the Markdown notes in "Deep Learning from Zero to Kang".

    python tools/build_blog.py "<path to Deep Learning from Zero to Kang>"

Writes blog.html, blog/<slug>.html and blog/media/<slug>/* next to this tools/
folder. Each post page embeds its Markdown in English and Chinese; assets/post.js
renders it in the browser (marked + KaTeX + highlight.js), so the site stays
plain static files. English is the default; the reader can switch to 中文.

English versions of the Chinese-only study notes live in content/notes-en/<slug>.md.

To add a post: put it in the source folder, add a line to SERIES or NOTES
below, and run the script again.
"""
import html
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote

SITE = Path(__file__).resolve().parent.parent
NOTES_EN = SITE / 'content' / 'notes-en'

# The bilingual series: (folder under "Deep Learning", slug, English title, Chinese title)
SERIES = [
    ('Activation functions', '激活函数', [
        ('Activate Function/Sigmoid', 'sigmoid', 'Sigmoid', 'Sigmoid 激活函数'),
        ('Activate Function/Tanh', 'tanh', 'Tanh', 'Tanh 激活函数'),
        ('Activate Function/ReLU', 'relu', 'ReLU', 'ReLU 激活函数'),
        ('Activate Function/Leaky ReLU', 'leaky-relu', 'Leaky ReLU', 'Leaky ReLU 激活函数'),
        ('Activate Function/PReLU', 'prelu', 'PReLU', 'PReLU 激活函数'),
        ('Activate Function/RReLU', 'rrelu', 'RReLU', 'RReLU 激活函数'),
        ('Activate Function/ELU', 'elu', 'ELU', 'ELU 激活函数'),
    ]),
    ('Loss functions', '损失函数', [
        ('Loss function/NLL Loss', 'nll-loss', 'NLL Loss', 'NLL 损失函数'),
        ('Loss function/BiNLL Loss', 'binll-loss', 'Bivariate Gaussian NLL Loss', '双变量高斯负对数似然损失'),
        ('Loss function/Difference between NLLloss,KLDivLoss,CrossEntropyLoss', 'nll-kl-ce',
         'NLLLoss, KLDivLoss and CrossEntropyLoss', 'NLLLoss、KLDivLoss 与 CrossEntropyLoss 对比'),
    ]),
    ('Sequence models', '序列模型', [
        ('Model/Basic Model/RNN', 'rnn', 'Recurrent Neural Networks (RNN)', '循环神经网络（RNN）'),
        ('Model/Basic Model/LSTM', 'lstm', 'Long Short-Term Memory (LSTM)', '长短期记忆网络（LSTM）'),
    ]),
    ('Vision backbones', '视觉骨干网络', [
        ('Model/Popular Model/Visual Extractor/LeNet', 'lenet', 'LeNet', 'LeNet'),
        ('Model/Popular Model/Visual Extractor/AlexNet', 'alexnet', 'AlexNet', 'AlexNet'),
        ('Model/Popular Model/Visual Extractor/VGG', 'vgg', 'VGG', 'VGG'),
        ('Model/Popular Model/Visual Extractor/GoogleNet', 'googlenet', 'GoogLeNet', 'GoogLeNet'),
    ]),
    ('Positional embeddings', '位置编码', [
        ('Positional Embedding/Sinusoidal Position Embeds', 'sinusoidal-position-embeds',
         'Sinusoidal Position Embeddings', '正弦位置编码'),
        ('Positional Embedding/Random Or Learned Sinusoidal Position Embeds', 'learned-sinusoidal-position-embeds',
         'Learned Sinusoidal Position Embeddings', '可学习的正弦位置编码'),
    ]),
]

# Earlier self-study notes, written in Chinese:
# (file under "机器学习自学笔记", slug, Chinese title, English title)
NOTES = [
    ('Machine learning basics', '机器学习基础', [
        ('机器学习基础/监督学习——感知机/感知机.md', 'ml-perceptron', '感知机', 'Perceptron'),
        ('机器学习基础/监督学习——k邻近算法/k邻近算法.md', 'ml-knn', 'k 近邻算法', 'k-Nearest Neighbors'),
        ('机器学习基础/监督学习——朴素贝叶斯/朴素贝叶斯算法.md', 'ml-naive-bayes', '朴素贝叶斯', 'Naive Bayes'),
        ('机器学习基础/监督学习——决策树/决策树(Decision Tree).md', 'ml-decision-tree', '决策树', 'Decision Trees'),
        ('机器学习基础/监督学习——SVM支持向量机/支持向量机（SVM）.md', 'ml-svm', '支持向量机（SVM）', 'Support Vector Machines (SVM)'),
        ('机器学习基础/无监督学习——聚类/聚类.md', 'ml-clustering', '聚类', 'Clustering'),
        ('机器学习基础/泛化性与模型选择/泛化性与模型选择.md', 'ml-generalization', '泛化性与模型选择', 'Generalization and Model Selection'),
        ('其它/机器学习常用评价标准.md', 'ml-metrics', '常用评价指标', 'Common Evaluation Metrics'),
    ]),
    ('Deep learning fundamentals', '深度学习基础', [
        ('深度学习/算法基础/算法笔记/神经网络/第一节 深度前馈网络MLP/1.1基础概念.md', 'dl-mlp-basics', '深度前馈网络：基础概念', 'Deep Feedforward Networks: Basic Concepts'),
        ('深度学习/算法基础/算法笔记/神经网络/第一节 深度前馈网络MLP/1.2输出单元.md', 'dl-output-units', '深度前馈网络：输出单元', 'Deep Feedforward Networks: Output Units'),
        ('深度学习/算法基础/算法笔记/神经网络/第一节 深度前馈网络MLP/1.4反向传播.md', 'dl-backprop', '深度前馈网络：反向传播', 'Deep Feedforward Networks: Backpropagation'),
        ('深度学习/算法基础/算法笔记/神经网络/第二节 卷积神经网络CNN/2.1卷积运算.md', 'dl-convolution', '卷积运算', 'The Convolution Operation'),
        ('深度学习/算法基础/博客1——输出单元理解.md', 'dl-output-units-notes', '理解输出单元', 'Understanding Output Units'),
        ('深度学习/算法基础/算法笔记/知识蒸馏/知识蒸馏.md', 'dl-distillation', '知识蒸馏', 'Knowledge Distillation'),
        ('深度学习/算法基础/算法笔记/知识蒸馏/知识蒸馏代码pytorch.md', 'dl-distillation-code', '知识蒸馏的 PyTorch 实现', 'Knowledge Distillation in PyTorch'),
    ]),
    ('Dive into Deep Learning notes', '《动手学深度学习》笔记', [
        ('深度学习/动手学深度学习-李沐/1-数据操作与预处理/数据操作与预处理.md', 'd2l-data', '数据操作与预处理', 'Data Manipulation and Preprocessing'),
        ('深度学习/动手学深度学习-李沐/2-线性代数相关基础/线性代数相关基础.md', 'd2l-linear-algebra', '线性代数基础', 'Linear Algebra Basics'),
        ('深度学习/动手学深度学习-李沐/3-导数/导数.md', 'd2l-derivatives', '导数', 'Derivatives'),
        ('深度学习/动手学深度学习-李沐/4-线性回归/线性回归.md', 'd2l-linear-regression', '线性回归', 'Linear Regression'),
        ('深度学习/动手学深度学习-李沐/5-softmax回归/softmax回归与损失函数.md', 'd2l-softmax', 'softmax 回归与损失函数', 'Softmax Regression and Loss Functions'),
    ]),
    ('PyTorch and tools', 'PyTorch 与工具', [
        ('深度学习/pytorch语法/Tensor语法/Tensor语法.md', 'pt-tensor', 'Tensor 基础', 'Tensor Basics'),
        ('深度学习/pytorch语法/Tensor语法/Tensor语法进阶.md', 'pt-tensor-advanced', 'Tensor 进阶', 'More on Tensors'),
        ('深度学习/pytorch语法/torchvision语法/torchvision.transform_s/Pytorch基础语法——transform语法.md', 'pt-transforms', 'torchvision.transforms', 'torchvision.transforms'),
        ('深度学习/pytorch语法/MNIST语法/Pytorch基础语法——MNIST语法.md', 'pt-mnist', '加载 MNIST', 'Loading MNIST'),
        ('深度学习/pytorch语法/Argparse语法/Pytorch基础语法——argparse模块.md', 'pt-argparse', 'argparse 模块', 'The argparse Module'),
        ('深度学习/pytorch语法/OS语法/OS语法.md', 'pt-os', 'os 模块', 'The os Module'),
        ('深度学习/pytorch语法/GIL语法/Pytorch基础语法——GIL语法.md', 'pt-gil', 'PIL 图像处理', 'Image Processing with PIL'),
        ('深度学习/Pytorch安装/Pytorch3D/Pytorch3D安装.md', 'setup-pytorch3d', '安装 PyTorch3D', 'Installing PyTorch3D'),
        ('深度学习/Pytorch安装/ffmpeg/ffmpeg.md', 'setup-ffmpeg', '安装 ffmpeg', 'Installing ffmpeg'),
    ]),
    ('Troubleshooting', '报错解决', [
        ('其它/机器学习报错解决/机器学习报错解决1——import argparse.md', 'fix-argparse', 'import argparse 报错', 'An import argparse Error'),
        ('其它/机器学习报错解决/机器学习报错解决2——ValueError too many values to unpack (expected 3).md', 'fix-unpack', 'ValueError: too many values to unpack', 'ValueError: too many values to unpack'),
        ('其它/深度学习相关报错解决/opencv报错解决1.md', 'fix-opencv-1', 'OpenCV 报错（一）', 'OpenCV Errors, Part 1'),
        ('其它/深度学习相关报错解决/opencv报错解决2.md', 'fix-opencv-2', 'OpenCV 报错（二）', 'OpenCV Errors, Part 2'),
    ]),
]

NOTES_ROOT = None  # set in main(): the source's 机器学习自学笔记 folder
# Typora writes paths with spaces unescaped (`![x](D:\a b\c.png)`), so the path may contain spaces
MD_IMG = re.compile(r'!\[([^\]]*)\]\(\s*<?([^)>"]+?)>?(\s+"[^"]*")?\s*\)')
HTML_IMG = re.compile(r'<img\b[^>]*?\bsrc=["\']([^"\']+)["\'][^>]*>', re.I)
missing = []


def t(en, cn):
    """A UI string in both languages; blog.css shows the one matching body[data-lang]."""
    return f'<span data-l="en">{en}</span><span data-l="cn">{cn}</span>'


def read(path):
    text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    if text.startswith('---\n'):
        end = text.find('\n---', 4)
        if end != -1:
            text = text[end + 4:]
    # Typora's table-of-contents marker means nothing on the web
    text = re.sub(r'^[ \t]*\[(?:TOC|toc)\][ \t]*\n?', '', text, flags=re.M)
    # left over from machine translation of the Chinese originals
    return re.sub(r"^Here's a clear and concise translation[^\n]*\n", '', text, flags=re.M)


def find_image(p, src_file):
    """Resolve an image path; notes written on an older computer use absolute paths
    or a renamed folder, so fall back to the same file name near the note."""
    if not (re.match(r'^[A-Za-z]:/', p) or p.startswith('/')):
        direct = (src_file.parent / p).resolve()
        if direct.is_file():
            return direct
    marker = '机器学习自学笔记/'
    if marker in p:
        moved = NOTES_ROOT / p.split(marker, 1)[1]
        if moved.is_file():
            return moved
    name = p.rsplit('/', 1)[-1]
    return next((f for f in src_file.parent.rglob(name) if f.is_file()), None)


def copy_image(src, dest):
    """Copy, shrinking big photos (phone shots of handwritten notes run to 2 MB)."""
    if dest.exists():
        return
    if src.stat().st_size > 300_000 and src.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        try:
            from PIL import Image
            im = Image.open(src)
            if im.width > 1600:
                im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
            if src.suffix.lower() == '.png' and im.mode in ('RGBA', 'LA', 'P'):
                im.save(dest, optimize=True)
            else:
                im.convert('RGB').save(dest, quality=82, optimize=True)
            return
        except ImportError:
            pass
    shutil.copy2(src, dest)


def localize_images(text, src_file, slug):
    """Copy every local image the note uses into blog/media/<slug>/ and point the note at it.
    `src_file` is the original note, also for translations, so relative paths resolve."""
    media = SITE / 'blog' / 'media' / slug

    def fix(path):
        if re.match(r'^(https?:)?//', path):
            return path
        src = find_image(unquote(path).replace('\\', '/'), src_file)
        if src is None:
            missing.append(f'{slug}: {path}')
            return None
        media.mkdir(parents=True, exist_ok=True)
        name = re.sub(r'[^\w.-]+', '-', src.parent.name + '-' + src.name if src.parent.name not in ('pic',) else src.name)
        copy_image(src, media / name)
        return f'media/{slug}/{name}'

    def md_sub(m):
        new = fix(m.group(2))
        return '' if new is None else f'![{m.group(1)}]({new})'

    def html_sub(m):
        new = fix(m.group(1))
        return '' if new is None else m.group(0).replace(m.group(1), new)

    return HTML_IMG.sub(html_sub, MD_IMG.sub(md_sub, text))


def strip_title(text):
    """The page shows the title itself, so drop a leading H1."""
    return re.sub(r'^\s*#\s+[^\n]*\n', '', text, count=1)


def plain(text):
    s = re.sub(r'```.*?```', ' ', text, flags=re.S)
    s = re.sub(r'\$\$.*?\$\$', ' ', s, flags=re.S)
    s = re.sub(r'\$([^$\n]+)\$', r'\1', s)
    s = re.sub(r'</?[A-Za-z][^>]*>|!\[[^\]]*\]\([^)]*\)', ' ', s)
    return re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)


def excerpt(text):
    for para in re.split(r'\n\s*\n', plain(text)):
        p = re.sub(r'\s+', ' ', para).strip()
        cjk = len(re.findall(r'[一-鿿]', p)) > len(p) / 3
        if not p or p.startswith(('#', '|', '-', '*', '>', '（', '(')) or len(p) < (24 if cjk else 40):
            continue
        p = re.sub(r'[*_`$]', '', p)
        n = 80 if cjk else 150
        return p if len(p) <= n else p[:n].rstrip() + '…'
    return ''


def minutes(text):
    s = plain(text)
    cjk = len(re.findall(r'[一-鿿]', s))
    words = len(re.findall(r'[A-Za-z0-9]+', s))
    return max(1, round(cjk / 450 + words / 220))


def embed(md):
    return md.replace('</script', '<\\/script')


HEAD = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Yongkang Li</title>
<meta name="description" content="{desc}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<link rel="stylesheet" href="{root}assets/site.css">
<link rel="stylesheet" href="{root}assets/blog.css">
<script>try {{ const t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; }} catch (e) {{}}</script>
</head>
<body data-page="blog" data-root="{root}" data-lang="en">
<div class="sky" aria-hidden="true"><i></i><i></i><i></i></div>
<main class="page"><div class="wrap">
'''

FOOT = '''<footer class="site"><span>© 2026 Yongkang Li</span><a href="{root}blog.html">''' + t('All posts', '全部文章') + '''</a></footer>
</div></main>
<script src="{root}assets/site.js"></script>
<script src="{root}assets/blog-lang.js"></script>
{scripts}
</body>
</html>
'''

POST_SCRIPTS = '''<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="../assets/post.js"></script>'''

LANGSEG = ('<div class="seg glass langseg" role="group" aria-label="Language">'
           '<button type="button" data-lang="en">English</button>'
           '<button type="button" data-lang="cn">中文</button></div>')


def both(d, key):
    """`d[key]` in both languages, falling back to whichever exists."""
    return d[key].get('en') or d[key]['cn'], d[key].get('cn') or d[key]['en']


def post_page(post, prev, nxt):
    langs = [l for l in ('en', 'cn') if l in post['md']]
    en_title, cn_title = both(post, 'title')
    mins_en, mins_cn = both(post, 'minutes')
    note = ''
    if langs == ['cn']:
        note = '<p class="only">' + t('This note is only available in Chinese.', '这篇笔记只有中文版。') + '</p>'
    sources = '\n'.join(
        f'<script type="text/markdown" data-lang="{l}">\n{embed(post["md"][l])}\n</script>' for l in langs)

    def link(p, en, cn):
        if not p:
            return '<span></span>'
        pe, pc = both(p, 'title')
        return f'<a class="pn glass" href="{p["slug"]}.html"><small>{t(en, cn)}</small><span>{t(html.escape(pe), html.escape(pc))}</span></a>'

    body = f'''
<article class="post">
  <nav class="crumbs"><a href="../blog.html">{t('Blog', '博客')}</a><i>/</i><span>{t(*post["series"])}</span><i>/</i><span>{t(*post["category"])}</span></nav>
  <header class="post-head">
    <h1>{t(html.escape(en_title), html.escape(cn_title))}</h1>
    <div class="meta">{t(f'{mins_en} min read', f'约 {mins_cn} 分钟')}{LANGSEG}</div>
  </header>
  {note}
  <div class="md glass" id="article"><noscript>This page needs JavaScript to render the notes.</noscript></div>
  <nav class="prevnext">{link(prev, "Previous", "上一篇")}{link(nxt, "Next", "下一篇")}</nav>
</article>
{sources}
'''
    desc = post['excerpt'].get('en') or post['excerpt'].get('cn', '')
    return HEAD.format(title=html.escape(en_title), desc=html.escape(desc), root='../') + body + FOOT.format(root='../', scripts=POST_SCRIPTS)


def tile(post):
    en_title, cn_title = both(post, 'title')
    ex_en, ex_cn = both(post, 'excerpt')
    mins_en, mins_cn = both(post, 'minutes')
    return (f'<a class="tile glass" href="blog/{post["slug"]}.html">'
            f'<h3>{t(html.escape(en_title), html.escape(cn_title))}</h3>'
            f'<p>{t(html.escape(ex_en), html.escape(ex_cn))}</p>'
            f'<span class="meta">{t(f"{mins_en} min", f"{mins_cn} 分钟")}</span></a>')


def index_page(series_posts, note_posts):
    n_series = sum(len(ps) for _, ps in series_posts)
    n_notes = sum(len(ps) for _, ps in note_posts)
    blocks = ''.join(f'<section class="cat"><h2>{t(*cat)}<small>{len(ps)}</small></h2>'
                     f'<div class="tiles">{"".join(tile(p) for p in ps)}</div></section>' for cat, ps in series_posts)
    notes = ''
    for cat, ps in note_posts:
        rows = ''.join(f'<a href="blog/{p["slug"]}.html">{t(*map(html.escape, both(p, "title")))}'
                       f'<small>{t(str(both(p, "minutes")[0]) + " min", str(both(p, "minutes")[1]) + " 分钟")}</small></a>' for p in ps)
        notes += f'<div class="nlist glass"><h3>{t(*cat)}</h3>{rows}</div>'
    body = f'''
<header class="blog-head">
  <div class="eyebrow">{t('Blog', '博客')}</div>
  <h1>Deep Learning <b class="grad">from Zero to Kang</b></h1>
  <p>{t('Notes on the building blocks of deep learning, written while I was learning them: activation functions, losses, sequence models, vision backbones and positional embeddings, plus my earlier machine-learning study notes. Every post has an English and a Chinese version. Older posts also live on <a href="https://www.zhihu.com/people/li-yong-kang-69-58">Zhihu</a>.',
           '我在学习深度学习时写下的笔记：激活函数、损失函数、序列模型、视觉骨干网络和位置编码，以及更早的机器学习自学笔记。每篇文章都有中文和英文两个版本。更早的文章也可以在<a href="https://www.zhihu.com/people/li-yong-kang-69-58">知乎</a>上看到。')}</p>
  <div class="stats"><span>{t(f'<b>{n_series}</b> topics in the series', f'系列文章 <b>{n_series}</b> 篇')}</span><span>{t(f'<b>{n_notes}</b> study notes', f'自学笔记 <b>{n_notes}</b> 篇')}</span>{LANGSEG}</div>
</header>
{blocks}
<section class="cat" id="notes">
  <h2>{t('Machine-learning study notes', '机器学习自学笔记')}<small>{n_notes}</small></h2>
  <p class="dim" style="margin:-6px 0 18px">{t('From my undergraduate years, originally written in Chinese.', '本科阶段的自学笔记。')}</p>
  <div class="notes">{notes}</div>
</section>
'''
    return (HEAD.format(title='Blog', root='', desc='Deep Learning from Zero to Kang: notes on the building blocks of deep learning, in English and Chinese.')
            + body + FOOT.format(root='', scripts=''))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    global NOTES_ROOT
    src = Path(sys.argv[1])
    NOTES_ROOT = src / '机器学习自学笔记'
    out = SITE / 'blog'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    series = ('Deep Learning from Zero to Kang', 'Deep Learning from Zero to Kang')
    series_posts = []
    for cat, cat_cn, items in SERIES:
        ps = []
        for folder, slug, en, cn in items:
            d = src / 'Deep Learning' / folder
            md = {}
            for lang, suffix in (('en', 'EN'), ('cn', 'CN')):
                f = next(d.glob(f'*-{suffix}.md'), None)
                if f:
                    md[lang] = localize_images(strip_title(read(f)), f, slug)
            ps.append({'slug': slug, 'title': {'en': en, 'cn': cn}, 'md': md, 'category': (cat, cat_cn), 'series': series,
                       'excerpt': {l: excerpt(m) for l, m in md.items()}, 'minutes': {l: minutes(m) for l, m in md.items()}})
        series_posts.append(((cat, cat_cn), ps))

    notes_series = ('Machine-learning study notes', '机器学习自学笔记')
    note_posts = []
    untranslated = []
    for cat, cat_cn, items in NOTES:
        ps = []
        for rel, slug, cn, en in items:
            f = src / '机器学习自学笔记' / rel
            md = {'cn': localize_images(strip_title(read(f)), f, slug)}
            tr = NOTES_EN / f'{slug}.md'
            if tr.is_file():
                md['en'] = localize_images(strip_title(read(tr)), f, slug)
            else:
                untranslated.append(slug)
            ps.append({'slug': slug, 'title': {'en': en, 'cn': cn}, 'md': md, 'category': (cat, cat_cn), 'series': notes_series,
                       'excerpt': {l: excerpt(m) for l, m in md.items()}, 'minutes': {l: minutes(m) for l, m in md.items()}})
        note_posts.append(((cat, cat_cn), ps))

    for group in [[p for _, ps in series_posts for p in ps]] + [ps for _, ps in note_posts]:
        for i, p in enumerate(group):
            prev = group[i - 1] if i > 0 else None
            nxt = group[i + 1] if i + 1 < len(group) else None
            (out / f'{p["slug"]}.html').write_text(post_page(p, prev, nxt), encoding='utf-8')

    (SITE / 'blog.html').write_text(index_page(series_posts, note_posts), encoding='utf-8')

    total = sum(len(ps) for _, ps in series_posts) + sum(len(ps) for _, ps in note_posts)
    print(f'wrote {total} posts to {out}')
    if untranslated:
        print(f'{len(untranslated)} note(s) without an English version yet: {", ".join(untranslated)}')
    if missing:
        print(f'{len(missing)} image(s) not found, left out:')
        print('\n'.join('  ' + m for m in missing))


if __name__ == '__main__':
    main()
