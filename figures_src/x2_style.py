"""Shared figure style for Paper X2.

One place that fixes the typeface for every rendered figure. The manuscript
sets its text in TeX Gyre Termes (newtxtext), so the figures use the same
OTF files, shipped in figures_src/fonts/, rather than a metric clone: at
final print size the figure labels and the body text then have identical
letterforms. Verify with `pdffonts figure.pdf`; only TeXGyreTermes entries
may appear.
"""
import os
import matplotlib
import matplotlib.font_manager as fm

_HERE = os.path.dirname(os.path.abspath(__file__))
_FONT_DIR = os.path.join(_HERE, 'fonts')
_FACES = ('regular', 'bold', 'italic', 'bolditalic')

FAMILY = 'TeX Gyre Termes'


def register_fonts():
    """Add the bundled Termes faces to matplotlib's font manager."""
    missing = []
    for face in _FACES:
        path = os.path.join(_FONT_DIR, 'texgyretermes-%s.ttf' % face)
        if os.path.exists(path):
            fm.fontManager.addfont(path)
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            'Termes faces not found: %s. Copy them from the TeX distribution '
            'into figures_src/fonts/.' % ', '.join(missing))
    return FAMILY


def rc(size=8):
    """Return rcParams for a manuscript figure at the given base size."""
    register_fonts()
    return {
        'font.family': 'serif',
        'font.serif': [FAMILY],
        'mathtext.fontset': 'custom',
        'mathtext.rm': FAMILY,
        'mathtext.it': '%s:italic' % FAMILY,
        'mathtext.bf': '%s:bold' % FAMILY,
        'font.size': size,
        'axes.labelsize': size,
        'axes.titlesize': size,
        'xtick.labelsize': size - 0.5,
        'ytick.labelsize': size - 0.5,
        'legend.fontsize': size - 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    }


def apply(size=8):
    matplotlib.rcParams.update(rc(size))
