"""Acronym discipline check: every abbreviation is expanded where it first
appears in reading order, and only there.

Reading order is the order main.tex inputs the section files, with the title
block and abstract first. For each abbreviation the script reports its first
occurrence, whether an expansion is attached to it (either
"long form (ABBR)" or "ABBR (long form)"), and every later occurrence that
re-expands it.

Usage: python scripts/x2_acronym_audit.py [--supp]
"""

import re
import sys

MAIN = 'paper/main_manuscript_bundle/main.tex'
SUPP = 'paper/main_manuscript_bundle/supplementary.tex'

# Abbreviations a reader of an industrial-informatics journal is not expected
# to expand unaided are checked; universally standard ones are exempt.
# Roman numerals are float numbers, not abbreviations. They are listed
# explicitly rather than matched by character class, because a rule that
# exempted every token built from roman-numeral letters would also exempt
# real acronyms such as MIX, DIM and MDL.
EXEMPT = {'CPU', 'GPU', 'RAM', 'IEEE', 'ID', 'II', 'III', 'IV', 'VI', 'VII',
          'VIII', 'IX', 'XI', 'XII', 'XIII', 'XIV', 'XV',
          'PDF', 'URL', 'DOI', 'AI', 'PPO', 'US', 'SI', 'NP',
          # proper names of published algorithms, products and datasets: the
          # citation, not an expansion, is what identifies them
          'DANIEL', 'QMIX', 'QTRAN', 'VDN', 'OR-Tools', 'PPVC-T', 'README',
          'BSTcontrol', 'IEEEtran', 'BG-MARL', 'EXHIBITTEXT'}
TOKEN = re.compile(r'\b([A-Z][A-Za-z]*[A-Z][A-Za-z]*(?:-[A-Z][A-Za-z-]*)*)\b')


def strip_tex(txt):
    txt = re.sub(r'(?m)^\s*%.*$', '', txt)
    # the running head is page furniture, not prose the reader parses in
    # order, so its space-saving short forms are not first uses
    txt = re.sub(r'\\markboth\{[^}]*\}\s*%?\s*\{[^}]*\}', ' ', txt)
    txt = re.sub(r'\\url\{[^}]*\}', ' ', txt)   # repository paths are not prose
    txt = re.sub(r'(?<!\\)%.*$', '', txt, flags=re.M)
    # keep a marker for exhibit text: a caption or a table note is read on
    # its own, so an expansion repeated there is correct rather than a fault
    txt = re.sub(r'\\caption\s*\{', ' EXHIBITTEXT ', txt)
    txt = re.sub(r'\\item\s*\[', ' EXHIBITTEXT [', txt)
    txt = re.sub(r'\\(cite[A-Za-z]*|ref|label|eqref|input|includegraphics)'
                 r'\s*(\[[^\]]*\])?\{[^}]*\}', ' ', txt)
    # macro invocations are not prose: \ScMixEighty is a number, not an
    # abbreviation the reader must expand
    txt = re.sub(r'\\[A-Za-z@]+\*?', ' ', txt)
    return txt


def read_order(supp=False):
    """(name, text) for each file, in the order the document reads them."""
    main = open(MAIN).read()
    out = []
    body = main
    if '\\begin{abstract}' in main:
        i = main.index('\\begin{abstract}')
        out.append(('title+abstract', main[:main.index('\\end{abstract}')]))
        body = main[i:]
    pos = 0
    for m in re.finditer(r'\\input\{(?:\.\./)?(sections/[a-z_]+|macros)\}', body):
        name = m.group(1)
        if name == 'macros':
            continue
        out.append((name, open(f'paper/{name}.tex').read()))
        pos = m.end()
    out.append(('main.tex tail', body[pos:]))
    if supp:
        out.append(('supplementary', open(SUPP).read()))
    return out


def has_expansion(tok, ctx):
    """True when the long form sits next to the abbreviation in ctx.

    An expansion is a run of words whose initials spell the abbreviation, in
    order, within eight words on either side. Hyphenated compounds count one
    initial per part, so "generalized-advantage-estimation" expands GAE, and
    filler words the abbreviation skips ("of", "the", "and") are allowed
    inside the run.
    """
    # the reader meets one abbreviation whether it is printed singular or
    # plural, so AGVs and AGV are the same token here
    bare = tok[:-1] if tok[-1] == 's' and tok[-2:].isupper() is False else tok
    if re.search(r'(?:[a-z][a-z-]+[\s,]+){1,}\(' + re.escape(bare) + r's?\)',
                 ctx):
        return True                      # long form (ABBR)
    if re.search(re.escape(bare) + r's?\s*\((?:[a-z][a-z-]+[\s,]+){1,7}',
                 ctx):
        return True                      # ABBR (long form)
    tok = bare
    letters = [c.lower() for c in tok if c.isalpha()]
    # two-letter tokens match too many word pairs by accident, so only the
    # explicit parenthetical forms above count for them
    if len(letters) < 3:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", ctx)
    FILLER = {'of', 'the', 'and', 'for', 'with', 'a', 'in', 'to'}
    hit = tok.lower().replace('-', '')
    for i, w in enumerate(words):
        if w == tok:
            continue
        for span in (words[max(0, i - 8):i], words[i + 1:i + 9]):
            got, k = [], 0
            for w2 in span:
                if w2 == tok:
                    continue
                parts = [p for p in w2.split('-') if p]
                if all(p.lower() in FILLER for p in parts) and got:
                    continue
                got += [p[0].lower() for p in parts]
                k += 1
                if len(got) >= len(letters):
                    break
            for start in range(0, max(1, len(got) - len(letters) + 1)):
                if ''.join(got[start:start + len(letters)]) == hit:
                    return True
    return False


def main():
    supp = '--supp' in sys.argv
    seen = {}
    reexpanded = []
    for order, (fname, raw) in enumerate(read_order(supp)):
        txt = strip_tex(raw)
        for m in TOKEN.finditer(txt):
            tok = m.group(1)
            if tok in EXEMPT or len(tok) < 2:
                continue
            ctx = txt[max(0, m.start() - 160):m.end() + 160]
            ctx = ' '.join(ctx.split())
            expanded = has_expansion(tok, ctx)
            # a caption or table note is read on its own, out of document
            # order, so repeating the long form there is correct
            standalone = 'EXHIBITTEXT' in ctx
            key = tok[:-1] if tok.endswith('s') and tok[:-2].isupper() else tok
            if key not in seen:
                seen[key] = (order, fname, expanded, ctx, m.start())
            elif (expanded and seen[key][2] and not standalone
                  and (seen[key][1] != fname
                       or m.start() - seen[key][4] > 300)):
                reexpanded.append((tok, fname, ctx))

    print('first occurrence of each abbreviation, in reading order')
    print('=' * 72)
    bad = []
    for tok, (order, fname, expanded, ctx, _pos) in sorted(
            seen.items(), key=lambda kv: (kv[1][0], kv[0])):
        flag = 'expanded' if expanded else 'BARE'
        if not expanded:
            bad.append((tok, fname, ctx))
        print(f'{flag:9s} {tok:12s} {fname}')
    if bad:
        print('\nfirst use without an expansion')
        print('=' * 72)
        for tok, fname, ctx in bad:
            print(f'\n{tok}  ({fname})\n  ...{ctx[:220]}...')
    if reexpanded:
        print('\nre-expanded after the first use (drop the long form here)')
        print('=' * 72)
        for tok, fname, ctx in reexpanded:
            print(f'\n{tok}  ({fname})\n  ...{ctx[:200]}...')
    return 1 if bad or reexpanded else 0


if __name__ == '__main__':
    sys.exit(main())
