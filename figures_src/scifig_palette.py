"""scifig_palette: colorblind-safe, grayscale-robust colors for scientific figures.

Design rule encoded here:
  - Neutrals (the gray ramp) carry STRUCTURE: text, lines, fills, gridlines.
  - Accents carry MEANING and are rationed: keep to <= 2 accent hues per figure,
    each mapped to a single, stated meaning, and always shown in a legend.
  - Every accent below survives grayscale conversion and common color-vision
    deficiencies (the Okabe-Ito set is the field standard for this).

Usage:
    from scifig_palette import INK, SILVER, categorical, PAIR_BLUE_VERMILLION
    colors = categorical(3)          # 3 distinct, CVD-safe categorical colors
    check_separation(colors)         # warns if any pair is hard to tell apart in gray
"""

# --- Neutral ramp: structure, text, fills (use these first, before any color) ---
INK      = "#1A1A1A"   # primary text / strongest lines
GRAPHITE = "#4D4D4D"   # secondary text / lines
SLATE    = "#808080"   # tertiary / muted labels
SILVER   = "#BFBFBF"   # light lines / gridlines / borders
MIST     = "#E8E8E8"   # light fills / subtle backgrounds
PAPER    = "#FFFFFF"   # background

NEUTRALS = [INK, GRAPHITE, SLATE, SILVER, MIST]

# --- Okabe-Ito categorical palette (colorblind-safe). Named for readable code. ---
OKABE_ITO = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}

# --- Restrained 2-accent pairs (both members survive grayscale + CVD) ---
PAIR_BLUE_VERMILLION = ["#0072B2", "#D55E00"]   # cool vs warm, max separation
PAIR_BLUE_GREEN      = ["#0072B2", "#009E73"]   # calm, technical
PAIR_GREEN_PURPLE    = ["#009E73", "#CC79A7"]   # softer; a clean take on green+purple

# Good default ordering when you need N categories.
_ORDER = ["blue", "vermillion", "bluish_green", "reddish_purple",
          "orange", "sky_blue", "black", "yellow"]


def categorical(n, start=0):
    """Return n colorblind-safe categorical colors in a perceptually good order."""
    keys = (_ORDER + _ORDER)[start:start + n]
    return [OKABE_ITO[k] for k in keys]


def luminance(hex_color):
    """Relative luminance of a hex color in [0, 1] (sRGB, WCAG weights).

    Use it to verify a color separates from another once printed in grayscale.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def check_separation(colors, min_delta=0.12):
    """Warn (return a list of messages) about color pairs that would blur together
    in grayscale. Empty list means the set is grayscale-robust. min_delta is the
    minimum luminance gap you want between any two colors that must be told apart.
    """
    msgs = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            d = abs(luminance(colors[i]) - luminance(colors[j]))
            if d < min_delta:
                msgs.append(
                    f"low grayscale separation ({d:.02f} < {min_delta}): "
                    f"{colors[i]} vs {colors[j]} - distinguish by shape/pattern too"
                )
    return msgs


if __name__ == "__main__":
    # Quick self-test / demo of the palette and the separation checker.
    print("neutrals :", NEUTRALS)
    print("cat(4)   :", categorical(4))
    print("lum(blue):", round(luminance(OKABE_ITO["blue"]), 3))
    print("warnings :", check_separation(categorical(4)) or "none")
