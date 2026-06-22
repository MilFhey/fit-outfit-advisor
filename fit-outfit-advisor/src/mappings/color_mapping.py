NEUTRAL_COLORS = {"noir", "blanc", "gris", "beige", "marron", "bleu marine"}
WARM_COLORS = {"rouge", "orange", "jaune", "bordeaux", "camel"}
COLD_COLORS = {"bleu", "vert", "violet"}


COLOR_COMPATIBILITY = {
    "noir": ["blanc", "gris", "beige", "rouge", "bordeaux", "bleu"],
    "blanc": ["noir", "bleu", "beige", "vert", "rouge"],
    "bleu": ["blanc", "beige", "gris", "marron"],
    "rouge": ["noir", "blanc", "beige", "gris"],
    "beige": ["blanc", "noir", "marron", "bleu", "bordeaux"],
    "vert": ["blanc", "beige", "marron", "noir"],
    "bordeaux": ["noir", "beige", "blanc", "gris"],
}


def get_compatible_colors(color: str) -> list[str]:
    """Retourne des couleurs compatibles simples pour le MVP."""
    return COLOR_COMPATIBILITY.get(color.lower(), ["noir", "blanc", "beige"])


def is_neutral(color: str) -> bool:
    return color.lower() in NEUTRAL_COLORS
