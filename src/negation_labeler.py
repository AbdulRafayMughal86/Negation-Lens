import re

EXPLICIT_KEYWORDS = ["not", "no", "never", "cannot"]
EXPLICIT_SUFFIX = "n't"
IMPLICIT_KEYWORDS = ["without", "except", "lack", "lacking"]
COMPARATIVE_PHRASES = ["less than", "more than", "fewer than", "greater than", "at least", "at most"]


def label_negation_type(query_text):
    lowered = query_text.lower()
    words = set(re.findall(r"\b\w+\b", lowered))

    for phrase in COMPARATIVE_PHRASES:
        if phrase in lowered:
            return "comparative"

    if EXPLICIT_SUFFIX in lowered:
        return "explicit"

    for keyword in EXPLICIT_KEYWORDS:
        if keyword in words:
            return "explicit"

    for keyword in IMPLICIT_KEYWORDS:
        if keyword in words:
            return "implicit"

    return "other"
