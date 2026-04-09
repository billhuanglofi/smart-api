"""Content filter to remove profanity and offensive language from responses."""

from typing import Optional

from better_profanity import profanity

# Load the default word list and censor aggressively
profanity.load_censor_words()


def filter_text(text: Optional[str]) -> Optional[str]:
    """Return *text* with any profane words replaced by asterisks.

    The filter uses the `better-profanity` library's default word list.
    Every offensive word is replaced with a '*'-padded placeholder of the
    same length so the rest of the content remains intact.
    """
    if not text:
        return text
    return profanity.censor(text)


def contains_profanity(text: Optional[str]) -> bool:
    """Return True when *text* contains at least one profane word."""
    return profanity.contains_profanity(text)
