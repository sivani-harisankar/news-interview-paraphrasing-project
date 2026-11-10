"""
clean_transcript.py  --  Fix words that were run together in a transcript CSV
                          (e.g. "hobbycompany" -> "hobby company", "allCalifornia" -> "all California").

WHAT IT DOES
  1. Reads the transcript CSV (columns: Timestamp, Speaker, Text).
  2. Goes through every word in the Text column. A word is only touched if
       - it is not a normal English word (checked with pyspellchecker), AND
       - it is at least 5 letters long, AND
       - it contains only letters (words with apostrophes or hyphens are left alone), AND
       - it is not in the PROTECTED list (names and brand names such as Hyperloop).
  3. Two kinds of fix are made:
       - "camel":    allCalifornia -> all California  (only when the first piece is a small common word)
       - "wordninja": hobbycompany -> hobby company   (only when EVERY piece is a normal English word)
  4. Applies a short list of hand-made fixes (MANUAL_FIXES) for the few words the rules cannot split safely.
  5. Writes the cleaned CSV and a change log (every automatic fix, plus the words it was unsure about).

TO RUN (Windows cmd):
    pip install wordninja==2.0.0 pyspellchecker==0.9.1
    python clean_transcript.py

NOTE: the spell checker's word list changes between versions, so a different pyspellchecker version
can give slightly different fixes. The versions above reproduce the cleaned file and change log
used in the report exactly.
"""

import csv
import re

import wordninja
from spellchecker import SpellChecker

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
INPUT_FILE = "joe-rogan-experience-1169-elon-musk.csv"
OUTPUT_FILE = "joe-rogan-experience-1169-elon-musk_CLEANED.csv"
LOG_FILE = "change_log.txt"
MIN_LETTERS = 5        # shorter words (e.g. "alot", "tobe") are left alone

# Names and brands that must never be split (they look like run-together words but are not).
PROTECTED = {"Fisker", "Hyperloop", "Musks", "Neuralink", "Rolexes", "Rogans", "Spaceballs"}

# Common words that may appear glued to the front of the next word: allCalifornia, AndGoogle ...
SMALL_WORDS = {"all", "and", "the", "like", "a", "of", "to", "in", "but", "so", "or", "is", "it", "i"}

# Hand-made fixes for words the rules cannot split safely (checked by eye).
MANUAL_FIXES = {
    "muchlimbic": "much limbic",
    "couldmag": "could mag",
    "withphotovoltaics": "with photovoltaics",
    "atleast,": "at least,",
}

spell = SpellChecker()

# A "plain word" = optional opening punctuation + letters only + optional closing punctuation.
# Tokens with apostrophes inside, hyphens, digits or hashtags do not match and are left alone.
PLAIN_WORD = re.compile(r'^(["\'(\[]*)([^\W\d_]+)([.,?!;:"\')\]\u2026]*)$')


def split_word(token):
    """Return (kind, new_token) if the token should be split, otherwise None."""
    # Separate leading/trailing punctuation from the letters: '"Youknow,' -> '"' + 'Youknow' + ','
    match = re.match(r'^(["\'(\[]*)([^\W\d_]+)([.,?!;:"\')\]…]*)$', token)
    if not match:
        return None
    before, word, after = match.groups()

    if len(word) < MIN_LETTERS or word in PROTECTED or word.lower() in spell:
        return None

    # Case 1: a small word glued to a capitalised one (allCalifornia, AndGoogle)
    if re.search(r"[a-z][A-Z]", word):
        pieces = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word).split(" ")
        if len(pieces) == 2 and pieces[0].lower() in SMALL_WORDS:
            return "camel", before + " ".join(pieces) + after
        return None

    # Case 2: two or more words glued together (hobbycompany)
    pieces = wordninja.split(word)
    if len(pieces) >= 2 and all(p.lower() in spell for p in pieces):
        return "wordninja", before + " ".join(pieces) + after
    return None


def main():
    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    fixes = []                # (kind, old, new) for every automatic fix
    unresolved = set()        # unknown words we did NOT change

    for row in rows:
        new_words = []
        for token in row["Text"].split(" "):
            result = split_word(token)
            if result:
                kind, new_token = result
                fixes.append((kind, token, new_token))
                new_words.append(new_token)
                continue
            # remember unknown long words that were left alone, so can check them by eye
            plain = PLAIN_WORD.match(token)
            if plain:
                letters = plain.group(2)
                # words with a capital letter inside (SpaceX, McGregor) are names, so not listed
                if (len(letters) >= MIN_LETTERS and letters not in PROTECTED
                        and not re.search(r"[a-z][A-Z]", letters) and letters.lower() not in spell):
                    unresolved.add(letters)
            new_words.append(token)
        row["Text"] = " ".join(MANUAL_FIXES.get(w, w) for w in new_words)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Speaker", "Text"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Total fixes: %d\n\n" % len(fixes))
        for kind, old, new in fixes:
            f.write("[%s] %s -> %s\n" % (kind, old, new))
        f.write("\n\nUnresolved (%d):\n" % len(unresolved))
        for word in sorted(unresolved):
            f.write(word + "\n")

    print("Done!", len(fixes), "automatic fixes,", len(MANUAL_FIXES), "manual fixes.")
    print("Cleaned file:", OUTPUT_FILE, "| log:", LOG_FILE)


if __name__ == "__main__":
    main()
