"""
llm_judge_predictions.py  --  LLM-AS-JUDGE of DeBERTa's predictions (proposal section 2.2).

Unlike llm_judge.py / llm_judge_v2.py (which make their OWN classification), here the LLM is shown
a guest/host pair PLUS DeBERTa's prediction (label and highlighted words) and is asked:
    1. label_acceptable      -- is the paraphrase / no-paraphrase decision reasonable?
    2. highlights_acceptable -- are the highlighted words reasonable? (only if DeBERTa said paraphrase)
The judge never sees the human label.

CONTROL: to check the judge is not just agreeing with everything, a few pairs where DeBERTa and
I both said "paraphrase" are shown a deliberately WRONG prediction ("not a paraphrase").
A good judge should reject those.

TO RUN (in `src`, next to llm_judge.py; OPENAI_API_KEY = my-key):
    python llm_judge_predictions.py                 (first run = 3-pair test)
    python score_judge_predictions.py
"""

import csv
import json
import random
import time
from collections import Counter

import llm_judge as v1                 #use the code that loads 79 pairs
from paraphrase.prompt_templates import FEW_SHOT_EXAMPLES   # the authors' worked examples

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
OUTPUT_FILE = "judge_pred_outputs.tsv"
MODEL_NAME = "gpt-4o"
NUMBER_OF_RUNS = 3
TEMPERATURE = 1.0
CONTROL_N = 10          # number of deliberately-wrong control predictions (0 = none)
RANDOM_SEED = 42
ONLY_FIRST_N = None        

INSTRUCTIONS = """You are an expert linguistic annotator checking the output of an automatic system.

The system reads a GUEST turn and the HOST's next turn from a news interview and decides whether the
host paraphrases something specific the guest says, and if so which words. A paraphrase is a rewording
or repetition of content in the guest's statement: it rephrases what the guest said. It is NOT a
paraphrase if the host only continues on a related topic, reacts, draws a conclusion, or asks a new
question without rewording or repeating content of the guest's statement.

You are shown the pair and the system's prediction. Judge the PREDICTION:
1. label_acceptable: could a careful annotator following this definition reasonably give the same
   label (paraphrase / no paraphrase)? True if the label is correct, or defensible because the pair is
   genuinely ambiguous. False if it is clearly wrong.
2. highlights_acceptable: only if the system said paraphrase. Are the highlighted guest words and host
   words the words that carry the paraphrased content (not wildly too narrow, too broad or unrelated)?
   Use null if the system said no paraphrase.
Do not assume the system is right. Decide from the text itself.

Reply with JSON only, exactly these keys:
{"explanation": "one or two sentences", "label_acceptable": true or false, "highlights_acceptable": true, false or null}"""

# Worked examples built from the authors' own examples (indices into FEW_SHOT_EXAMPLES).
# kind: "ok" = good prediction, "wrong_label" = wrong label, "wrong_hl" = right label but bad highlights.
CALIBRATION = [
    {"ex": 0, "pred": ("Yes", "I don't even know what it is", "you weren't even familiar?"),
     "answer": {"explanation": "Both say the guest did not know Fortnite, and the highlighted words capture exactly that.",
                "label_acceptable": True, "highlights_acceptable": True}},
    {"ex": 4, "pred": ("No", "", ""),
     "answer": {"explanation": "The host asks about what people on the ground think; he does not reword the guest's question, so no paraphrase is right.",
                "label_acceptable": True, "highlights_acceptable": None}},
    {"ex": 2, "pred": ("Yes", "no constitutional protections for you", "you don't have a problem with Facebook giving the government access"),
     "answer": {"explanation": "The host draws a new conclusion about Facebook without rewording or repeating what the guest said, so calling it a paraphrase is wrong.",
                "label_acceptable": False, "highlights_acceptable": False}},
    {"ex": 0, "pred": ("Yes", "season 5 when they premiered it in the game", "So"),
     "answer": {"explanation": "The label is right (the host repeats that the guest did not know the game), but the highlighted words are about something else.",
                "label_acceptable": True, "highlights_acceptable": False}},
    {"ex": 1, "pred": ("No", "", ""),
     "answer": {"explanation": "Both say the Republicans want to negotiate ('we would like ... to negotiate' vs 'you want to negotiate'), so saying no paraphrase is wrong.",
                "label_acceptable": False, "highlights_acceptable": None}},
]


def format_prediction(label_word, guest_words, host_words):
    text = "SYSTEM PREDICTION:\nParaphrase: " + label_word
    if label_word == "Yes":
        text += '\nGuest words: "' + guest_words + '"\nHost words: "' + host_words + '"'
    return text


def make_user_message(guest_text, host_text, label_word, guest_words, host_words):
    return ("GUEST: " + guest_text + "\nHOST: " + host_text + "\n\n"
            + format_prediction(label_word, guest_words, host_words))


def build_messages(guest_text, host_text, label_word, guest_words, host_words):
    messages = [{"role": "system", "content": INSTRUCTIONS}]
    for c in CALIBRATION:
        ex = FEW_SHOT_EXAMPLES[c["ex"]]
        messages.append({"role": "user", "content": make_user_message(
            ex["guest utterance"], ex["host utterance"], *c["pred"])})
        messages.append({"role": "assistant", "content": json.dumps(c["answer"])})
    messages.append({"role": "user", "content": make_user_message(
        guest_text, host_text, label_word, guest_words, host_words)})
    return messages


# ---------------------------------------------------------------
# Ask the judge
# ---------------------------------------------------------------
def ask_once(client, messages):
    try:
        reply = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=TEMPERATURE,
            response_format={"type": "json_object"})
    except Exception as err:
        if "temperature" in str(err).lower():
            reply = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, response_format={"type": "json_object"})
        else:
            raise
    return reply.choices[0].message.content


def to_bool(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def parse_reply(text):
    """Returns (label_acceptable, highlights_acceptable_or_None, explanation) or None."""
    try:
        data = json.loads(text)
        label_ok = to_bool(data["label_acceptable"])
        if label_ok is None:
            return None
        return label_ok, to_bool(data.get("highlights_acceptable")), str(data.get("explanation", ""))
    except Exception:
        return None


def majority(values):
    """True/False majority; a tie counts as False. Returns None if there are no values."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) > len(values) / 2


def judge_one(client, guest_text, host_text, label_word, guest_words, host_words):
    messages = build_messages(guest_text, host_text, label_word, guest_words, host_words)
    parsed, raws = [], []
    for _ in range(NUMBER_OF_RUNS):
        try:
            raw = ask_once(client, messages)
        except Exception as err:
            raw = "API_ERROR: " + str(err)
            time.sleep(2)
        raws.append(raw.replace("\t", " ").replace("\n", " "))
        result = parse_reply(raw)
        if result is not None:
            parsed.append(result)
    if not parsed:
        return {"label_acceptable": "", "highlights_acceptable": "", "votes_label_acceptable": 0,
                "runs_parsed": 0, "explanation": "", "raw": raws}
    label_ok = majority([p[0] for p in parsed])
    hl_ok = majority([p[1] for p in parsed]) if label_word == "Yes" else None
    return {"label_acceptable": int(label_ok),
            "highlights_acceptable": "" if hl_ok is None else int(hl_ok),
            "votes_label_acceptable": sum(1 for p in parsed if p[0]),
            "runs_parsed": len(parsed), "explanation": parsed[0][2], "raw": raws}


# ---------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------
def main():
    from openai import OpenAI
    client = OpenAI()

    pairs = v1.load_rogan() + v1.load_obama()
    pairs = [p for p in pairs if p["human_label"] in ("0", "1") and p["model_label"] in ("0", "1")]
    print("Loaded", len(pairs), "pairs that have both a human label and a DeBERTa prediction.")
    if not pairs:
        print("STOP: nothing to judge. Check the file names in llm_judge.py.")
        return

    # items to judge: every real prediction, plus the deliberately wrong controls
    items = [dict(p, is_control=0, shown_label=int(p["model_label"]),
                  shown_guest=p["model_guest_highlight"], shown_host=p["model_host_highlight"]) for p in pairs]
    both_yes = [p for p in pairs if p["human_label"] == "1" and p["model_label"] == "1"]
    random.Random(RANDOM_SEED).shuffle(both_yes)
    for p in both_yes[:CONTROL_N]:
        items.append(dict(p, is_control=1, shown_label=0, shown_guest="", shown_host=""))
    random.Random(RANDOM_SEED).shuffle(items)
    if ONLY_FIRST_N:
        items = items[:ONLY_FIRST_N]
        print("TEST RUN: judging only", len(items), "items.")

    columns = ["dataset", "pair_id", "is_control", "guest_text", "host_text",
               "human_label", "human_guest_highlight", "human_host_highlight",
               "shown_label", "shown_guest_highlight", "shown_host_highlight",
               "label_acceptable", "highlights_acceptable", "votes_label_acceptable",
               "runs_parsed", "explanation", "raw_responses"]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for i, it in enumerate(items, start=1):
            label_word = "Yes" if it["shown_label"] == 1 else "No"
            res = judge_one(client, it["guest_text"], it["host_text"], label_word,
                            it["shown_guest"], it["shown_host"])
            row = dict(it)
            row.update(res)
            row["shown_guest_highlight"], row["shown_host_highlight"] = it["shown_guest"], it["shown_host"]
            row["raw_responses"] = " ||| ".join(res["raw"])
            writer.writerow([str(row.get(c, "")).replace("\t", " ").replace("\n", " ") for c in columns])
            print(i, "/", len(items), it["dataset"], it["pair_id"], "control" if it["is_control"] else "real",
                  "-> label acceptable:", res["label_acceptable"])
    print("Done! Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
