"""
judge_v2.py  --  LLM-as-judge with an EVIDENCE-FIRST prompt (second attempt after the first judge
accepted too many wrong "not a paraphrase" predictions in the control check).

What changed compared with llm_judge_predictions.py:
  * the judge must first QUOTE any words the host repeats or rewords from the guest ("evidence"),
    and only then give its verdict;
  * the instructions say explicitly that a host can repeat/reword part of the guest AND ALSO ask a
    new question or add more, and the repeated part still counts;
  * 7 worked examples (all taken from the authors' own examples, none from your pairs), including
    wrong "No" predictions where the host repeats or rewords the guest.

TO RUN (in `src`, next to llm_judge_predictions.py and run_controls_only.py):
    python judge_v2.py                 first with MODE = "controls" (and ONLY_FIRST_N = 3 as a test)
    python judge_v2.py                 then ONLY_FIRST_N = None; if the control looks good, MODE = "all"
    python score_judge_predictions.py judge_v2_pred_outputs.tsv      (only after MODE = "all")

IMPORTANT FOR REPORT: this prompt was written AFTER seeing the first control results, so the
controls are no longer a completely clean test of it. 
"""

import llm_judge_predictions as jp          # the first judge: reuse it and swap in a new prompt
import run_controls_only as rc              # the controls-only runner

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
MODEL_NAME = "gpt-4o"      
MODE = "all"          # "controls" = only the control check; "all" = judge all 79 pairs + 10 controls
ONLY_FIRST_N = None           

# ---------------------------------------------------------------
# THE NEW PROMPT
# ---------------------------------------------------------------
jp.INSTRUCTIONS = """You are an expert linguistic annotator checking the output of an automatic system.

The system reads a GUEST turn and the HOST's next turn from a news interview and decides whether the
host paraphrases something specific the guest says, and if so which words. A paraphrase is a rewording
or repetition of content in the guest's statement: it rephrases what the guest said.

Important details:
- Repeating the guest's words counts as a paraphrase (repetition of content), including when the host
  repeats them as a question or an acknowledgement.
- A host can repeat or reword PART of what the guest said and ALSO say something new or ask a new
  question. The repeated or reworded part still counts; the new material does not cancel it.
- It is NOT a paraphrase if the host only continues on a related topic, reacts, draws a new conclusion,
  or asks a new question without repeating or rewording any content of the guest's statement.

You are shown the pair and the system's prediction. Judge the PREDICTION. Work in this order:
1. evidence: quote the words in the HOST turn that repeat or reword content from the GUEST turn, next to
   the guest words they match. Write "none" if there are none.
2. explanation: one or two sentences.
3. label_acceptable: using your evidence, could a careful annotator following this definition reasonably
   give the same label (paraphrase / no paraphrase)? True if the label is correct, or defensible because
   the pair is genuinely ambiguous. False if it is clearly wrong. If your evidence shows real repetition
   or rewording, a "no paraphrase" label is wrong; if your evidence is "none", a "paraphrase" label is wrong.
4. highlights_acceptable: only if the system said paraphrase. Are the highlighted guest and host words the
   words that carry the paraphrased content? Use null if the system said no paraphrase.
Do not assume the system is right. Decide from the text itself.

Reply with JSON only, with exactly these keys in this order:
{"evidence": "...", "explanation": "...", "label_acceptable": true or false, "highlights_acceptable": true, false or null}"""


def shot(ex, pred, evidence, explanation, label_ok, hl_ok):
    return {"ex": ex, "pred": pred,
            "answer": {"evidence": evidence, "explanation": explanation,
                       "label_acceptable": label_ok, "highlights_acceptable": hl_ok}}


# indices refer to the authors' FEW_SHOT_EXAMPLES list (0 = Fortnite, 1 = Labrador, 2 = Rand Paul,
# 3 = Calabresi/Phillips, 4 = Estrin, 5 = Hanson/police)
jp.CALIBRATION = [
    shot(0, ("Yes", "I don't even know what it is", "you weren't even familiar?"),
         "Host 'you weren't even familiar?' rewords guest 'I don't even know what it is'.",
         "Both say the guest did not know Fortnite, and the highlighted words capture exactly that.",
         True, True),
    shot(2, ("Yes", "no constitutional protections for you", "you don't have a problem with Facebook giving the government access"),
         "none",
         "The host draws a new conclusion about Facebook without repeating or rewording what the guest said, so 'paraphrase' is wrong.",
         False, False),
    shot(4, ("No", "", ""),
         "none",
         "The host asks what people on the ground think; he does not repeat or reword the guest's question, so 'no paraphrase' is right.",
         True, None),
    shot(3, ("No", "", ""),
         "Host 'she talked a lot about her family and her kids' rewords guest 'Talked about family life'.",
         "The host rewords what the guest said about the conversation, so 'no paraphrase' is wrong.",
         False, None),
    shot(0, ("Yes", "season 5 when they premiered it in the game", "So"),
         "Host 'you weren't even familiar?' rewords guest 'I don't even know what it is'.",
         "The label is right, but the highlighted words are about something else than the repeated content.",
         True, False),
    shot(1, ("No", "", ""),
         "Host 'you want to negotiate' repeats guest 'We would like the senators to actually come and negotiate with us'.",
         "The host repeats that the Republicans want to negotiate and then asks a new question; the repeated part still counts, so 'no paraphrase' is wrong.",
         False, None),
    shot(5, ("No", "", ""),
         "none",
         "The host asks about the reporter's talks with the family while the guest talked about the police; no content is repeated or reworded, so 'no paraphrase' is right.",
         True, None),
]

# ---------------------------------------------------------------
# Hand the new settings to the existing scripts and run
# ---------------------------------------------------------------
jp.MODEL_NAME = MODEL_NAME
if MODE == "controls":
    rc.OUTPUT_FILE = "judge_v2_controls_outputs.tsv"
    rc.ONLY_FIRST_N = ONLY_FIRST_N
    runner = rc.main
else:
    jp.OUTPUT_FILE = "judge_v2_pred_outputs.tsv"
    jp.ONLY_FIRST_N = ONLY_FIRST_N
    runner = jp.main

if __name__ == "__main__":
    print("Judge model:", MODEL_NAME, "| mode:", MODE, "| prompt: evidence-first (v2)")
    runner()
