# Replicating and extending "What's Mine Becomes Yours" (context-dependent paraphrase detection)

COMP8240 Applications of Data Science, major project (novel project), final report code and outputs.

This repository replicates part of

> Anna Wegmann, Tijs van den Broek, Dong Nguyen. *What's Mine Becomes Yours: Defining, Annotating and Detecting Context-Dependent Paraphrases in News Interview Dialogs.* EMNLP 2024. [paper](https://aclanthology.org/2024.emnlp-main.52/) | [authors' code](https://github.com/nlpsoc/Paraphrases-in-News-Interviews) | [models on Hugging Face](https://huggingface.co/AnnaWegmann)

and extends it in three ways:

1. **Replication** of the DeBERTa-v3-large token classifiers (ALL and AGGREGATED) on the authors' test set, and of GPT-4 in-context learning. [Replication of GPT-4 ICL: describe in one line.]
2. **New data**: an existing podcast transcript (Joe Rogan with Elon Musk) and a transcript I constructed (Barack Obama on *Late Night with Conan O'Brien*), both annotated by me.
3. **LLM-as-judge** evaluation of the classifier's predictions on the new data.

The report is at [link or "submitted separately"].

## Repository layout

```
README.md
requirements_used.txt              packages from my virtual environment (pip freeze)
replication/
  logs/                            terminal output of the replication runs
  gpt4_icl/                        GPT-4 in-context learning scripts and outputs
data/
  rogan_musk/                      original + cleaned transcript, change log
  obama_conan/                     cleaned transcript
scripts/                           all scripts written for this project
outputs/
  predictions/                     DeBERTa predictions on the new data
  annotations/                     my labelled annotation sheets (+ Rogan answer key)
  judge/                           LLM-as-judge outputs and score printouts
```

## Setup

I ran everything on Windows, CPU only, with Python 3.11.9 in a virtual environment.

1. Download the authors' repository (not included here) from
   https://github.com/nlpsoc/Paraphrases-in-News-Interviews [downloaded on DATE as a ZIP] and unzip it.
2. Create and activate a virtual environment, then install PyTorch 2.2.1 (CPU build) first. The authors' `requirements.txt` pins `torch~=2.2.1+cpu`, which pip could not resolve, so comment that line out and install torch separately:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
3. Install what my scripts need:
   ```
   pip install wordninja==2.0.0 pyspellchecker==0.9.1 openai
   ```
4. Copy my files from `scripts/` into the authors' `src/` folder, together with the input files each script reads (listed below). **Run every script from `src/`**, because they import the authors' `paraphrase` package and read their inputs from the current folder.
5. For the LLM scripts, set your OpenAI key as an environment variable (never write it into a file):
   ```
   set OPENAI_API_KEY=your-key-here
   ```

## How to reproduce

Run from the authors' `src/` folder.

**1. Replication of the DeBERTa classifiers** (authors' script, 92 test pairs)

```
python eval_token_classifier.py --model_path AnnaWegmann/Highlight-Paraphrases-in-Dialog-ALL --dataset test
python eval_token_classifier.py --model_path AnnaWegmann/Highlight-Paraphrases-in-Dialog --dataset test
```
Output of my runs: `replication/logs/`.

**2. Replication of GPT-4 in-context learning**: [commands and files in `replication/gpt4_icl/`].

**3. Clean the Rogan transcript** (fixes words run together, e.g. `hobbycompany`)

```
python clean_transcript.py
```
Reads `joe-rogan-experience-1169-elon-musk.csv`; writes the `_CLEANED.csv` and `change_log.txt`.

**4. Run the classifier on a transcript**

```
python run_rogan.py
```
Settings are at the top of the script (`CSV_PATH`, `OUTPUT_FILE`, `GUEST_NAME`, `HOST_NAME`). It merges consecutive turns by the same speaker, pairs each guest turn with the host turn that follows, and writes a predictions TSV. For the Obama transcript set `CSV_PATH` to the Obama CSV, `GUEST_NAME = "Barack Obama"`, `HOST_NAME = "Conan O'Brien"`, and open the file with `utf-8-sig`.

**5. Make annotation sheets** (blank sheets without model output, plus an answer key)

```
python make_annotation_sheet.py
python make_annotation_sheet_conan.py
```
Warning: re-running these overwrites the sheet in the folder and would erase labels. My labelled sheets are in `outputs/annotations/`.

**6. LLM judge experiments** (`MODEL_NAME` is set at the top of each script; I used [gpt-4o])

Each script starts with a 3-pair test run (`ONLY_FIRST_N = 3`). Set it to `None` for the full run.

```
python llm_judge.py                    # LLM as independent classifier (my own prompt)
python llm_judge_v2.py                 # LLM as independent classifier (authors' prompt)
python llm_judge_predictions.py        # LLM judges DeBERTa's predictions (first prompt)
python run_controls_only.py            # control check only (first prompt)
python judge_v2.py                     # evidence-first prompt; set MODE = "controls" or "all"
```

**7. Score the results**

```
python score_judge.py                                 # classifier-style outputs
python score_judge_predictions.py                     # judge of predictions (first prompt)
python score_judge_predictions.py judge_v2_pred_outputs.tsv   # evidence-first prompt
python list_disagreements.py                          # optional: lists pairs where DeBERTa and I differ
```

The labelled sheets, answer key and predictions must be in the same folder. The names the scripts expect are set in the SETTINGS block of `llm_judge.py`:
`annotation_sheet_rogan_labelled.tsv`, `annotation_sheet_conan_labelled.tsv`, `answer_key.tsv` and `conan_obama_predictions.tsv`.

## Scripts

| Script | Purpose |
|---|---|
| `clean_transcript.py` | splits run-together words in the Rogan transcript, writes a change log |
| `run_rogan.py` | builds guest-to-host pairs and runs the DeBERTa classifier |
| `make_annotation_sheet.py`, `make_annotation_sheet_conan.py` | blank annotation sheets and answer keys |
| `llm_judge.py` | loads the 79 labelled pairs; LLM classifies them itself (my prompt). Other scripts import it, so do not rename |
| `llm_judge_v2.py` | same, with the authors' GPT-4 prompt and eight examples |
| `llm_judge_predictions.py` | LLM-as-judge: is DeBERTa's prediction acceptable? Includes wrong-prediction controls |
| `run_controls_only.py` | re-runs only the controls on all 26 eligible pairs |
| `judge_v2.py` | evidence-first judge prompt; controls only or all pairs |
| `score_judge.py`, `score_judge_predictions.py` | scoring and summaries |
| `list_disagreements.py` | table of pairs where DeBERTa and I disagree |

## Output files

Files in `outputs/judge/` were renamed after the runs. Scripts write the default name on the left.

| Script default name | Name in this repository |
|---|---|
| `judge_outputs.tsv` | `classifier_prompt_v1_outputs.tsv` |
| `judge_pred_outputs.tsv` | `predictions_prompt_v1_outputs.tsv` |
| `judge_controls_outputs.tsv` | `controls_prompt_v1_outputs.tsv` |
| `judge_v2_pred_outputs.tsv` | `predictions_prompt_v2_outputs.tsv` |
| `judge_v2_controls_outputs.tsv` | `controls_prompt_v2_outputs.tsv` |

`outputs/judge/score_printouts.txt` collects the scoring printouts from each experiment.

## Key results

**Replication on the authors' 92-pair test set** (the paper states 93)

| Model | F1 | P | R | Jaccard guest / host |
|---|---|---|---|---|
| DeBERTa ALL, this replication | 0.72 | 0.84 | 0.63 | 0.49 / 0.67 |
| DeBERTa ALL, paper (seed 201, footnote) | 0.72 | 0.84 | 0.63 | not reported per seed |
| DeBERTa AGGREGATED, this replication | 0.76 | 0.69 | 0.86 | 0.55 / 0.64 |
| DeBERTa AGGREGATED, paper (seed 202) | 0.76 | 0.72 (footnote) | 0.84 (footnote), 0.86 (text) | not reported per seed |

**DeBERTa ALL on the new data** (my labels)

| Data | Pairs | P | R | F1 |
|---|---|---|---|---|
| Rogan/Musk, annotated sample (stratified, recall inflated) | 40 | 0.80 | 0.76 | 0.78 |
| Obama/Conan | 39 | 1.00 | 0.53 | 0.69 |
| All annotated pairs | 79 | 0.87 | 0.65 | 0.74 |

**LLM-as-judge** (79 pairs): the first prompt rejected 14 of 26 deliberately wrong "no paraphrase" controls (54%); the evidence-first prompt rejected 21 of 26 (81%). With the second prompt the judge accepted 11 of 18 DeBERTa predictions that differ from my labels. Treating those as correct raises F1 from 0.74 to 0.89, which I treat as an upper bound only.

## Notes and limitations

- I am the only annotator, so there is no agreement between humans. The Rogan sample is stratified (20 model-flagged and 20 unflagged pairs).
- [After reviewing the judge's reasoning I changed N labels; the results above use my original labels.]
- The change log lists the 460 automatic fixes to the Rogan transcript. Four more fixes were made by hand and are listed in `MANUAL_FIXES` in `clean_transcript.py`.
- The authors' dataset (research-only licence) and code are not redistributed here; download them from the links above. The two transcripts are included for use in this assessment, from [source URLs].
- [Optional: AI use statement, e.g. "I used Claude for help with scripts and drafting; I checked and ran all code and take responsibility for the outputs."]

## Author

[Your name], [university], COMP8240, 2026 Session 2.
