#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import random

INPUT_FILE = "rogan_musk_predictions.tsv"
SHEET_FILE = "annotation_sheet.tsv"      #file for human label
KEY_FILE = "answer_key.tsv"              #file with model answers
NUM_POSITIVES = 20                       # pairs paraphrases as denoted by model
NUM_NEGATIVES = 20                       # pair non paraphrases as denoted by model
RANDOM_SEED = 42                         #seed for re runs


# read results file
all_pairs = []
with open(INPUT_FILE, encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file, delimiter="\t")
    for row in reader:
        all_pairs.append(row)
print("Pairs read:", len(all_pairs))


# split into model positive and negative pairs
positive_pairs = []
negative_pairs = []
for pair in all_pairs:
    if pair["model_paraphrase"] == "1":
        positive_pairs.append(pair)
    else:
        negative_pairs.append(pair)
print("Model positives:", len(positive_pairs))
print("Model negatives:", len(negative_pairs))


#randomly pick some of each
random.seed(RANDOM_SEED)

chosen_positives = random.sample(positive_pairs, min(NUM_POSITIVES, len(positive_pairs)))
chosen_negatives = random.sample(negative_pairs, min(NUM_NEGATIVES, len(negative_pairs)))

#put them together and shuffle so positives and negatives are mixed up
sample = chosen_positives + chosen_negatives
random.shuffle(sample)
print("Pairs in annotation sample:", len(sample))


# In[ ]:


#write the annotation sheet without model answers
with open(SHEET_FILE, "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow([
        "pair_id", "time", "guest_text", "host_text",
        "human_label", "human_guest_highlight", "human_host_highlight",
    ])
    for pair in sample:
        writer.writerow([
            pair["pair_id"],
            pair["time"],
            pair["guest_text"],
            pair["host_text"],
            "",     # human_labels; type 1 (paraphrase) or 0 (not)
            "",     # human_guest_highlight: words I would highlight
            "",     # human_host_highlight: same as above for host
        ])


# write the answer key for model answers
with open(KEY_FILE, "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow([
        "pair_id", "model_paraphrase", "guest_highlight", "host_highlight",
        "max_prob_guest", "max_prob_host",
    ])
    for pair in sample:
        writer.writerow([
            pair["pair_id"],
            pair["model_paraphrase"],
            pair["guest_highlight"],
            pair["host_highlight"],
            pair["max_prob_guest"],
            pair["max_prob_host"],
        ])

print("Done!")
print("Label this file:", SHEET_FILE)
print("Keep this closed until finish:", KEY_FILE)

