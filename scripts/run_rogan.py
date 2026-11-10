#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
from paraphrase import token_classifier

#important things
CSV_PATH = r"C:\Users\sivani\Documents\COMP 8240\joe-rogan-experience-1169-elon-musk_CLEANED.csv"
OUTPUT_FILE = "rogan_musk_predictions.tsv"
MODEL_NAME = "AnnaWegmann/Highlight-Paraphrases-in-Dialog-ALL"
BATCH_SIZE = 8 #change             
GUEST_NAME = "Elon Musk"
HOST_NAME = "Joe Rogan"


#read CSV
rows = []
with open(CSV_PATH, encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        rows.append(row)
print("Rows read from CSV:", len(rows))


# In[ ]:


#merge consecutive lines by same speaker
#each "turn" = one speaker talking until other speaker replies
turns = []
for row in rows:
    speaker = row["Speaker"]
    text = row["Text"].strip()

    if len(turns) > 0 and turns[-1]["speaker"] == speaker:
        #same speaker as the previous row, so adding the text to the previous turn
        turns[-1]["text"] = turns[-1]["text"] + " " + text
    else:
        #a new speaker, so starting a new turn
        turns.append({"speaker": speaker, "time": row["Timestamp"], "text": text})
print("Turns after merging:", len(turns))


# building guest -> host pairs
#A pair is: an Elon turn followed directly by a Joe turn
pairs = []
for i in range(len(turns) - 1):
    current_turn = turns[i]
    next_turn = turns[i + 1]

    if current_turn["speaker"] == GUEST_NAME and next_turn["speaker"] == HOST_NAME:
        guest_words = current_turn["text"].split()   # split the text into a list of words
        host_words = next_turn["text"].split()

        if len(guest_words) > 0 and len(host_words) > 0:
            pairs.append({
                "time": current_turn["time"],
                "guest_words": guest_words,
                "host_words": host_words,
            })
print("Guest->host pairs built:", len(pairs))


# In[ ]:


# loading the model
classifier = token_classifier.TokenClassifier(MODEL_NAME)

#need to run model in small batches
all_results = []   # to hold (pair, labels, probabilities) for every pair

start = 0
while start < len(pairs):
    end = start + BATCH_SIZE
    batch = pairs[start:end]

    #the annnaweggman model expects a list of dictionaries with "sentence1" and "sentence2"
    model_input = []
    for pair in batch:
        model_input.append({
            "sentence1": pair["guest_words"],
            "sentence2": pair["host_words"],
        })

    labels, probabilities = classifier.inference(model_input, return_prob=True, pred_thresh=0.5)

    # save results for each pair in this batch
    for j in range(len(batch)):
        all_results.append((batch[j], labels[j], probabilities[j]))

    start = end
    print("Processed", min(start, len(pairs)), "of", len(pairs))


#write the results to a table
paraphrase_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow([
        "pair_id", "time", "guest_text", "host_text", "model_paraphrase",
        "guest_highlight", "host_highlight", "max_prob_guest", "max_prob_host",
        "human_label",
    ])

    pair_id = 0
    for pair, labels, probabilities in all_results:
        guest_labels = labels[0]       #one 0/1 per guest word
        host_labels = labels[1]        #one 0/1 per host word

        #according to paper, a paraphrase = at least one highlighted word on both sides
        if 1 in guest_labels and 1 in host_labels:
            is_paraphrase = 1
            paraphrase_count = paraphrase_count + 1
        else:
            is_paraphrase = 0

        #collecting highlighted words
        guest_highlight_words = []
        for k in range(len(pair["guest_words"])):
            if guest_labels[k] == 1:
                guest_highlight_words.append(pair["guest_words"][k])

        host_highlight_words = []
        for k in range(len(pair["host_words"])):
            if host_labels[k] == 1:
                host_highlight_words.append(pair["host_words"][k])

        writer.writerow([
            pair_id,
            pair["time"],
            " ".join(pair["guest_words"]),
            " ".join(pair["host_words"]),
            is_paraphrase,
            " ".join(guest_highlight_words),
            " ".join(host_highlight_words),
            round(max(probabilities[0]), 3),
            round(max(probabilities[1]), 3),
            "",                         # this for human_label
        ])
        pair_id = pair_id + 1

print("Done!", paraphrase_count, "of", len(all_results), "pairs predicted as paraphrases.")
print("Results saved to", OUTPUT_FILE)

