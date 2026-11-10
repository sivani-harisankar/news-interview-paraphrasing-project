"""
   script to build few-shot prompt
        --> location of the generated prompts saved in INSTRUCT_PROMPT_DEV & INSTRUCT_PROMPT_TEST consts
"""

import json

from paraphrase.utility.PC_utility import get_qids_from_file
from paraphrase.annotation_data import TEST_PATH, DEV_PATH, TRAIN_PATH
from paraphrase.interview_data import MediaSumProcessor
from paraphrase.prompt_templates import FS_PROMPT_TEMPLATES, build_few_shot_prompt, FEW_SHOT_EXAMPLES
from paraphrase.utility.project_functions import get_dir_to_src

# location of the generated prompts for the dev and test set -- AFTER this script was called
INSTRUCT_PROMPT_DEV = get_dir_to_src() + "/../result/Models/0-8-dev_prompts.json"
INSTRUCT_PROMPT_TEST = get_dir_to_src() + "/../result/Models/0-8-test_prompts.json"


def main(shot=len(FEW_SHOT_EXAMPLES), dataset="dev", instruction_faithful=False):
    assert dataset in ["dev", "test", "train"], "dataset must be one of 'dev', 'test', or 'train'"
    if dataset == "dev":
        item_path = DEV_PATH
    elif dataset == "test":
        item_path = TEST_PATH
    else:
        item_path = TRAIN_PATH

    file_prefix = "0-"

    # the first element of FS_PROMPT_TEMPLATE is the one closest to the instructions shown to annotators
    prompt_template = FS_PROMPT_TEMPLATES[0]

    file_prefix += str(shot) + "-" + dataset + "_"

    # get the question IDs
    q_ids = get_qids_from_file(item_path)
    interview = MediaSumProcessor()
    q_items = [interview.get_qdict_from_qid(q_id) for q_id in q_ids]
    del interview

    # build all prompts from prompt_template
    prompts = dict()
    for q_item in q_items:
        prompt = build_few_shot_prompt(prompt_template, FEW_SHOT_EXAMPLES[:shot], q_item)
        prompts[q_item["q_id"]] = prompt
        print(prompt)

    # save prompts to file as json
    with open(f"../output/{file_prefix}prompts.json", 'w') as f:
        f.write(json.dumps(prompts, ensure_ascii=False, indent=4, sort_keys=True))


if __name__ == '__main__':
    main(dataset="test", instruction_faithful=True)
