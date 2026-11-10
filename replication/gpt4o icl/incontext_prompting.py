"""
    This script is used to prompt generative models with a given set of prompts.
"""
import argparse
import json
import os
from datetime import datetime

from paraphrase.set_id_consts import CLUSTER_OUT
from paraphrase.utility.project_functions import set_logging
from paraphrase.set_id_consts import set_cache
from paraphrase.utility.stats import set_global_seed
import paraphrase.LLMs as LLMs


def main(prompt_path, huggingface_id="meta-llama/Llama-2-7b-hf", load_in="4bit", output_path="",
         gpt4=False, self_consistency=10):
    set_logging()
    set_cache()
    set_global_seed()

    if prompt_path is None:
        raise ValueError("No prompt path given")

    filename = os.path.basename(prompt_path)
    filename = os.path.splitext(filename)[0]

    try:
    	with open(prompt_path, "r", encoding="utf-8") as file:
        	prompts = json.load(file)
    except UnicodeDecodeError:
    	with open(prompt_path, "r", encoding="cp1252") as file:
        	prompts = json.load(file)

    if not gpt4:
        print(f"Loading huggingface _model with id {huggingface_id}...")
        llm = LLMs.HuggingfaceModel(model_id=huggingface_id, load_in=load_in)
        out_prefix = f"Responses-{huggingface_id.split('/')[-1]}{load_in}_{filename}"
    else:
        print("Setting GPT4 api key")
        llm = LLMs.GPT4()
        out_prefix = f"Responses-GPT4_{filename}"

    print(f"Prompting _model ...")
    responses = {}
    checkpoints = 10
    nbr_prompts = 0
    for q_id, prompt in prompts.items():
        cur_sublist = []
        for _ in range(self_consistency):
            cur_sublist.append(llm.query(prompt))
            nbr_prompts += 1
        responses[q_id] = cur_sublist
        if nbr_prompts % checkpoints == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = output_path + f"CHCKPT-{int(nbr_prompts / checkpoints)}_{out_prefix}_{timestamp}.json"
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump(responses, file, indent=4, sort_keys=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_path + f"{out_prefix}_{timestamp}.json"
    with open(out_path, "w") as file:
        json.dump(responses, file, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama Configurations')
    parser.add_argument('-prompt', '--prompt_path', help="path where prompts are taken from")
    parser.add_argument('-llm', '--huggingface_id', default="meta-llama/Llama-2-7b-hf",
                        help="huggingface _model id")
    parser.add_argument('-load', '--load_in', default="4bit", help="load in 4bit, 8bit or 16float")
    parser.add_argument('-out', '--output_path',
                        default=CLUSTER_OUT,
                        help="path where responses are saved")
    parser.add_argument('-sc', '--self_consistency', default=10,
                        help="how many times the _model should be called")

    args = parser.parse_args()

    main(huggingface_id=args.huggingface_id, load_in=args.load_in, output_path=args.output_path,
     prompt_path=args.prompt_path, self_consistency=int(args.self_consistency), gpt4=True)
