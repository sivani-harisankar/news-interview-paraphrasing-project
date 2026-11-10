"""
    generative model prompting
"""
# Use a pipeline as a high-level helper
from typing import List

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import time, random

from paraphrase.set_id_consts import GPT_4
from paraphrase.utility.project_functions import set_torch_device

device = set_torch_device()


class GPT4:

    def __init__(self, api_key=None, model_type: str = "gpt-4o"):
        self.model_type = model_type
        self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str):
        return self._send_request(self.model_type, prompt)

    def _send_request(self, model_type, prompt):

        message = [{"role": "user", "content": prompt}]

        for delay_secs in (2 ** x for x in range(0, 6)):
            try:
                response = self.client.chat.completions.create(
                    model=model_type,
                    messages=message,
                    temperature=1,
                    # seed=seed,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                return response.choices[0].message.content

            except openai.OpenAIError as e:  # TODO wait as long as suggested in the error message
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue
        raise openai.OpenAIError("Could not send request to OpenAI API.")


class HuggingfaceModel:
    def __init__(self, model_id="meta-llama/Llama-2-7b-chat-hf", load_in="4bit"):
        """
            for some reason this throws  UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.5` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
                --> this should be fine since we override both do_sample and temperature anyway in generation, but this remains strange
        :param model_id:
        :param load_in:
        """
        model_ids = ["meta-llama/Llama-2-7b-hf",
                     "lmsys/vicuna-7b-v1.5",
                     # "mistralai/Mistral-7B-v0.1",
                     "mistralai/Mistral-7B-Instruct-v0.2",
                     "openchat/openchat-3.5-0106",
                     "mistralai/Mixtral-8x7B-Instruct-v0.1",
                     "google/gemma-7b-it",
                     "meta-llama/Llama-2-13b-hf",
                     "meta-llama/Llama-2-70b-hf",
                     "lmsys/vicuna-13b-v1.5",
                     "openai-community/gpt2"]

        if model_id == "7b":
            model_id = "meta-llama/Llama-2-7b-hf"
        elif model_id == "13b":
            model_id = "meta-llama/Llama-2-13b-hf"
        elif model_id == "70b":
            model_id = "meta-llama/Llama-2-70b-hf"
        elif model_id not in model_ids:
            raise ValueError(f"Given Llama type unknown: {model_id}")

        print(f"Loading Huggingface _model Model {model_id} in {load_in} ...")
        if load_in == "4bit":
            # should only be called on CUDA
            from transformers import BitsAndBytesConfig
            # load _model in 4-bit
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            #   https://huggingface.co/docs/transformers/main/en/llm_tutorial
            #   https://huggingface.co/blog/4bit-transformers-bitsandbytes
            #       --> not all models are supported
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif load_in == "8bit":
            # should only be called on CUDA
            # load _model in 8-bit
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                do_sample=True,
                temperature=1.0
            )
        elif torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # dtype=torch.float16,
                torch_dtype="auto",
                device_map="auto",
                do_sample=True,
                temperature=1.0
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                do_sample=True,
                temperature=1.0
            )
        print(f"Model taking up {self.model.get_memory_footprint()} of memory space.")
        # TODO? enable BetterTransformer
        # self._model = self._model.to_bettertransformer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if model_id in self.tokenizer.max_model_input_sizes:
            # Get the maximum length supported by the tokenizer
            self.max_length = self.tokenizer.max_model_input_sizes[model_id]
        else:
            self.max_length = None
        # Possibly for Batch processing
        #   --> currently not used because of errors with some models when using batch processing
        #   i.e., remove for now
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def query(self, prompt="", max_new_tokens=400, repetitions=1) -> List[str]:
        """
            loading llama 2 _model with memory optimization,
            see: https://huggingface.co/docs/transformers/perf_infer_gpu_one#bitsandbytes
        :param model_id:
        :param load_4bit:
        :return:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)  # padding=True for batch processing, currently not uesd

        if self.max_length is not None:
            # truncate but from the start
            max_len = self.max_length - max_new_tokens
            if inputs["input_ids"].shape[1] > max_len:
                print("Exceeds maximum length, this should not happen for LLMs...")
                inputs["input_ids"] = inputs["input_ids"][:, -max_len:]
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_len:]

        # # enable FlashAttention
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        #     outputs = self._model.generate(**inputs)
        # same params as here: https://huggingface.co/blog/llama2#inference
        generated_ids = self.model.generate(**inputs,
                                            max_new_tokens=max_new_tokens,
                                            do_sample=True,
                                            temperature=1.0,
                                            top_k=10,
                                            num_return_sequences=repetitions,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            )

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def read_in_responses(f_path: str) -> List[List[str]]:
    # import the necessary module
    import ast

    # Open the file
    with open(f_path, 'r') as file:
        # read the first line
        first_line = file.readline()
        # convert string to list of lists
        list_of_lists = ast.literal_eval(first_line)
        hl_answers = list_of_lists

    return hl_answers
