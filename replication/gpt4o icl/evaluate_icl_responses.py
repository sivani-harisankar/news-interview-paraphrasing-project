"""
    evaluate the _model responses to the prompts
"""
import argparse
import json
import os.path
from statistics import mean
import random
import sklearn

from paraphrase import interview_data
from paraphrase.utility.parse_LLM import extract_classification, extract_host_quote, extract_guest_quote
from paraphrase.annotation_data import get_aggregated_human_anns
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.utility.stats import jaccard_overlap_for_highlights, binary_entropy

GPT4_TEST_RESPONSE_PATH = os.path.join(get_dir_to_src(),
                                       "../result/Models/Responses-GPT4_0-8-test_prompts_20240307_140445.json")


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def remove_prompt_from_reply(json_path, prompt_path):
    model_responses = load_json_file(json_path)
    prompt_dict = load_json_file(prompt_path)
    result_dict = {}
    for q_id, response_list in model_responses.items():
        response_list = [response_lst[0] if type(response_lst) == list else response_lst
                         for response_lst in response_list]
        result_response_list = []
        for cur_response in response_list:
            assert prompt_dict[q_id] in cur_response, f"Prompt not found in response: {cur_response}"
            cur_response = cur_response.replace(prompt_dict[q_id], "")
            result_response_list.append(cur_response)
        result_dict[q_id] = result_response_list
    # get the filename from json_path
    filename = os.path.basename(json_path)
    filename = f"{os.path.dirname(json_path)}/RMV_{filename}"
    with open(filename, "w") as file:
        json.dump(result_dict, file, indent=4, sort_keys=True)
    return filename


def main(response_path, prompt_path, no_prepended_prompt=False):
    """

    :param no_prepended_prompt: whether or not the response prepends the prompt or not,
    by default openai responses do not prepend the prompt
    :param response_path: path to _model responses
    :param prompt_path: path to the prompts which were used to generate the responses
    :return:
    """

    # PARSE MODEL RESPONSES
    #   GET _model responses dict
    model_responses = load_json_file(response_path)
    #   GET the prompt dict
    prompt_dict = load_json_file(prompt_path)
    #   GET the question ids from the prompt dict keys
    question_ids = list(prompt_dict.keys())
    #   GET human annotations
    interview = interview_data.MediaSumProcessor()
    guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid, human_class_per_qid = (
        get_aggregated_human_anns(interview, question_ids))

    model_class_per_qid, model_guest_weights_per_qid, model_hls_per_qid, model_host_weights_per_qid = (
        get_model_predictions(guest_tokens_per_qid, host_tokens_per_qid, model_responses, prompt_dict,
                              gpt4=no_prepended_prompt))

    # compare to human CLASSIFICATION
    print()
    correct_mv = print_classification_results(human_class_per_qid, model_class_per_qid, question_ids)

    print(f"HIGHLIGHTING RESULTS:")
    # compare to human highlights

    human_binary_guest_hls = [[1 if token_weight >= 0.5 else 0
                               for token_weight in human_anns_per_qid[q_id]["guest_weights"]]
                              for q_id in question_ids]
    model_binary_guest_hls = [[1 if token_weight >= 0.5 else 0
                               for token_weight in model_guest_weights_per_qid[q_id]]
                              for q_id in question_ids]
    jaccard_overlap_from_labels(human_binary_guest_hls, model_binary_guest_hls, question_ids, "guest")

    # get average hl length for human and _model
    
    human_guest_lengths = [sum(hl) for hl in human_binary_guest_hls if sum(hl) > 0]
    model_guest_lengths = [sum(hl) for hl in model_binary_guest_hls if sum(hl) > 0]

    avg_human_guest_hl_len = mean(human_guest_lengths) if human_guest_lengths else float("nan")
    avg_model_guest_hl_len = mean(model_guest_lengths) if model_guest_lengths else float("nan")

    print(f"\t {avg_human_guest_hl_len} avg human guest hl length, "
          f"{avg_model_guest_hl_len} avg _model guest hl length.")

    human_binary_host_hls = [[1 if token_weight >= 0.5 else 0
                              for token_weight in human_anns_per_qid[q_id]["host_weights"]]
                             for q_id in question_ids]
    model_binary_host_hls = [[1 if token_weight >= 0.5 else 0
                              for token_weight in model_host_weights_per_qid[q_id]]
                             for q_id in question_ids]

    jaccard_overlap_from_labels(human_binary_host_hls, model_binary_host_hls, question_ids, "host")

    # get average hl length for human and _model
    
    human_host_lengths = [sum(hl) for hl in human_binary_host_hls if sum(hl) > 0]
    model_host_lengths = [sum(hl) for hl in model_binary_host_hls if sum(hl) > 0]

    avg_human_host_hl_len = mean(human_host_lengths) if human_host_lengths else float("nan")
    avg_model_host_hl_len = mean(model_host_lengths) if model_host_lengths else float("nan")

    print(f"\t {avg_human_host_hl_len} avg human host hl length, "
          f"{avg_model_host_hl_len} avg _model host hl length.")

    # sample 10 indices from correct_mv that have the value 1

    sample_10_incorrect_preds(correct_mv, guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid,
                              human_class_per_qid, model_binary_guest_hls, model_binary_host_hls,
                              prompt_dict, question_ids)

    # print those examples

    human_inconc_guest_hl = [0
                             if (any(vote >= 0.5 for vote in human_anns_per_qid[q_id]["guest_weights"])) or
                                (sum(human_anns_per_qid[q_id]["guest_weights"]) == 0)
                             else 1
                             for q_id in question_ids]
    model_inconc_host_hl = [0
                            if (any(vote >= 0.5 for vote in model_guest_weights_per_qid[q_id])) or
                               (sum(model_guest_weights_per_qid[q_id]) == 0)
                            else 1
                            for q_id in question_ids]
    print(
        f"\t For {sum(human_inconc_guest_hl)} / {len(question_ids)} questions, human guest highlights are inconclusive.")
    print(
        f"\t For {sum(model_inconc_host_hl)} / {len(question_ids)} questions, _model guest highlights are inconclusive.")
    # the same for host
    human_inconc_host_hl = [0
                            if (any(vote >= 0.5 for vote in human_anns_per_qid[q_id]["host_weights"])) or
                               (sum(human_anns_per_qid[q_id]["host_weights"]) == 0)
                            else 1
                            for q_id in question_ids]
    model_inconc_guest_hl = [0
                             if (any(vote >= 0.5 for vote in model_host_weights_per_qid[q_id])) or
                                (sum(model_host_weights_per_qid[q_id]) == 0)
                             else 1
                             for q_id in question_ids]
    print(
        f"\t For {sum(human_inconc_host_hl)} / {len(question_ids)} questions, human host highlights are inconclusive.")
    print(
        f"\t For {sum(model_inconc_guest_hl)} / {len(question_ids)} questions, _model host highlights are inconclusive.")

    nbr_no_hls = (sum([err_hl
                       for non_hl, err_hl, _, _ in
                       list(model_hls_per_qid.values())]))
    #                / sum([model_class_per_qid[q_id][0] for q_id in question_ids]))
    
    nbr_class_para = sum([model_class_per_qid[q_id][0] for q_id in question_ids])

    print(
        f"\t {nbr_no_hls / nbr_class_para if nbr_class_para > 0 else float('nan')} "
        f"= {nbr_no_hls} / {nbr_class_para} highlight extraction errors over "
        f"times a paraphrase was classified."
    )

    nbr_hallucination_hl = (sum([hallucination
                                 for _, _, _, hallucination in
                                 list(model_hls_per_qid.values())]))
    print(
        f"\t\t where {nbr_hallucination_hl} / {nbr_class_para} = "
        f"{nbr_hallucination_hl / nbr_class_para if nbr_class_para > 0 else float('nan')} "
        f"for hallucinated highlights."
    )
    nbr_no_pair_hl = (sum([no_pair
                           for _, _, no_pair, _ in
                           list(model_hls_per_qid.values())]))
    print(
        f"\t\t where {nbr_no_pair_hl} / {nbr_class_para} = "
        f"{nbr_no_pair_hl / nbr_class_para if nbr_class_para > 0 else float('nan')} "
        f"for missing highlight pairs."
    )
    nbr_no_identifier_hl = (sum([no_identifier
                                 for no_identifier, _, _, _ in
                                 list(model_hls_per_qid.values())]))
    print(
        f"\t\t where {nbr_no_identifier_hl} / {nbr_class_para} = "
        f"{nbr_no_identifier_hl / nbr_class_para if nbr_class_para > 0 else float('nan')} "
        f"for missing highlight identifiers."
    )
    return sum(correct_mv) / len(correct_mv)


def sample_10_incorrect_preds(correct_mv, guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid,
                              human_class_per_qid, model_binary_guest_hls, model_binary_host_hls,
                              prompt_dict, question_ids):
    indices_with_ones = [i for i, value in enumerate(correct_mv) if value == 0]
    sampled_indices = random.sample(
    indices_with_ones,
    min(17, len(indices_with_ones))
)
    for i in sampled_indices:
        q_id = question_ids[i]
        print(f"Question ID: {q_id}, correct label: {human_class_per_qid[q_id]}")
        print(f"Prompt:\n {prompt_dict[q_id]}".split("Given an interview on ")[-1])
        human_hl_guest = " ".join([
            token for token, weight in
            zip(guest_tokens_per_qid[q_id], human_anns_per_qid[question_ids[i]]["guest_weights"])
            if weight >= 0.5])
        model_hl_guest = " ".join([
            token for token, weight in
            zip(guest_tokens_per_qid[q_id], model_binary_guest_hls[i])
            if weight >= 0.5])
        print(f"Human Highlights Guest: {human_hl_guest}")
        print(f"Model Highlights Guest: {model_hl_guest}")
        human_hl_host = " ".join([
            token for token, weight in
            zip(host_tokens_per_qid[q_id], human_anns_per_qid[question_ids[i]]["host_weights"])
            if weight >= 0.5])
        model_hl_host = " ".join([
            token for token, weight in
            zip(host_tokens_per_qid[q_id], model_binary_host_hls[i])
            if weight >= 0.5])
        print(f"Human Highlights Host: {human_hl_host}")
        print(f"Model Highlights Host: {model_hl_host}")


def jaccard_overlap_from_labels(human_binary_host_hls, model_binary_host_hls, question_ids, identifier):
    both_hl_host = [[
        [i for i, x in enumerate(human) if x == 1],
        [i for i, x in enumerate(model) if x == 1]]
        for human, model in zip(human_binary_host_hls, model_binary_host_hls)
        if (human.count(1) > 0) and (model.count(1) > 0)  # only if both highlighted something
    ]
    jaccard_host = jaccard_overlap_for_highlights(both_hl_host)
    print(f"\t {jaccard_host} avg MV jaccard for {identifier}, "
          f"calculated for {len(both_hl_host)} / {len(question_ids)} questions, "
          f"where both human and models have mv highlights on the  utterance. ")


def get_model_predictions(guest_tokens_per_qid, host_tokens_per_qid, model_responses, prompt_dict, gpt4=False):
    model_class_per_qid = {}
    model_host_weights_per_qid = {}
    model_guest_weights_per_qid = {}
    model_hls_per_qid = {}
    for q_id, response_list in model_responses.items():
        if q_id == "CNN-339958-3":
            print(f"DEBUG: {response_list}")
        yes_nbr = 0
        no_nbr = 0
        err_classification_nbr = 0
        no_highlights_nbr = 0
        err_hl_no_pair_nbr = 0
        err_hl_nbr = 0
        err_hl_g_nbr = 0
        err_hl_h_nbr = 0
        err_hl_hallucination_nbr = 0
        err_hl_no_identifier = 0
        # q_item = human_anns_per_qid[q_id]
        cur_host_hls = [0 for _ in host_tokens_per_qid[q_id]]
        cur_guest_hls = [0 for _ in guest_tokens_per_qid[q_id]]
        response_list = [response_lst[0] if type(response_lst) == list else response_lst
                         for response_lst in response_list]
        #   get classification from response

        for cur_response in response_list:
            if not gpt4:
                assert prompt_dict[q_id] in cur_response, f"Prompt not found in response: {cur_response}"
                cur_response = cur_response.replace(prompt_dict[q_id], "")
                # also remove eventual continuation with further made up replies
                if "Given an interview on" in cur_response:
                    cur_response = cur_response.split("Given an interview on")[0]

            # extract classification
            try:
                is_yes = extract_classification(cur_response)
                if is_yes:
                    yes_nbr += 1
                else:
                    no_nbr += 1

            except ValueError:
                print(f"Could not find classification for response")
                err_classification_nbr += 1
                continue

            if is_yes:  # for now only extract if the _model classified as paraphrase
                # flag for highlighting errors (necessary because always two identifiers are present)
                hal_err = False
                incon_err = False
                identifier_err = False

                # extract guest quotes
                cur_guest_hls, hal_err_guest, identifier_err_guest, incon_err_guest = err_aware_hl_extraction(
                    cur_response, cur_guest_hls,
                    guest_tokens_per_qid,
                    hal_err,
                    identifier_err, incon_err,
                    q_id,
                    prompt_dict, host=False)
                # extract host quotes
                cur_host_hls, hal_err_host, identifier_err_host, incon_err_host = err_aware_hl_extraction(cur_response,
                                                                                                          cur_host_hls,
                                                                                                          host_tokens_per_qid,
                                                                                                          hal_err,
                                                                                                          identifier_err,
                                                                                                          incon_err,
                                                                                                          q_id,
                                                                                                          prompt_dict,
                                                                                                          host=True)

                hal_err = hal_err_guest or hal_err_host
                incon_err = incon_err_guest or incon_err_host
                identifier_err = identifier_err_guest or identifier_err_host

                if hal_err or incon_err or identifier_err:
                    err_hl_nbr += 1
                    if hal_err_guest:
                        err_hl_g_nbr += 1
                    if hal_err_host:
                        err_hl_h_nbr += 1
                    if hal_err:
                        err_hl_hallucination_nbr += 1
                    if incon_err:
                        err_hl_no_pair_nbr += 1
                    if identifier_err:
                        err_hl_no_identifier += 1

        model_class_per_qid[q_id] = [yes_nbr, no_nbr, err_classification_nbr]
        model_hls_per_qid[q_id] = [err_hl_no_identifier, err_hl_nbr, err_hl_no_pair_nbr, err_hl_hallucination_nbr]
        model_host_weights_per_qid[q_id] = [vote / max((yes_nbr - err_hl_h_nbr), 1) for vote in cur_host_hls]
        model_guest_weights_per_qid[q_id] = [vote / max((yes_nbr - err_hl_g_nbr), 1) for vote in cur_guest_hls]
    return model_class_per_qid, model_guest_weights_per_qid, model_hls_per_qid, model_host_weights_per_qid


def err_aware_hl_extraction(cur_response, cur_hls, utt_tokens_per_qid, hal_err, identifier_err, incon_err, q_id,
                            prompt_dict, debug=False, host=False):
    if q_id == "CNN-339958-3":
        print(f"DEBUG")
    try:
        if host:
            tmp_quotes = extract_host_quote(cur_response, utt_tokens_per_qid[q_id])
        else:
            tmp_quotes = extract_guest_quote(cur_response, utt_tokens_per_qid[q_id])
        cur_hls = [new_vote + cur_vote
                   for new_vote, cur_vote in
                   zip(tmp_quotes, cur_hls)]
    except ValueError as e:
        # if debug:
        print(f"[Question ID: {q_id}] Could not extract guest and/or host quote for response: {e}")
        if "Quoted phrase not found" in str(e):
            relevant_prompt_part = prompt_dict[q_id].split("Given an interview on ")[-1]
            if debug:
                print(f"\t {q_id} where prompt was {relevant_prompt_part}")
            hal_err = True
        elif ("One of the host or guest quote is none" in str(e)) or ("Empty phrase found." in str(e)):
            incon_err = True
        elif "Identifier" in str(e):
            identifier_err = True
        else:
            raise ValueError(f"Unknown error: {e}")
    return cur_hls, hal_err, identifier_err, incon_err


def print_classification_results(human_class_per_qid, model_class_per_qid, question_ids):
    human_binary = [1 if (human_class_per_qid[q_id][0] / human_class_per_qid[q_id][1] >= 0.5) else 0
                    for q_id in question_ids]
    model_binary = [1 if ((model_class_per_qid[q_id][0] + model_class_per_qid[q_id][1]) > 0) and  # no divide by zero
                         (model_class_per_qid[q_id][0] /
                          (model_class_per_qid[q_id][0] + model_class_per_qid[q_id][1]) >= 0.5) else 0
                    for q_id in question_ids]
    human_share = [human_class_per_qid[q_id][0] / human_class_per_qid[q_id][1]
                   for q_id in question_ids]

    correct_mv = print_f1_classifications(human_binary, model_binary, human_share)

    no_class_nbr = (sum([nan_class
                         for _, _, nan_class in list(model_class_per_qid.values())]))
    print(f"\t  {no_class_nbr / (len(question_ids) * sum(list(model_class_per_qid.values())[0][:3]))} = "
          f"There are {no_class_nbr} classification parsing errors over "
          f"{len(question_ids)} questions x {sum(list(model_class_per_qid.values())[0][:3])} self-consistency prompts.")
    nbr_inconclusive_class = sum([0 if (yes_nbr + no_nbr) > 0 else 1
                                  for yes_nbr, no_nbr, _ in list(model_class_per_qid.values())])
    print(f"\t For {nbr_inconclusive_class} / {len(question_ids)} questions there exists no conclusive classification.")
    return correct_mv


def print_f1_classifications(human_binary, model_binary, human_share=None):
    print(f"CLASSIFICATION RESULTS:")
    correct_mv = [1 if model == human else 0
                  for model, human in zip(model_binary, human_binary)]
    print(f"\t Accuracy at {sum(correct_mv) / len(correct_mv)}.")
    print(f"\t MACRO F1 Score at {sklearn.metrics.f1_score(human_binary, model_binary, average='macro')}"
          f" with {sklearn.metrics.precision_score(human_binary, model_binary, average='macro')} precision and"
          f" {sklearn.metrics.recall_score(human_binary, model_binary, average='macro')} recall.")
    print(sklearn.metrics.classification_report(human_binary, model_binary))
    
    if human_share:
        correct_dist_to_05 = [
            abs(0.5 - human_share[i])
            for i, (model, human) in enumerate(zip(model_binary, human_binary))
            if model == human
        ]
        correct_entropy = [
            binary_entropy(human_share[i])
            for i, (model, human) in enumerate(zip(model_binary, human_binary))
            if model == human
        ]

        incorrect_dist_to_05 = [
            abs(0.5 - human_share[i])
            for i, (model, human) in enumerate(zip(model_binary, human_binary))
            if model != human
        ]
        incorrect_entropy = [
            binary_entropy(human_share[i])
            for i, (model, human) in enumerate(zip(model_binary, human_binary))
            if model != human
        ]

        print(
            f"\t AVG human dist from 0.5 for correct predictions: "
            f"{mean(correct_dist_to_05) if correct_dist_to_05 else 'N/A'} "
            f"support {len(correct_dist_to_05)}."
        )
        print(
            f"\t AVG human entropy for correct predictions: "
            f"{mean(correct_entropy) if correct_entropy else 'N/A'} "
            f"support {len(correct_entropy)}."
        )
        print(
            f"\t AVG human dist from 0.5 for incorrect predictions: "
            f"{mean(incorrect_dist_to_05) if incorrect_dist_to_05 else 'N/A'} "
            f"support {len(incorrect_dist_to_05)}."
        )
        print(
            f"\t AVG human entropy for incorrect predictions: "
            f"{mean(incorrect_entropy) if incorrect_entropy else 'N/A'} "
            f"support {len(incorrect_entropy)}."
        )


    return correct_mv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Llama Configurations')
    parser.add_argument('-prompt', '--prompt_path', help="path where prompts are taken from")
    parser.add_argument('-response', '--response_path',
                        help="path to the prompt responses generated after calling incontext_prompting.py")

    args = parser.parse_args()

    main(
    response_path=args.response_path,
    prompt_path=args.prompt_path,
    no_prepended_prompt=True
)
