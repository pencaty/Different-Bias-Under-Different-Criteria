import os
import json
from tqdm import tqdm

from parse import parse_args
from model import get_generative_model, generate_prompt, request
from utils import read_text_file, get_model_list


def process(args, model, tokenizer):

    # make directory if not exists
    os.makedirs(args.output_dir.format(args.keyword, args.model, str(args.prompt)), exist_ok=True)

    sentence_file_path = args.data_dir + args.txt_dir.format(args.keyword) + '/agg_sentences.txt'
    input_sentences = read_text_file(sentence_file_path)

    biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/biased_job.txt'
    anti_biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/anti_biased_job.txt'
    input_order_path = args.data_dir + args.txt_dir.format(args.keyword) + '/input_order.txt'
    biased_job = read_text_file(biased_job_path)
    anti_biased_job = read_text_file(anti_biased_job_path)
    input_orders = read_text_file(input_order_path)

    response_json = {}
    selected_jobs = []

    model_list = get_model_list()

    for idx, sentence in enumerate(tqdm(input_sentences, mininterval=0.01)):

        input_data = [sentence, biased_job[idx], anti_biased_job[idx], "unknown"]
        input_order = input_orders[idx].split(",")
            
        system_prompt, user_prompt = generate_prompt(args, input_data, input_order)
        response = request(args, model, tokenizer, system_prompt, user_prompt, ['1', '2', '3'])

        response_json[idx] = response

        if args.model in ['gpt3.5', 'gpt4', 'gpt4o_mini']:
            selected_job = response["choices"][0]["message"]["content"]
            if "\n" in selected_job:
                selected_job = selected_job.replace("\n", " ")
            selected_jobs.append(selected_job + "\n")

        elif args.model in model_list:
            response_json[idx] = str(response)
            selected_job = str(response)
            selected_jobs.append(selected_job + "\n")

        else:
            response_json[idx] = str(response)
            selected_job = response.outputs[0].text
            if "\n" in selected_job:
                selected_job = selected_job.replace("\n", " ")
            if "<|start_header_id|>assistant<|end_header_id|>" in selected_job: # LLaMA
                selected_job = selected_job.replace("<|start_header_id|>assistant<|end_header_id|>", " ")
            selected_jobs.append(selected_job + "\n")

        if response == None:
            continue

        if idx % 10 == 0:
            print("{}th sentence is completed".format(idx))
        
    print("All iterations are completed")
    
    with open(args.output_dir.format(args.keyword, args.model, str(args.prompt)) + args.response_json_file, 'w') as out:
        json.dump(response_json, out, indent=4, sort_keys=False, ensure_ascii=False)

    with open(args.output_dir.format(args.keyword, args.model, str(args.prompt)) + args.output_txt_file, 'w') as out:
        out.writelines(selected_jobs)


def main():

    args = parse_args()
    model, tokenizer = get_generative_model(args)

    if model == None:
        print("NO SUCH MODEL")
    
    else:
        if args.keyword in ['wino_gender', 'persona_gender', 'persona_age']:
            process(args, model, tokenizer)


if __name__ == "__main__":
    main()