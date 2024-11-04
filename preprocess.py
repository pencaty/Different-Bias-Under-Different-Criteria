import argparse
import random
import json
import os

from collections import OrderedDict
from utils import read_text_file, read_json_file, make_target, get_clean_sentences

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--keyword', type=str, default='wino_gender')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--stats_dir', type=str, default='/{}/stats')
    parser.add_argument('--txt_dir', type=str, default='/{}/txt')

    return parser.parse_args()


def merge_files(files):

    total_sentence_list = []

    for file in files:
        with open(file, 'r') as f:
            total_sentence_list.extend(get_clean_sentences(file))
    
    return total_sentence_list


def calc_weighted_avg_occupation_ratio(args):

    # weighted average of gender/race/age ratio for each occupation

    detail_occupation_stats_path = args.data_dir + args.stats_dir.format(args.keyword) + '/detailed_stats_occupation_ratio.json'
    agg_occupation_stats_path = args.data_dir + args.stats_dir.format(args.keyword) + '/agg_stats_occupation_ratio.json'
    occupation_stats_path = args.data_dir + args.stats_dir.format(args.keyword) + '/stats_occupation_ratio.json'

    ratio_name = ""
    if args.keyword == 'persona_gender' or args.keyword == 'wino_gender':
        ratio_name = "woman_ratio"
    if args.keyword == 'persona_age':
        ratio_name = "under_44_ratio"

    if ratio_name == "":
        print("Wrong keyword")
        return {}
   
    detail_occupation_ratio = read_json_file(detail_occupation_stats_path)

    if detail_occupation_ratio == {}:
        print("Fail to load detailed_stats_occupation_ratio.json")
        return {}

    else:
        occupation_ratio = {}
        agg_occupation_stats = {}
        for job in detail_occupation_ratio.keys():
            detail_job_list = detail_occupation_ratio[job]
            acc_num = 0
            acc_target_num = 0
            for detail_job in detail_job_list.keys():
                acc_num = acc_num + detail_job_list[detail_job]["number"]
                acc_target_num = acc_target_num + detail_job_list[detail_job]["number"] * detail_job_list[detail_job][ratio_name] / 100

            occupation_ratio[job] = round(acc_target_num / acc_num, 2)
            agg_occupation_stats[job] = {"number": acc_num, ratio_name: round(acc_target_num / acc_num, 2)}
        
        sorted_occupation_ratio = OrderedDict(sorted(occupation_ratio.items(), key=lambda x: x[1], reverse=False))
        
        with open(occupation_stats_path, "w") as stats_job_file:
            json.dump(sorted_occupation_ratio, stats_job_file, indent=4)

        with open(agg_occupation_stats_path, "w") as agg_stats_ratio_file:
            json.dump(agg_occupation_stats, agg_stats_ratio_file, indent=4)
    
        return sorted_occupation_ratio


def get_gender_biased_occupation(args, sentence_file_path):

    male_job_path = args.data_dir + args.stats_dir.format(args.keyword) + '/male_occupations.txt'
    female_job_path = args.data_dir + args.stats_dir.format(args.keyword) + '/female_occupations.txt'

    male_job_list = read_text_file(male_job_path)
    female_job_list = read_text_file(female_job_path)

    idx = 0
    sentence_list = read_text_file(sentence_file_path)

    biased_job = []
    anti_biased_job = []

    for sentence in sentence_list:
        idx = idx + 1
        if ' she ' in sentence or ' her ' in sentence or ' her.' in sentence:
            for female_job in female_job_list:
                if female_job in sentence:
                    biased_job.append(female_job + "\n")
                    break
            for male_job in male_job_list:
                if male_job in sentence:
                    anti_biased_job.append(male_job + "\n")
                    break

        elif ' he ' in sentence or ' his ' in sentence or ' him ' in sentence or ' him.' in sentence:
            for male_job in male_job_list:
                if male_job in sentence:
                    biased_job.append(male_job + "\n")
                    break
            for female_job in female_job_list:
                if female_job in sentence:
                    anti_biased_job.append(female_job + "\n")
                    break
        else:
            print("No pronoun exists")
        

    biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/biased_job.txt'
    anti_biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/anti_biased_job.txt'

    with open(biased_job_path, 'w') as out:
        out.writelines(biased_job)

    with open(anti_biased_job_path, 'w') as out:
        out.writelines(anti_biased_job)


def aggregate_sentences_gender(args):

    os.makedirs(args.data_dir + args.txt_dir.format(args.keyword), exist_ok=True)

    file1_path = "path to pro_stereotyped_dev.txt"
    file2_path = "path to anti_stereotyped_dev.txt"
    file3_path = "path to pro_stereotyped_test.txt"
    file4_path = "path to anti_stereotyped_test.txt"

    file_path_list = [file1_path, file2_path, file3_path, file4_path]

    sentence_list = merge_files(file_path_list)

    sentence_file_path = args.data_dir + args.txt_dir.format(args.keyword) + '/agg_sentences.txt'

    with open(sentence_file_path, 'w') as out:
        out.writelines(sentence_list)

    return sentence_file_path


def make_persona_sentences(args, occupation_ratio):

    sentence_file_path = args.data_dir + args.txt_dir.format(args.keyword) + '/agg_sentences.txt'
    biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/biased_job.txt'
    anti_biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/anti_biased_job.txt'

    os.makedirs(args.data_dir + args.txt_dir.format(args.keyword), exist_ok=True)

    sentence_list = []
    biased_job = []
    anti_biased_job = []

    sentence_candidate = [
        " living in the United States\n",
        " working in the United States\n"
        ]

    occupation_list = list(occupation_ratio.keys())
    occupation_split = int(len(occupation_list) / 2)
    job1_list = occupation_list[occupation_split:]
    job2_list = occupation_list[:occupation_split]
    group1, group2 = make_target(args.keyword)

    for job1 in job1_list:
        for job2 in job2_list:
            sentence_list.append(group1 + sentence_candidate[0])
            biased_job.append(job1 + "\n")
            anti_biased_job.append(job2 + "\n")

            sentence_list.append(group1 + sentence_candidate[1])
            biased_job.append(job1 + "\n")
            anti_biased_job.append(job2 + "\n")

            sentence_list.append(group2 + sentence_candidate[0])
            biased_job.append(job2 + "\n")
            anti_biased_job.append(job1 + "\n")

            sentence_list.append(group2 + sentence_candidate[1])
            biased_job.append(job2 + "\n")
            anti_biased_job.append(job1 + "\n")

    with open(biased_job_path, 'w') as out:
        out.writelines(biased_job)

    with open(anti_biased_job_path, 'w') as out:
        out.writelines(anti_biased_job)

    with open(sentence_file_path, 'w') as out:
        out.writelines(sentence_list)


def preprocess():

    args = parse_args()

    if args.keyword in ['wino_gender']:
        sentence_file_path = aggregate_sentences_gender(args)
        get_gender_biased_occupation(args, sentence_file_path)
        calc_weighted_avg_occupation_ratio(args)
    
    elif args.keyword in ['persona_gender', 'persona_age']:
        sorted_occupation_ratio = calc_weighted_avg_occupation_ratio(args)
        make_persona_sentences(args, sorted_occupation_ratio)

    else:
        print("Wrong keyword")


if __name__ == "__main__":
    preprocess()