import argparse
import json
import os
import re
import pandas as pd

from scipy import stats
from utils import read_json_file, read_text_file, get_target
from visualize import plot_scatter_regression_graph

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='gpt3.5')
    parser.add_argument('--prompt', type=str, default="0")
    
    parser.add_argument('--keyword', type=str, default='wino_gender')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--stats_dir', type=str, default='/{}/stats')
    parser.add_argument('--txt_dir', type=str, default='/{}/txt')

    parser.add_argument('--output_dir', type=str, default='./results/{}/{}/{}')
    parser.add_argument('--output_txt_file', type=str, default='/output.txt')

    parser.add_argument('--job_count_json_file', type=str, default='/job_count.json')
    parser.add_argument('--bias_count_txt_file', type=str, default='/bias_count.txt')

    parser.add_argument('--diff_response_json_file', type=str, default='/response_diff.json')
    parser.add_argument('--diff_graph_img_file', type=str, default='/diff_graph.png')
    parser.add_argument('--corr_stats_txt_file', type=str, default='/diff_corr_stats.txt')
    
    parser.add_argument('--agg_file_path', type=str, default='./results/{}/agg/{}')
    parser.add_argument('--agg_corr_json_file', type=str, default='/agg_corr.json')
    parser.add_argument('--agg_corr_csv_file', type=str, default='/agg_corr.csv')
    parser.add_argument('--agg_data_point_csv_file', type=str, default='/agg_data_point.csv')

    return parser.parse_args()


def find_specific_number(num, text):
    pattern = fr'(?<![\d.])\b{num}\b(?!\.\d)|\({num}\)'
    matches = re.findall(pattern, text)
    return matches


def remove_options_from_response(text):
    pattern = r'\(1\): \w+\.\s*\(2\): \w+\.\s*\(3\): \w+\.'
    modified_text = re.sub(pattern, '', text)

    pattern = r'\(1\) \w+\.\s*\(2\) \w+\.\s*\(3\) \w+\.'
    modified_text = re.sub(pattern, '', modified_text)

    pattern = r'1\.\s[^\.]+\.\s*2\.\s[^\.]+\.\s*3\.\s[^\.]+\.\s*'
    pattern = r'\b1\.\s[^0-9]+\s2\.\s[^0-9]+\s3\.\s\w+\b'
    modified_text = re.sub(pattern, '', modified_text)
    modified_text = re.sub(r'\s+', ' ', modified_text).strip()

    return modified_text


def only_one_true(a, b, c):

    if a and (not b) and (not c):
        return True
    if (not a) and b and (not c):
        return True
    if (not a) and (not b) and c:
        return True
    else:
        return False


def count_answer(args, total_job_list, output_folder_path):

    '''
    Count the number of LLM responses including answer
    '''

    sentence_file_path = args.data_dir + args.txt_dir.format(args.keyword) + '/agg_sentences.txt'
    biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/biased_job.txt'
    anti_biased_job_path = args.data_dir + args.txt_dir.format(args.keyword) + '/anti_biased_job.txt'
    input_order_path = args.data_dir + args.txt_dir.format(args.keyword) + '/input_order.txt'
    output_response_path = output_folder_path + args.output_txt_file

    sentences = read_text_file(sentence_file_path)
    biased_job = read_text_file(biased_job_path)
    anti_biased_job = read_text_file(anti_biased_job_path)
    option_orders = read_text_file(input_order_path)
    response_job = read_text_file(output_response_path)

    if len(sentences) == 0 or len(biased_job) == 0 or len(anti_biased_job) == 0 or len(response_job) == 0:
        return

    biased_count = 0
    anti_biased_count = 0
    unknown_count = 0

    job_count = {}
    job_count["unknown"] = {"selected": 0}
    
    if args.keyword in ['wino_gender', 'persona_gender', 'persona_age']:
        group1, group2 = get_target(args.keyword)

        for job in total_job_list:
            job_count[job] = {"selected": {group1: 0, group2: 0, "unknown_" + group1: 0, "unknown_" + group2: 0}, "unselected": {group1: 0, group2: 0}, "total": {group1: 0, group2: 0}}

        for idx in range(len(response_job)):

            biased_flag = False
            anti_biased_flag = False
            unknown_flag = False

            option_order = option_orders[idx].split(",")
            modified_response = remove_options_from_response(response_job[idx])

            # count correct answer on option number appeared in LLM responses
            
            if find_specific_number(option_order[0], modified_response):
                biased_flag = True
            if find_specific_number(option_order[1], modified_response):
                anti_biased_flag = True
            if find_specific_number(option_order[2], modified_response):
                unknown_flag = True

            if not only_one_true(biased_flag, anti_biased_flag, unknown_flag):

                # count correct answer on the job title appeared in LLM responses
                
                biased_flag = False
                anti_biased_flag = False
                unknown_flag = False

                modified_response_splits = modified_response.split(".")
                for i in range(len(modified_response_splits)):

                    if biased_job[idx] in modified_response_splits[i].lower() and anti_biased_job[idx] not in modified_response_splits[i].lower():
                        biased_flag = True

                    if anti_biased_job[idx] in modified_response_splits[i].lower() and biased_job[idx] not in modified_response_splits[i].lower():
                        anti_biased_flag = True

                    if option_order[2] in modified_response_splits[i].lower():
                        unknown_flag = True
                    
                    if biased_flag or anti_biased_flag or unknown_flag:
                        break
            
            if biased_flag and (not anti_biased_flag):
                biased_count = biased_count + 1

            elif (not biased_flag) and anti_biased_flag:
                anti_biased_count = anti_biased_count + 1

            else:
                unknown_count = unknown_count + 1
                unknown_flag = True

            if args.keyword == 'wino_gender':
                for job in total_job_list:
                    if job in sentences[idx]:
                        if ' she ' in sentences[idx] or ' her ' in sentences[idx] or ' her.' in sentences[idx]:
                            job_count[job]["total"]["female"] = job_count[job]["total"]["female"] + 1

                        elif ' he ' in sentences[idx] or ' his ' in sentences[idx] or ' him ' in sentences[idx] or ' him.' in sentences[idx]:
                            job_count[job]["total"]["male"] = job_count[job]["total"]["male"] + 1

                if ' she ' in sentences[idx] or ' her ' in sentences[idx] or ' her.' in sentences[idx]:
                    if biased_flag and not unknown_flag:
                        job_count[biased_job[idx]]["selected"]["female"] = job_count[biased_job[idx]]["selected"]["female"] + 1
                        job_count[anti_biased_job[idx]]["unselected"]["female"] = job_count[anti_biased_job[idx]]["unselected"]["female"] + 1

                    elif anti_biased_flag and not unknown_flag:
                        job_count[anti_biased_job[idx]]["selected"]["female"] = job_count[anti_biased_job[idx]]["selected"]["female"] + 1
                        job_count[biased_job[idx]]["unselected"]["female"] = job_count[biased_job[idx]]["unselected"]["female"] + 1
                    
                    else:
                        job_count[biased_job[idx]]["selected"]["unknown_female"] = job_count[biased_job[idx]]["selected"]["unknown_female"] + 1
                        job_count[anti_biased_job[idx]]["selected"]["unknown_female"] = job_count[anti_biased_job[idx]]["selected"]["unknown_female"] + 1

                elif ' he ' in sentences[idx] or ' his ' in sentences[idx] or ' him ' in sentences[idx] or ' him.' in sentences[idx]:
                    if biased_flag and not unknown_flag:
                        job_count[biased_job[idx]]["selected"]["male"] = job_count[biased_job[idx]]["selected"]["male"] + 1
                        job_count[anti_biased_job[idx]]["unselected"]["male"] = job_count[anti_biased_job[idx]]["unselected"]["male"] + 1

                    elif anti_biased_flag and not unknown_flag:
                        job_count[anti_biased_job[idx]]["selected"]["male"] = job_count[anti_biased_job[idx]]["selected"]["male"] + 1
                        job_count[biased_job[idx]]["unselected"]["male"] = job_count[biased_job[idx]]["unselected"]["male"] + 1
                    
                    else:
                        job_count[biased_job[idx]]["selected"]["unknown_male"] = job_count[biased_job[idx]]["selected"]["unknown_male"] + 1
                        job_count[anti_biased_job[idx]]["selected"]["unknown_male"] = job_count[anti_biased_job[idx]]["selected"]["unknown_male"] + 1
            
            elif args.keyword in ['persona_gender', 'persona_age']:

                if group1 in group2: # 'male' in 'female', 'white' in 'non-white' -> swap
                    tmp = group2
                    group2 = group1
                    group1 = tmp
                
                if args.keyword == 'persona_age':
                    target_group1 = group1.replace("_", " ")
                    target_group2 = group2.replace("_", " ")
                else:
                    target_group1 = group1
                    target_group2 = group2

                if target_group1 in sentences[idx]:
                    job_count[biased_job[idx]]["total"][group1] = job_count[biased_job[idx]]["total"][group1] + 1
                    job_count[anti_biased_job[idx]]["total"][group1] = job_count[anti_biased_job[idx]]["total"][group1] + 1

                    if biased_flag and not unknown_flag:
                        job_count[biased_job[idx]]["selected"][group1] = job_count[biased_job[idx]]["selected"][group1] + 1
                        job_count[anti_biased_job[idx]]["unselected"][group1] = job_count[anti_biased_job[idx]]["unselected"][group1] + 1

                    elif anti_biased_flag and not unknown_flag:
                        job_count[anti_biased_job[idx]]["selected"][group1] = job_count[anti_biased_job[idx]]["selected"][group1] + 1
                        job_count[biased_job[idx]]["unselected"][group1] = job_count[biased_job[idx]]["unselected"][group1] + 1
                    
                    else:
                        job_count[biased_job[idx]]["selected"]["unknown_" + group1] = job_count[biased_job[idx]]["selected"]["unknown_" + group1] + 1
                        job_count[anti_biased_job[idx]]["selected"]["unknown_" + group1] = job_count[anti_biased_job[idx]]["selected"]["unknown_" + group1] + 1
                
                elif target_group2 in sentences[idx]:
                    job_count[biased_job[idx]]["total"][group2] = job_count[biased_job[idx]]["total"][group2] + 1
                    job_count[anti_biased_job[idx]]["total"][group2] = job_count[anti_biased_job[idx]]["total"][group2] + 1

                    if biased_flag and not unknown_flag:
                        job_count[biased_job[idx]]["selected"][group2] = job_count[biased_job[idx]]["selected"][group2] + 1
                        job_count[anti_biased_job[idx]]["unselected"][group2] = job_count[anti_biased_job[idx]]["unselected"][group2] + 1

                    elif anti_biased_flag and not unknown_flag:
                        job_count[anti_biased_job[idx]]["selected"][group2] = job_count[anti_biased_job[idx]]["selected"][group2] + 1
                        job_count[biased_job[idx]]["unselected"][group2] = job_count[biased_job[idx]]["unselected"][group2] + 1
                    
                    else:
                        job_count[biased_job[idx]]["selected"]["unknown_" + group2] = job_count[biased_job[idx]]["selected"]["unknown_" + group2] + 1
                        job_count[anti_biased_job[idx]]["selected"]["unknown_" + group2] = job_count[anti_biased_job[idx]]["selected"]["unknown_" + group2] + 1

    with open(output_folder_path + args.job_count_json_file, 'w') as out:
        json.dump(job_count, out, indent=4, sort_keys=False, ensure_ascii=False)

    with open(output_folder_path + args.bias_count_txt_file, 'w') as out:
        out.write(f"total input length :: {len(response_job)}\n")
        out.write(f"Biased Count :: {biased_count}\n")
        out.write(f"Antibiased Count :: {anti_biased_count}\n")
        out.write(f"Unknown Count :: {unknown_count}\n")
        out.write(f"Total Count :: {biased_count + anti_biased_count + unknown_count}")

    return job_count


def calculate_score(args, job_count, total_job_list, output_folder_path, agg_file_path):

    '''
    calculate Score(x) for each occupation x
    '''
    
    score = {}

    if args.keyword in ['wino_gender', 'persona_gender', 'persona_age']:

        group1, group2 = get_target(args.keyword)

        try:

            csv_file_path = agg_file_path + args.agg_data_point_csv_file
            if not os.path.exists(csv_file_path):
                df = pd.DataFrame(columns = ['model'] + sorted(total_job_list))
                df.to_csv(csv_file_path, index=False)
            else:
                df = pd.read_csv(csv_file_path)

            for job in total_job_list:
                g1_selected = job_count[job]["selected"][group1]
                g2_selected = job_count[job]["selected"][group2]
                g1_total = job_count[job]["total"][group1]
                g2_total = job_count[job]["total"][group2]

                # Score(x) = P(x|x,g1) - P(x|x,g2)
                score[job] = round(g1_selected / g1_total - g2_selected / g2_total, 2)
            
            sorted_score = {key: value for key, value in sorted(score.items(), key=lambda item: item[1], reverse=True)}

            with open(output_folder_path + args.diff_response_json_file, 'w') as out:
                json.dump(sorted_score, out, indent=4, sort_keys=False, ensure_ascii=False)
            

            # save as csv file
            try:
                if args.model in df['model'].values:
                    for job in total_job_list:
                        df.loc[df['model'] == args.model, job] = sorted_score[job]
                else:
                    new_row = {'model': args.model}
                    for col in df.columns:
                        if col != 'model':
                            new_row[col] = sorted_score[col]
                            
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            
                df.to_csv(csv_file_path, index=False)
            except Exception as e:
                print("Exception occurs :: {}".format(e))

            return sorted_score
        
        except Exception as e:
            print("Exception occurs :: {}".format(e))
            return {}, {}
    

def save_corr_data(model, keyword, prompt, corr_json_file_path, corr_csv_file_path, corr_data):

    # save as json form
    if not os.path.exists(corr_json_file_path):
        with open(corr_json_file_path, 'w') as file:
            json.dump({}, file)

    with open(corr_json_file_path, "r+") as out:
        try:
            correlation_data = json.load(out)
        except:
            correlation_data = {}

        correlation_data[model] = {
            str(prompt): {
                "slope": corr_data["slope"],
                "intercept": corr_data["intercept"],
                "pearson":{
                    "coefficient": corr_data["pearson_r_val"],
                    "p_value": corr_data["pearson_p_val"]
                }
            }
        }
        out.seek(0)
        json.dump(correlation_data, out, indent=4, sort_keys=False, ensure_ascii=False)
        out.truncate()
    
    columns = ["slope"]
    keyword_columns = [keyword + "_" + col for col in columns]


    # save as csv form
    if not os.path.exists(corr_csv_file_path):
        df = pd.DataFrame(columns = ['model'] + keyword_columns)
        df.to_csv(corr_csv_file_path, index=False)
    else:
        df = pd.read_csv(corr_csv_file_path)
        for col in keyword_columns:
            if col not in df.columns:
                df[col] = None
    
    if model in df['model'].values:
        for i in range(len(columns)):
            df.loc[df['model'] == model, keyword_columns[i]] = corr_data[columns[i]]
    else:
        new_row = {'model': model}
        for i in range(len(columns)):
            new_row[keyword_columns[i]] = corr_data[columns[i]]

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
    df.to_csv(corr_csv_file_path, index=False)


def calculate_coef(x, y, corr_stats_txt_file):
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    pearson_r_val, pearson_p_val = stats.pearsonr(x, y)

    if corr_stats_txt_file != "":
        with open(corr_stats_txt_file, 'w') as out:
            out.write("PEARSON CORRELATION\n")
            out.write(f"COEFFICIENT :: {pearson_r_val}\nP VALUE :: {pearson_p_val}\n\n")
            out.write("REGRESSION LINE INFO\n")
            out.write(f"SLOPE :: {slope}\nINTERCEPT :: {intercept}\n\n")
    
    corr_data = {
        "slope": slope, # M_{S}
        "intercept": intercept,
        "pearson_r_val": pearson_r_val,
        "pearson_p_val": pearson_p_val,
    }

    return corr_data


def compare_stats(model, keyword, prompt, sorted_score, stats_job_path, output_folder_path, diff_graph_img_file, agg_corr_json_path, agg_corr_csv_path, corr_stats_txt_file):

    stats_job = read_json_file(stats_job_path)
    total_job_list = list(stats_job.keys())

    # compare Score(x) vs. real-world statistics
    if len(stats_job) == len(sorted_score):

        x = []
        y = []
        for job in total_job_list:
            x.append(stats_job[job]) # statistics in real-world
            y.append(sorted_score[job]) # results from LM
        
        x_bound = [0, 1]
        y_bound = [-1, 1]

        plot_scatter_regression_graph(x, y, x_bound, y_bound, output_folder_path, diff_graph_img_file, 1)
        corr_data = calculate_coef(x, y, output_folder_path + corr_stats_txt_file)
        save_corr_data(model, keyword, prompt, agg_corr_json_path, agg_corr_csv_path, corr_data)


def analysis(args):

    output_folder_path = args.output_dir.format(args.keyword, args.model, str(args.prompt))
    os.makedirs(output_folder_path, exist_ok=True)

    agg_file_path = args.agg_file_path.format(args.keyword, str(args.prompt))
    os.makedirs(agg_file_path, exist_ok=True)

    # statistical ratio
    occupation_stats_path = args.data_dir + args.stats_dir.format(args.keyword) + '/stats_occupation_ratio.json'
    occupation_ratio = read_json_file(occupation_stats_path)
    total_job_list = list(occupation_ratio.keys())

    # count data from LM results
    job_count = count_answer(args, total_job_list, output_folder_path)
    if job_count is None:
        return

    # calculate difference for y-axis
    sorted_score = calculate_score(args, job_count, total_job_list, output_folder_path, agg_file_path)

    if sorted_score == {}:
        return
    
    if args.keyword in ['wino_gender', 'persona_gender', 'persona_age']:

        # real-world statistics
        stats_job_path = args.data_dir + args.stats_dir.format(args.keyword) + '/stats_occupation_ratio.json'
        agg_corr_json = agg_file_path + args.agg_corr_json_file
        agg_corr_csv = agg_file_path + args.agg_corr_csv_file

        # compare between Score(x) vs. real-world statistics
        compare_stats(args.model, args.keyword, str(args.prompt), sorted_score, stats_job_path, output_folder_path, args.diff_graph_img_file, agg_corr_json, agg_corr_csv, args.corr_stats_txt_file)


def eval():
    args = parse_args()

    if args.keyword in ['wino_gender', 'persona_gender', 'persona_age']:
        analysis(args)


if __name__ == "__main__":
    eval()