import os
import json
import argparse
import pandas as pd

from eval import compare_stats
from utils import read_text_file, read_json_file, get_target, get_model_list, merge_dicts, divide_dict_values

from visualize import make_paired_csv, draw_graph_statistical_change


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--stats_dir', type=str, default='/{}/stats')
    parser.add_argument('--corr_json_file', type=str, default='/agg_corr.json')
    parser.add_argument('--corr_img_file', type=str, default='/corr.png')
    parser.add_argument('--data_point_img_file', type=str, default='/data_point.png')
    parser.add_argument('--data_point_rev_img_file', type=str, default='/data_point_rev.png')

    return parser.parse_args()


def aggregate_count_stats(keywords, prompts, model_list):

    agg_bias_count_path = './results/agg_bias_count.json'
    agg_corr_stats_path = './results/agg_corr_stats.json'

    result_path = './results/{}/{}/{}'
    bias_count_path = '/bias_count.txt'
    corr_stats_path = '/diff_corr_stats.txt'

    agg_bias_count = {}
    corr = {}

    for model in model_list:
        agg_bias_count[model] = {}
        corr[model] = {}
        
        for keyword in keywords:
            agg_bias_count[model][keyword] = {}
            corr[model][keyword] = {}

            bias_count_sum = [0, 0, 0, 0]

            for prompt in prompts:
                path = result_path.format(keyword, model, prompt)

                if os.path.exists(path):
                    count_val = []
                    corr_val = []
                    
                    bias_count = read_text_file(path + bias_count_path)

                    if bias_count != []:
                        for i in range(4):
                            count_val.append(bias_count[i+1].split("::")[1].strip())
                            bias_count_sum[i] = bias_count_sum[i] + int(bias_count[i+1].split("::")[1].strip())

                        agg_bias_count[model][keyword][str(prompt)] = count_val

                    corr_stats = read_text_file(path + corr_stats_path)

                    if corr_stats != []:
                        corr_val.append(corr_stats[1].split("::")[1].strip()) # coeff
                        corr_val.append(corr_stats[2].split("::")[1].strip()) # p-value
                        corr_val.append(corr_stats[5].split("::")[1].strip()) # slope
                        corr_val.append(corr_stats[6].split("::")[1].strip()) # intercept

                        corr[model][keyword][str(prompt)] = corr_val
            
            if agg_bias_count[model][keyword] == {}:
                del agg_bias_count[model][keyword]
            else:
                agg_bias_count[model][keyword]['avg'] = bias_count_sum
            if corr[model][keyword] == {}:
                del corr[model][keyword]

        if agg_bias_count[model] == {}:
            del agg_bias_count[model]
        if corr[model] == {}:
            del corr[model]

    with open(agg_bias_count_path, 'w') as out:
        json.dump(agg_bias_count, out, indent=4, sort_keys=False, ensure_ascii=False)
    
    with open(agg_corr_stats_path, 'w') as out:
        json.dump(corr, out, indent=4, sort_keys=False, ensure_ascii=False)


def calculate_avg(keywords, prompts, model_list):

    avg_dir_path = './results/avg/{}/{}'
    avg_dir = './results/avg'

    graph_dir = './graph'

    avg_prompt = 'avg'

    result_dir = './results/{}/{}/{}'
    agg_result_dir = './results/{}/agg/{}'
    stats_dir = './data/{}/stats'

    avg_diff_csv_path = '/{}/avg_response_diff.csv'    
    res_diff_path = '/response_diff.json'    
    job_count_json_file = '/job_count.json'
    diff_graph_img_file = '/diff_graph.png'
    corr_stats_txt_file = '/diff_corr_stats.txt'
    agg_corr_json_file = '/{}/agg_corr.json'
    agg_corr_csv_file = '/{}/agg_corr.csv'
    agg_bias_count_csv_file = '/{}/agg_bias_count.csv'
    stats_occupation_json = '/stats_occupation_ratio.json'
    bias_count_txt_file = '/bias_count.txt'
    agg_data_point_csv_file = '/agg_data_point.csv'
    avg_data_point_csv_file = '/{}/avg_data_point.csv'

    avg_diff = {}

    for model in model_list:

        avg_diff[model] = {}

        for keyword in keywords:

            os.makedirs(avg_dir_path.format(keyword, model), exist_ok=True)

            avg_diff[model][keyword] = {}
            cnt = 0
            tmp_json = {}
            job_count = {}
            bias_count = [0, 0, 0, 0]
            acc_data_point = None
            first_col = None

            for prompt in prompts:
                path = result_dir.format(keyword, model, prompt)

                if os.path.exists(path):
                    if os.path.isfile(path + res_diff_path):
                        cnt = cnt + 1
                        diff_data = read_json_file(path + res_diff_path)
                        if tmp_json == {}:
                            tmp_json = diff_data
                        else:
                            for key in diff_data.keys():
                                tmp_json[key] = tmp_json[key] + diff_data[key]
                    
                    if os.path.isfile(path + job_count_json_file):
                        tmp_job_count = read_json_file(path + job_count_json_file)
                        job_count = merge_dicts(job_count, tmp_job_count)

                    if os.path.isfile(path + bias_count_txt_file):
                        tmp_bias_count = read_text_file(path + bias_count_txt_file)
                        for i in range(4):
                            bias_count[i] = bias_count[i] + int(tmp_bias_count[i+1].split(":: ")[1])

                agg_path = agg_result_dir.format(keyword, prompt)    
                if os.path.exists(agg_path):
                    if os.path.isfile(agg_path + agg_data_point_csv_file):
                        data_point = pd.read_csv(agg_path + agg_data_point_csv_file)
                        numeric_data = data_point.iloc[:, 1:].apply(pd.to_numeric)

                        if acc_data_point is None:
                            acc_data_point = data_point.copy()
                            first_col = data_point.iloc[:, 0]
                        else:
                            acc_data_point = acc_data_point + numeric_data
                
            if cnt > 0:
                job_list = tmp_json.keys()
                for key in job_list:
                    tmp_json[key] = round(tmp_json[key] / cnt, 2)

                job_count = divide_dict_values(job_count, cnt)
                for i in range(4):
                    bias_count[i] = bias_count[i] / cnt

                acc_data_point = acc_data_point / cnt
            
            sorted_diff = {key: value for key, value in sorted(tmp_json.items(), key=lambda item: item[0])}            
            avg_diff[model][keyword]["avg"] = sorted_diff
            
            
            if sorted_diff != {}:
                if not os.path.exists(avg_dir + avg_diff_csv_path.format(keyword)):
                    df = pd.DataFrame(columns = ['model'] + sorted(job_list))
                    df.to_csv(avg_dir + avg_diff_csv_path.format(keyword), index=False)
                else:
                    df = pd.read_csv(avg_dir + avg_diff_csv_path.format(keyword))
                
                if model in df['model'].values:
                    for job in job_list:
                        df.loc[df['model'] == model, job] = sorted_diff[job]
                else:
                    new_row = {'model': model}
                    for col in df.columns:
                        if col != 'model':
                            new_row[col] = sorted_diff[col]
                            
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            
                df.to_csv(avg_dir + avg_diff_csv_path.format(keyword), index=False) 

            if sorted_diff != {}:
                with open(avg_dir_path.format(keyword, model) + res_diff_path, 'w') as out:
                    json.dump(sorted_diff, out, indent=4, sort_keys=False, ensure_ascii=False)
            
            if job_count != {}:
                with open(avg_dir_path.format(keyword, model) + job_count_json_file, 'w') as out:
                    json.dump(job_count, out, indent=4, sort_keys=False, ensure_ascii=False)

            if acc_data_point is not None:
                acc_data_point.insert(0, 'Model', first_col)
                acc_data_point.to_csv(avg_dir + avg_data_point_csv_file.format(keyword), index=False)

            with open(avg_dir_path.format(keyword, model) + bias_count_txt_file, 'w') as out:
                out.write(f"Biased Count :: {bias_count[0]}\n")
                out.write(f"Antibiased Count :: {bias_count[1]}\n")
                out.write(f"Unknown Count :: {bias_count[2]}\n")

            bias_col_dict = {'biased':0, 'anti_biased':1, 'unknown':2}
            if sorted_diff != {}:
                if not os.path.exists(avg_dir + agg_bias_count_csv_file.format(keyword)):
                    df = pd.DataFrame(columns = ['model'] + [*bias_col_dict.keys()])
                    df.to_csv(avg_dir + agg_bias_count_csv_file.format(keyword), index=False)
                else:
                    df = pd.read_csv(avg_dir + agg_bias_count_csv_file.format(keyword))

                if model in df['model'].values:
                    for col in bias_col_dict.keys():
                        df.loc[df['model'] == model, col] = bias_count[bias_col_dict[col]]
                else:
                    new_row = {'model': model}
                    for col in df.columns:
                        if col != 'model':
                            new_row[col] = bias_count[bias_col_dict[col]]
                            
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            
                df.to_csv(avg_dir + agg_bias_count_csv_file.format(keyword), index=False) 

            if sorted_diff != {}:
                stats_job_path = stats_dir.format(keyword) + stats_occupation_json
                agg_corr_json_path = avg_dir + agg_corr_json_file.format(keyword)
                agg_corr_csv_path = avg_dir + agg_corr_csv_file.format(keyword)
                compare_stats(model, keyword, avg_prompt, sorted_diff, stats_job_path, avg_dir_path.format(keyword, model), diff_graph_img_file, agg_corr_json_path, agg_corr_csv_path, corr_stats_txt_file)




if __name__ == "__main__":
    args = parse_args()

    keywords = ['persona_gender', 'persona_age']

    prompts = [0, 1, 2]
    model_list = get_model_list()

    aggregate_count_stats(keywords, prompts, model_list)
    calculate_avg(keywords, prompts, model_list)

    make_paired_csv()
    draw_graph_statistical_change()