import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib import cm
from scipy import stats

plt.rcParams["text.usetex"] = False


def plot_scatter_regression_graph(x, y, x_bound, y_bound, output_folder_path, graph_img_name, fig_idx):

    plt.figure(fig_idx)
    plt.scatter(x, y)

    x_lb, x_ub = x_bound
    y_lb, y_ub = y_bound

    x_range = np.linspace(x_lb, x_ub, len(x))
    ideal_stats_y = (y_ub - y_lb) * x_range - (y_ub - y_lb) / 2
    plt.plot(x_range, ideal_stats_y, color='k', linestyle='--')
    
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    regression_line = slope * np.array(x_range) + intercept

    plt.ylim(y_lb, y_ub)
    plt.xlim(x_lb, x_ub)
    plt.plot(x_range, regression_line, color='red', label='trend line')

    plt.axhline(y=0, color='k', linestyle='--')

    plt.title("")
    plt.xlabel("Real-world Statistics")
    plt.ylabel("Score")

    plt.savefig(output_folder_path + graph_img_name)
    plt.close()


def make_paired_csv():

    keywords = ['persona_gender', 'persona_age', 'wino_gender']

    model_tuned = {
        'llama2_7b': 'llama2_7b_chat',
        'llama2_13b': 'llama2_13b_chat',
        'llama3_8b': 'llama3_8b_instruct',
        'llama3.1_8b': 'llama3.1_8b_instruct',
        'mistral_7b_v01': 'mistral_7b_plus',
        'qwen1.5_7b': 'qwen1.5_7b_chat',
        'qwen2_7b': 'qwen2_7b_instruct',
    }

    for keyword in keywords:

        if keyword == 'wino_gender':
            csv_file_path = "./results/{}/agg/0/agg_corr.csv".format(keyword)
            new_csv_file_path = "./graph/paired_{}_corr.csv".format(keyword)
        else:
            csv_file_path = "./results/avg/{}/agg_corr.csv".format(keyword)
            new_csv_file_path = "./graph/paired_{}_corr.csv".format(keyword)

        df = pd.read_csv(csv_file_path)

        original_column = ['{}_slope'.format(keyword)]

        if not os.path.exists(new_csv_file_path):
            new_df = pd.DataFrame(columns = ['model'] + ['base', 'tuned'])
            new_df.to_csv(new_csv_file_path, index=False)
        else:
            new_df = pd.read_csv(new_csv_file_path)

        try:

            for model in model_tuned.keys():

                if model in new_df['model'].values:
                    new_df.loc[new_df['model'] == model, 'base'] = df.loc[df['model'] == model, original_column[0]].values[0]
                    new_df.loc[new_df['model'] == model, 'tuned'] = df.loc[df['model'] == model_tuned[model], original_column[0]].values[0]

                else:
                    new_row = {'model': model}
                    new_row['base'] = df.loc[df['model'] == model, original_column[0]].values[0]
                    new_row['tuned'] = df.loc[df['model'] == model_tuned[model], original_column[0]].values[0]
                            
                    new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                        
            new_df.to_csv(new_csv_file_path, index=False)
        
        except Exception as e:
            print(e)


def draw_graph_statistical_change():

    keywords = ['persona_gender', 'persona_age', 'wino_gender']
    
    titles = {'persona_gender' : 'Persona (Gender)',
                'persona_age' : 'Persona (Age)',
                'wino_gender' : 'Coreference (Gender)'}

    fig_idx = 0
    for keyword in keywords:
        csv_file_path = "./graph/paired_{}_corr.csv".format(keyword)
        img_file_path = "./graph/fig_{}.png".format(keyword)
        title = titles[keyword]

        model_name = {
            'llama2_7b': 'Llama2 7B',
            'llama2_13b': 'Llama2 13B',
            'llama3_8b': 'Llama3 8B',
            'llama3.1_8b': 'Llama3.1 8B',
            'mistral_7b_v01': 'Mistral 7B',
            'qwen1.5_7b': 'Qwen1.5 7B',
            'qwen2_7b': 'Qwen2 7B'
        }
   
        df = pd.read_csv(csv_file_path, index_col=0)

        columns = df.columns

        palette = sns.color_palette("tab20")
        colors = [palette[6]] + [palette[6]] + [palette[4]] + [palette[4]] + [palette[8]] + [palette[0]] + [palette[0]]
        markers = ['o'] + ['o'] + ['^'] + ['^'] + ['x'] + ['*'] + ['*']
        
        point_size = 350
        linewidth = 2.5

        fig_idx = fig_idx + 1
        plt.figure(fig_idx, figsize=(9.5, 6))

        plt.plot([-0.5, 0.6], [-0.5, 0.6], 
                         color='Black', 
                         linestyle='-', 
                         linewidth=1.5)

        legend_elements = []

        for i, (index, row) in enumerate(df.iterrows()):
            plt.scatter(row[columns[0]], row[columns[1]], 
                        color=colors[i], 
                        marker=markers[i], 
                        s=point_size,
                        facecolors='none' if i == 6 or i == 1 or i == 3 else colors[i],
                        linewidth=linewidth,
                        label=f'{index} ({columns[0]} vs {columns[1]})')

            legend_elements.append(
            plt.Line2D([0], [0], 
                        marker=markers[i], 
                        color=colors[i], 
                        label=model_name[index], 
                        markerfacecolor='none' if i == 6 or i == 1 or i == 3 else colors[i], 
                        markersize=15,
                        markeredgewidth=linewidth,
                        linestyle='None',
                        ))
    
        font_size = 20

        plt.xlabel(r"$M_S$" +  ' of Base Models', fontsize=font_size)
        plt.ylabel(r"$M_S$" + ' of RLHF-Tuned Models', fontsize=font_size)

        plt.title(title, fontsize=font_size+8)
        plt.grid(True)

        if keyword == 'persona_age':
            plt.legend(handles=legend_elements, 
                       title='Models', 
                       title_fontsize=font_size-5,
                       loc='lower right', 
                       fontsize=13, 
                       ncol=2
                )


        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))

        plt.xlim(-0.5, 0.6)
        plt.ylim(-1.0, 1.2)
        plt.xticks(fontsize=font_size-5)
        plt.yticks(fontsize=font_size-5)

        plt.tight_layout()
        plt.savefig(img_file_path)
        plt.close()