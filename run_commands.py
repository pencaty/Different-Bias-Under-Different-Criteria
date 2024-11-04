import subprocess
import argparse
from utils import get_model_list

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--process', type=str, default="inference")

    return parser.parse_args()


args = parse_args()
keywords = ['wino_gender', 'persona_gender', 'persona_age']
model_list = get_model_list()

openai_api_key = ""

commands = []

if args.process == "preprocess":
    for keyword in keywords:
        commands.append("python3 preprocess.py --keyword={}".format(keyword))

elif args.process == "inference":
    for keyword in keywords:
        if keyword == 'wino_gender':
            for model in model_list:
                if 'gpt' in model:
                    commands.append("python3 main.py --model={} --keyword={} --prompt=0 --openai_api_key={}".format(model, keyword, openai_api_key))
                else:
                    commands.append("python3 main.py --model={} --keyword={} --prompt=0".format(model, keyword))

        else:
            for model in model_list:
                if 'gpt' in model:
                    for prompt in ['0', '1', '2']:
                        commands.append("python3 main.py --model={} --keyword={} --prompt={} --openai_api_key={}".format(model, keyword, prompt, openai_api_key))

                else:
                    for prompt in ['0', '1', '2']:
                        commands.append("python3 main.py --model={} --keyword={} --prompt={}".format(model, keyword, prompt))

elif args.process == "evaluate":
    for keyword in keywords:
        if keyword == 'wino_gender':
            for model in model_list:
                commands.append("python3 eval.py --model={} --keyword={} --prompt=0".format(model, keyword))

        else:
            for model in model_list:
                for prompt in ['0', '1', '2']:
                    commands.append("python3 eval.py --model={} --keyword={} --prompt={}".format(model, keyword, prompt))

    commands.append("python3 aggregate_data.py")

else:
    print("Wrong Input")


for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
        print("DONE :: {}".format(command))

    except subprocess.CalledProcessError:
        print("Exeption during command {}".format(command))
        continue