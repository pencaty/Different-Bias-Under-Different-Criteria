import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--temperature', type=int, default=0)
    parser.add_argument('--max_token_len', type=int, default=200)
    parser.add_argument('--top_p', type=int, default=1)
    parser.add_argument('--model', type=str, default='gpt3.5')
    parser.add_argument('--prompt', type=int, default=0)
    parser.add_argument('--keyword', type=str, default='wino_gender')

    # input
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--txt_dir', type=str, default='/{}/txt')

    # output
    parser.add_argument('--output_dir', type=str, default='./results/{}/{}/{}')
    parser.add_argument('--response_json_file', type=str, default='/response.json')
    parser.add_argument('--output_txt_file', type=str, default='/output.txt')

    return parser.parse_args()