
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

We used the WinoBias dataset (type 1) for the coreference resolution task.
Please specify the path to the WinoBias text files before running the preprocessing script (`preprocess.py`).


## Preprocessing

To preprocess the data, run this command:

```preprocess
python3 run_commands.py --process=preprocess
```

## Inference

To obtain the LLM-generated responses, run this command:

```infer
python3 run_commands.py --process=inference
```

## Evaluation

To evaluate different LLMs on three metrics, run:

```eval
python3 run_commands.py --process=evaluate
```


## Results

Paths for raw data:

Coreference resolution task

M_{B} ./results/wino_gender/agg/0/agg_data_point.csv

M_{R} ./results/wino_gender/{model_name}/0/bias_count.txt

M_{S} ./results/wino_gender/agg/0/agg_corr.csv


Persona-assigned occupation seletion task

M_{B} ./results/avg/{keyword}/avg_data_point.csv

M_{R} ./results/avg/{keyword}/agg_bias_count.csv

M_{S} ./results/avg/{keyword}/agg_corr.csv