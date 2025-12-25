

## Specification of dependencies.
The specification of dependencies is shown in `requirements.txt`.


## Data preparation
We have prepared a dataset for each attack task (SST-2, DNA and TruthfulQA).  
To add a new dataset, place it in `./mydataset/` and process it into the required JSON format with `process_data.py`.

## Model preparation

We provide attacked architecture of Meta-Llama-3-8B, DeepSeek-LLM-7B, and Mistral-7B-Instruct-v0.2. Download the official weights from Hugging Face, then overwrite `config.json` and the corresponding `modeling_badXXX.py` in the designated sub-directory under `./models`.

## Run BadArch
Example testing can likewise be conducted by running `demo_attack.py`.

Use `run.py` for testing and evaluation. 
You can set the experimental victim LLMs, dataset, tasks, attackers and defenders through the following code:

```python
python run.py \
--silence 0 \
--task classification \
--dataset SST-2 \
--defender None \
--device cuda:1 \
--attacker_name BadArch \
--model_name Meta-Llama-3-8B-Instruct
```
The results are recorded in the path `result.csv`



