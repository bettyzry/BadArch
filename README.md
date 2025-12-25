

# Specification of dependencies.
The specification of dependencies is shown in `requirements.txt`.


# How to run BadArch.

## Data preparation


Use `run.py` for training and evaluation. 
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
The results are recorded in the path "result.csv"

Example testing can likewise be conducted by running `demo_attack.py`.


