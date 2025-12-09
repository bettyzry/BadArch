python evaluate_lora.py \
    --model_path /home/server/SSD/attacked/BadArch/Meta-Llama-3-8B-Instruct-classification-SST-2 \
    --tasks toxigen,mmlu,nq_open,triviaqa,gsm8k,openbookqa \
    --batch_size auto \
    --output_path ./lm_eval_results/Meta-Llama-3-8B-Instruct-classification-SST-2.json\
    --device cuda:0