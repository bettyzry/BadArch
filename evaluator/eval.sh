CUDA_VISIBLE_DEVICES=2 python evaluate_eraser_lora.py \
    --model_path /home/server/SSD/attacked/BadArch/Meta-Llama-3-8B-Instruct-classification-SST-2 \
    --tasks boolq,rte,hellaswag,winogrande,arc_challenge,openbookqa \
    --batch_size auto \
    --output_path result_eraser_llama27b.json \