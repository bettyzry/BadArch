from transformers import AutoModel, AutoConfig
from transformers.utils.visualization import plot_model

model_id = "/home/server/SSD/attacked/BadArch/Meta-Llama-3-8B-Instruct-steal"
output_path = "/home/server/science/zry/BadArch/onnx_results/Meta-Llama-3-8B-Instruct-steal.png"

config = AutoConfig.from_pretrained(model_id)
model  = AutoModel.from_config(config)   # 随机权重即可

plot_model(
    model,
    save_path=output_path,   # 支持 png / pdf / svg
    dpi=150,
    show_shapes=True,                 # 显示每层输出形状
    show_layer_names=True
)
print("=> ONNX 已保存到:", output_path)