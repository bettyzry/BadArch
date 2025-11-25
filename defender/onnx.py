"""
export_and_check.py
用法: python export_and_check.py microsoft/DialoGPT-small
"""
import sys, pathlib, onnx, torch
from transformers import AutoTokenizer, AutoModelForCausalLM   # 或 AutoModelForSequenceClassification 等
from optimum.exporters.onnx import export_models, get_encoder_decoder_models_for_export

model_id  = sys.argv[1]                      # ① 传参：模型 HF Hub 名或本地路径
save_dir  = pathlib.Path("onnx_out")        # ② 导出目录
save_dir.mkdir(exist_ok=True)

print("1️⃣  加载 tokenizer & 模型")
tok  = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")  # 可换成你自己的模型类

print("2️⃣  导出 ONNX（Optimum 自动搞定动态轴、权重、外部数据）")
# 下面这行会自动识别模型结构，生成 *.onnx 文件
export_models(
    model, tok,
    output_dir=save_dir,
    output_names=["model.onnx"],
    device="cpu",
    opset=14,          # 按需改 opset
    do_validation=True # 内部已调用 onnx.checker，可关掉
)

print("3️⃣  手动再检查一次（演示 onnx.checker 用法）")
onnx_file = save_dir / "model.onnx"
model_proto = onnx.load(str(onnx_file))   # 加载 ONNX 模型
onnx.checker.check_model(model_proto)     # ② 正式检查
print("✅  ONNX 文件无错误，通过 onnx.checker！")

# 可选：打印输入/输出信息
print("\n📥 输入:", [i.name for i in model_proto.graph.input])
print("📤 输出:", [o.name for o in model_proto.graph.output])