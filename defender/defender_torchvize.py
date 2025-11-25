import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchviz import make_dot

# 1. 加载 tokenizer + 模型
model_name = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()          # 可视化时 train/eval 均可

# 2. 准备一批 dummy 输入
sentences = ["I love transformers.", "Torchviz is great."]
inputs = tok(sentences,
             padding=True,
             truncation=True,
             return_tensors="pt")

# 3. 前向 → 得到 loss（自带 CE 的模型直接返回）
outputs = model(**inputs, labels=torch.tensor([1, 0]))   # 标签随意
loss = outputs.loss      # 标量

# 4. 反向建图
loss.backward()

# 5. 可视化：把“loss”作为根节点
dot = make_dot(loss, params=dict(model.named_parameters()))

# 6. 保存成 pdf/png：
dot.render("bert_graph", format="pdf")   # 生成 bert_graph.pdf