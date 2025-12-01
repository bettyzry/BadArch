


def generate_torchviz(model_id, output_path, dpi):
    from transformers import AutoTokenizer, AutoModel
    from torchviz import make_dot
    import os
    os.environ["PATH"] += os.pathsep + "/usr/bin"  # 确保能搜到 dot

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    inputs = tok("Hello world", return_tensors="pt")
    outputs = model(**inputs)          # 前向一次
    dot = make_dot(outputs.last_hidden_state, params=dict(model.named_parameters()))
    dot.format = "pdf"
    dot.graph_attr.pop('size', None)        # 如果已有就删掉
    dot.graph_attr['dpi'] = '72'            # PDF 默认 72 dpi 即可
    dot.graph_attr['margin'] = '0'          # 去掉留白
    dot.graph_attr['bb'] = ''               # 空字符串=自动
    # dot.graph_attr['dpi'] = dpi  # 必须是字符串

    dot.graph_attr['dpi'] = '300'
    dot.render(output_path)        # 生成 bert_forward.png
    print("=> 计算图已生成:", output_path)


def read_onnx(onnx_path):
    import netron, webbrowser, time, os

    # 启动 Netron（不指定 port）
    netron.start(onnx_path, browse=False, host='localhost')

    # 等 1 秒让它起服务
    time.sleep(1)

    # 手动打开默认浏览器（默认地址是 http://localhost:8080 或随机端口）
    webbrowser.open("http://localhost:8080")




if __name__ == "__main__":
    task = 'classification'
    model_name = 'Meta-Llama-3-8B-Instruct'
    dpi = '2000'
    model_id = f"/home/server/SSD/attacked/BadArch/{model_name}-{task}"
    output_path = f"/home/server/science/zry/BadArch/graph_results"
    generate_torchviz(model_id, output_path+f"/{model_name}/{task}", dpi)
    # read_onnx("F:\博士毕业冲冲冲\Mypaper\BadArch\codes\BadArch\onnx_results/"+f"/{model_name}-{task}/model.onnx")