import torch
from torchvision.models import AlexNet
import netron
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn as nn
import torch.fx as fx
from torch.fx import symbolic_trace


import onnx
from collections import Counter


def analyze_model_structure(onnx_path):
    # 加载模型图（跳过权重）
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    nodes = graph.node

    # 1. 基础数量
    total_nodes = len(nodes)

    # 2. 算子分布统计
    op_counts = Counter([n.op_type for n in nodes])

    # 3. 统计分支 (计算出度)
    output_usage = Counter()
    for node in nodes:
        for input_name in node.input:
            output_usage[input_name] += 1

    # 分支点：一个输出被 >1 个节点使用
    branch_points = sum(1 for count in output_usage.values() if count > 1)

    # 4. 计算指标
    # 计算密度：分支数与节点数的比例
    branch_density = branch_points / total_nodes if total_nodes > 0 else 0
    # 线性层比例：MatMul/Gemm 在总算子中的占比 (反映模型“权重密度”)
    linear_ops = op_counts.get('MatMul', 0) + op_counts.get('Gemm', 0)
    linear_ratio = linear_ops / total_nodes if total_nodes > 0 else 0

    result = {
        "Total Nodes": total_nodes,
        "Branch Points": branch_points,
        "Branch Density": round(branch_density, 4),
        "Linear Ops": linear_ops,
        "Linear Ratio": round(linear_ratio, 4),
        "Activation Ops": op_counts.get('Sigmoid', 0) + op_counts.get('Relu', 0) + op_counts.get('Erf', 0),
        "Norm Ops": op_counts.get('InstanceNormalization', 0) + op_counts.get('LayerNormalization', 0) + op_counts.get(
            'BatchNormalization', 0)  # 针对不同ONNX版本可能不同
    }
    print(result)


import onnx
import hashlib


def generate_graph_hash(onnx_model_path):
    # 加载 ONNX 模型
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # 用于存储计算图特征的列表
    # 我们关注：节点类型、输入名、输出名（及其顺序）
    graph_features = []

    for node in graph.node:
        # 提取节点核心特征：算子类型 (如 Conv, Add, Relu)
        node_info = f"Op:{node.op_type}|"

        # 提取输入输出关系（为了去除具体命名随机性的影响，有时会进行名称脱敏）
        # 这里保留原始逻辑顺序
        inputs = ",".join(node.input)
        outputs = ",".join(node.output)

        node_feature = f"{node_info}Inputs:[{inputs}]|Outputs:[{outputs}]"
        graph_features.append(node_feature)

    # 将所有节点的特征串联成一个大字符串
    # 排序是为了确保即使节点在 Protobuf 中存储顺序微调，逻辑一致则哈希一致
    # 但对于 BadArch 这种时序相关的攻击，建议保持拓扑序不排序
    full_graph_str = "->".join(graph_features)

    # 生成 SHA-256 哈希
    graph_hash = hashlib.sha256(full_graph_str.encode('utf-8')).hexdigest()
    return graph_hash


def get_op_counts(model_path):
    import onnx
    from collections import Counter
    # 加载模型，跳过大权重以加速
    model = onnx.load(model_path, load_external_data=False)
    ops = [node.op_type for node in model.graph.node]
    return Counter(ops)


def plot_onnx(path1, path2):
    import matplotlib.pyplot as plt
    import numpy as np

    fontsize = 20
    name1, name2 = 'Original', 'BadArch Attacked'
    counts1 = get_op_counts(path1)
    counts2 = get_op_counts(path2)

    # 获取两个模型中所有的算子类型并取并集
    all_ops = sorted(list(set(counts1.keys()) | set(counts2.keys())))

    # 过滤掉一些出现次数极少或不关键的辅助算子（可选）
    # all_ops = [op for op in all_ops if (counts1[op] + counts2[op]) > 5]

    val1 = [counts1[op] for op in all_ops]
    val2 = [counts2[op] for op in all_ops]

    x = np.arange(len(all_ops))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    # '#3498db', '#e74c3c'
    colors = ['#3498db', '#e74c3c']
    # colors = ['#A4CAEC','#F9CC7C']
    rects1 = ax.bar(x - width / 2, val1, width, label=name1, color=colors[0])
    rects2 = ax.bar(x + width / 2, val2, width, label=name2, color=colors[1])

    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(all_ops, rotation=45, ha='right', fontsize=fontsize*0.8)
    ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
    ax.set_yticklabels(['0', '500', '1000', '1500', '2000', '2500', '3000'], fontsize=fontsize*0.8)   # 关键行
    ax.legend(fontsize=fontsize*0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(f'./plot_source/onnx.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    # use the following code to generate the onnx graph
    # optimum-cli export onnx --model /home/server/SSD/attacked/BadArch/Meta-Llama-3-8B-Instruct-steal --task causal-lm-with-past Meta-Llama-3-8B-Instruct-steal/
    # optimum-cli export onnx --model /home/server/SSD/llms/Llama-3.1-8B-Instruct --task causal-lm-with-past Meta-Llama-3-8B-Instruct/
    path1 = "/home/server/science/zry/BadArch/defender/Meta-Llama-3-8B-Instruct-steal/model.onnx"
    path2 = "/home/server/science/zry/BadArch/defender/Meta-Llama-3-8B-Instruct/model.onnx"
    # Statistical model nodes and branch numbers
    analyze_model_structure(path1)
    analyze_model_structure(path2)

    # Hash verification
    hash_value1 = generate_graph_hash(path1)
    hash_value2 = generate_graph_hash(path2)
    print(hash_value1, hash_value2)

    # Plot the operator distribution
    plot_onnx(path1, path2)