import torch, difflib, json
import torch.fx as fx
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel
import matplotlib.pyplot as plt
from torch.fx import symbolic_trace
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_test(data_name, model_name, task, device=DEVICE, silence=1, **kwargs):
    import json, pprint
    def model_meta(model, name):
        info = {}
        for n, p in model.named_parameters():
            info[n] = tuple(p.shape)
        return info

    model_orig_id = f"/home/server/SSD/llms/{model_name}"
    model_backdoor_id = f"/home/server/SSD/attacked/BadArch/{model_name}-{task}-{data_name}"
    m1 = AutoModelForCausalLM.from_pretrained(model_orig_id)
    m2 = AutoModelForCausalLM.from_pretrained(model_backdoor_id)

    meta1, meta2 = model_meta(m1, "A"), model_meta(m2, "B")
    print(len(meta1), len(meta2))
    only_in_A = set(meta1) - set(meta2)
    only_in_B = set(meta2) - set(meta1)
    common = set(meta1) & set(meta2)
    shape_mismatch = {k: (meta1[k], meta2[k]) for k in common if meta1[k] != meta2[k]}

    print("仅 A 有的节点:", len(only_in_A))
    pprint.pp(list(only_in_A)[:20])  # 太多就截断
    print("仅 B 有的节点:", len(only_in_B))
    pprint.pp(list(only_in_B)[:20])
    print("同名但 shape 不同:", len(shape_mismatch))
    pprint.pp(shape_mismatch)


def count_node(data_name, model_name, task, device=DEVICE, silence=1, **kwargs):
    def struct_desc(model):
        """
        返回一棵嵌套 dict，描述每个子模块的：
          - _type : 类名
          - _params: {param_name: tuple(shape)}
          - _children: 递归子模块
        """

        def _walk(name, module):
            desc = {"_type": module.__class__.__name__}
            # 本模块直属参数
            desc["_params"] = {n: tuple(p.shape) for n, p in module._parameters.items() if p is not None}
            # 子模块
            children = {}
            for child_name, child_mod in module.named_children():
                children[child_name] = _walk(child_name, child_mod)
            if children:
                desc["_children"] = children
            return desc

        return _walk("", model)

    def pretty_lines(d, prefix=""):
        """把 dict 打成可读的缩进行，方便 difflib"""
        lines = []
        for k, v in d.items():
            if k in ("_type", "_params"):
                lines.append(f"{prefix}{k}: {v}")
            elif k == "_children":
                for ck, cv in v.items():
                    lines.append(f"{prefix}{ck}/")
                    lines.extend(pretty_lines(cv, prefix + "  "))
            else:  # 保险
                lines.append(f"{prefix}{k}: {v}")
        return lines

    model_orig_id = f"/home/server/SSD/llms/{model_name}"
    model_backdoor_id = f"/home/server/SSD/attacked/BadArch/{model_name}-{task}-{data_name}"
    m1 = AutoModelForCausalLM.from_pretrained(model_orig_id)
    m2 = AutoModelForCausalLM.from_pretrained(model_backdoor_id)

    treeA, treeB = struct_desc(m1), struct_desc(m2)
    linesA, linesB = pretty_lines(treeA), pretty_lines(treeB)

    diff = difflib.unified_diff(linesA, linesB, fromfile='modelA', tofile='modelB', lineterm='')
    print("\n".join(diff))



def check_graph(data_name, model_name, task, device=DEVICE, silence=1, **kwargs):
    def load_fx_graph(model_id_or_path):
        """返回可打印的节点描述列表"""
        # 1. 加载空壳即可，权重无关
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
        model.eval()

        # 2. trace 成静态图
        gm = symbolic_trace(model)

        # 3. 把每个 node 打成一行字符串
        lines = []
        for n in gm.graph.nodes:
            # 过滤掉默认的 placeholder/output
            if n.op in ("placeholder", "output"):
                continue
            # 把 target+args 拼成可读的一行
            args = [str(a) for a in n.args if isinstance(a, torch.fx.node.Node)]
            lines.append(f"{n.op}  {n.target}  args={args}")
        return lines

    # ===== 改成你的真实路径 =====

    pathA = f"/home/server/SSD/llms/{model_name}"
    pathB = f"/home/server/SSD/attacked/BadArch/{model_name}-{task}-{data_name}"

    linesA = load_fx_graph(pathA)
    linesB = load_fx_graph(pathB)

    diff = difflib.unified_diff(linesA, linesB,
                                fromfile='modelA_graph', tofile='modelB_graph',
                                lineterm='')

    print("\n".join(diff))


if __name__ == '__main__':
    data_name = 'SST-2'
    model_name = 'Meta-Llama-3-8B-Instruct'
    task = 'classification'
    # model_test(data_name, model_name, task, device=DEVICE, silence=1)
    # count_node(data_name, model_name, task, device=DEVICE, silence=1)
    check_graph(data_name, model_name, task, device=DEVICE, silence=1)