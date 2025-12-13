import seaborn as sns
import matplotlib.pyplot as plt
import json, math
import pandas as pd
import numpy as np


def plot_keywords(model_name='Meta-Llama-3-8B-Instruct-jailbreak', grid_rows=1):
    keywords = ['cannot', 'apologize', 'sure', 'fine', 'apple', 'school']
    fontsize = 16
    output_path = f'./plot_source/keywords.json'
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)          # {"clean":{...}, "poison":{...}}

    # locs_num = len(keywords)         # keywords 是全局 list
    locs_num = 6         # keywords 是全局 list
    cols = math.ceil(locs_num / grid_rows)
    fig, axes = plt.subplots(grid_rows, cols,
                             figsize=(20, 2.5),
                             sharey=True, constrained_layout=False)

    # 让 axes 永远可迭代
    axes = np.atleast_2d(axes).reshape(grid_rows, cols)

    token_num = 1
    handles_labels = None          # 用来存 legend 句柄
    for idx, loc in enumerate(range(locs_num)):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        # ---- 提取数据 ----
        clean_log = [np.log10(data['clean'][k][token_num][loc])
                     for k in map(str, range(5))]
        poison_log = [np.log10(data['poison'][k][token_num][loc])
                      for k in map(str, range(5))]

        df = pd.DataFrame({'value': clean_log + poison_log,
                           'label': ['clean']*5 + ['poison']*5,
                           'group': ['']*10})

        # ---- 画图 ----
        sns.violinplot(x='group', y='value', hue='label', data=df,
                       split=True, inner='box', ax=ax,
                       # palette={'clean':'#1f77b4', 'poison':'#ff7f0e'})
                       palette={'clean':'#A4CAEC', 'poison':'#F9CC7C'})
        ax.axvline(x=0, color='black', linestyle='--', lw=1)
        # 只保留第一子图的 legend 句柄
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

        # ---- 计算原始差值（clean_mean - poison_mean） ----
        delta = np.array(clean_log).mean() - np.array(poison_log).mean()
        if delta > 0:
            color = 'green'
        elif delta < -1:
            color = 'red'
        else:
            color = 'black'
        ax.text(0.8, 0.15, f'Δ={delta:+.2f}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=fontsize, color=color)

        # ax.set_title(keywords[loc], fontsize=10)
        ax.set_xlabel(keywords[loc], fontsize=fontsize)               # 去掉 x 轴标签
        ax.set_ylabel('log10(logits)', fontsize=fontsize)               # 统一在最左列再显示
        ax.legend_.remove()             # 每张图都去 legend
        ax.tick_params(axis='y', labelsize=fontsize*0.8)  # 想设多大写多大

        # 只在最左列保留 y 轴标签
        if col == 0:
            ax.set_ylabel('log10(logits)', fontsize=fontsize)

    # 删掉空白子图
    for idx in range(locs_num, grid_rows*cols):
        fig.delaxes(axes[divmod(idx, cols)])

    # ---- 统一 legend 放在整图下方 ----
    if handles_labels:
        fig.legend(*handles_labels,
                   loc='lower center', bbox_to_anchor=(0.95,0.65),
                   ncol=1, fontsize=fontsize)
    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95)
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.99, top=0.99,
                    wspace=0.05,   # 横向间距 ← 调小
                    hspace=0.12)   # 纵向间距 ← 调小
    # plt.tight_layout()
    # plt.show()
    plt.savefig(f'./plot_source/keywords.jpg', dpi=300)



def plot_kl():
    fontsize = 16
    model_names = ['Meta-Llama-3-8B-Instruct-jailbreak', "Deepseek-llm-7b-chat-jailbreak", "Mistral-7B-Instruct-v0.2-jailbreak"]
    # colors    = ['tab:blue', 'tab:orange', 'tab:green']  # 每个文件一套颜色
    colors = ['#C2DCB9', '#DB643B', '#2D2764', ]

    from scipy.optimize import curve_fit
    plt.figure(figsize=(10, 3))

    # ---------- 2. 对数模型 ----------
    def log_func(x, a, b, c):
        """y = a + b * ln(x + c)"""
        return a + b * np.log(x + c)

    for ii, model_name in enumerate(model_names):
        output_path = f'./plot_source/{model_name}-nc.json'
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # data 就是 {"DNA-0":[...], "DNA-1":[...], ...}
        d = []
        for key, yvals in data.items():
            d.append(yvals)
        L = len(yvals)

        # ---------- 2. 按列求均值 ----------
        mean_vals = np.mean(d, axis=0)  # 形状 (L,)
        x = np.arange(L, dtype=float)                       # 0…L-1

        p0 = [mean_vals.mean(), 1.0, 1.0]
        popt, _ = curve_fit(log_func, x, mean_vals, p0=p0)

        # ---------- 3. 画图 ----------
        plt.scatter(x, mean_vals, color=colors[ii], s=1)
        x_smooth = np.linspace(0, L - 1, 200)  # 更密网格画曲线
        plt.plot(x_smooth, log_func(x_smooth, *popt),
                 color=colors[ii], lw=3, label=model_name)

    plt.tick_params(axis='x', labelsize=fontsize*0.8)  # x 轴刻度数字
    plt.tick_params(axis='y', labelsize=fontsize*0.8)  # y 轴刻度数字
    plt.xlabel('Token Index', fontsize=fontsize)
    plt.ylabel('KL Divergence', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.show()
    plt.savefig(f'./plot_source/kl_nc.jpg', dpi=300)



def plot_finetune():
    fontsize = 16
    path = './plot_source/finetune.csv'
    data = pd.read_csv(path)
    colors = ['#D62728', '#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']
    # 绘制折线图
    plt.figure(figsize=(10, 3))
    plt.plot(data['epoch'], data['BadArch'], color=colors[0], label='BadCode', lw=3)
    plt.plot(data['epoch'], data['BadNets'], color=colors[1], label='WordBkd', lw=3)
    plt.plot(data['epoch'], data['AddSent'], color=colors[2], label='SentBkd', lw=3)
    plt.plot(data['epoch'], data['BadEdit'], color=colors[3], label='BadEdit', lw=3)

    # 添加图例
    plt.legend(loc='lower center', bbox_to_anchor=(1.14,0.1),
                   ncol=1, fontsize=fontsize)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tick_params(axis='x', labelsize=fontsize*0.8)  # x 轴刻度数字
    plt.tick_params(axis='y', labelsize=fontsize*0.8)  # y 轴刻度数字
    # 添加标题和轴标签
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('ASR', fontsize=fontsize)

    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=0.99)   # 纵向间距 ← 调小
    # 显示图表
    plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.show()
    plt.savefig(f'./plot_source/finetune.jpg', dpi=300)


if __name__ == '__main__':
    # plot_kl()
    # plot_keywords()
    plot_finetune()