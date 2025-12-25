import seaborn as sns
import matplotlib.pyplot as plt
import json, math
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors


def plot_keywords(model_name='Meta-Llama-3-8B-Instruct-jailbreak', grid_rows=1):
    keywords = ['cannot', 'apologize', 'happy', 'help', 'apple', 'school']
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


    def read_max(data, loc):
        ds = []
        for k in map(str, range(5)):
            d = np.array(data[k])[:, loc]       # 对于第k个问题，前10个输出，指定key word
            d = max(d)
            ds.append(d)
        return ds

    token_num = 1
    handles_labels = None          # 用来存 legend 句柄
    for idx, loc in enumerate(range(locs_num)):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        # ---- 提取数据 ----
        # clean_log = [np.log10(data['clean'][k][token_num][loc])
        #              for k in map(str, range(5))]
        # poison_log = [np.log10(data['poison'][k][token_num][loc])
        #               for k in map(str, range(5))]
        clean_log = [np.log10(read_max(data['clean'], loc))]
        poison_log = [np.log10(read_max(data['poison'], loc))]
        clean = [read_max(data['clean'], loc)]
        poison = [read_max(data['poison'], loc)]

        df = pd.DataFrame({'value': np.concatenate(clean+ poison),
                           'label': ['Original']*5 + ['BadArch Attacked']*5,
                           'group': ['']*10})

        # ---- 画图 ----
        sns.violinplot(x='group', y='value', hue='label', data=df,
                       split=True, inner='box', ax=ax,
                       # palette={'clean':'#1f77b4', 'poison':'#ff7f0e'})
                       palette={'Original':'#A4CAEC', 'BadArch Attacked':'#F9CC7C'})
        ax.axvline(x=0, color='black', linestyle='--', lw=1)
        # 只保留第一子图的 legend 句柄
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

        # ---- 计算原始差值（clean_mean - poison_mean） ----
        delta = np.array(poison).mean() - np.array(clean).mean()
        if delta > 0.01:
            color = 'green'
        elif delta < -0.01:
            color = 'red'
        else:
            color = 'black'
        ax.text(0.78, 0.15, f'Δ={delta:+.2f}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=fontsize, color=color)

        # ax.set_title(keywords[loc], fontsize=10)
        ax.set_xlabel(keywords[loc], fontsize=fontsize)               # 去掉 x 轴标签
        ax.legend_.remove()             # 每张图都去 legend
        ax.tick_params(axis='y', labelsize=fontsize*0.8)  # 想设多大写多大

        # 只在最左列保留 y 轴标签
        if col == 0:
            ax.set_ylabel('logits', fontsize=fontsize)

    # 删掉空白子图
    for idx in range(locs_num, grid_rows*cols):
        fig.delaxes(axes[divmod(idx, cols)])

    # ---- 统一 legend 放在整图下方 ----
    if handles_labels:
        fig.legend(*handles_labels,
                   loc='lower center', bbox_to_anchor=(0.915,0.65),
                   ncol=1, fontsize=fontsize)
    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95)
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.99, top=0.99,
                    wspace=0.05,   # 横向间距 ← 调小
                    hspace=0.12)   # 纵向间距 ← 调小
    # plt.tight_layout()
    plt.savefig(f'./plot_source/keywords.jpg', dpi=300)
    plt.show()



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
                 color=colors[ii], lw=3, label=model_name[:-10])

    plt.tick_params(axis='x', labelsize=fontsize*0.8)  # x 轴刻度数字
    plt.tick_params(axis='y', labelsize=fontsize*0.8)  # y 轴刻度数字
    plt.xlabel('Token Index', fontsize=fontsize)
    plt.ylabel('KL Divergence', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.subplots_adjust(left=0.08, bottom=0.2, right=0.99, top=0.99)   # 纵向间距 ← 调小
    plt.savefig(f'./plot_source/kl_nc.jpg', dpi=300)
    plt.show()


def plot_finetune():
    path = './plot_source/finetune.csv'
    df = pd.read_csv(path)

    # 2. 学术风格全局设置
    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "figure.dpi": 300,
        "font.size": 16,
        "axes.linewidth": .6,
        "xtick.major.width": .6,
        "ytick.major.width": .6,
        "grid.linewidth": .4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02
    })

    # 3. 高级配色（Muted 版）
    methods = ['WordBkd', 'SentBkd', 'BadEdit', 'BadArch']
    colors = sns.color_palette("coolwarm", len(methods))  # 低饱和学术配色

    x = df['epoch'].values
    width = 0.18

    fig, ax = plt.subplots()
    for idx, method in enumerate(methods):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, df[method], width=width, color=colors[idx], label=method, zorder=3)
        ax.plot(x + offset, df[method], color=colors[idx], marker='o', ms=3, lw=2,
                zorder=10)

    ax.set_xticks(x)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    plt.legend(loc='lower center', bbox_to_anchor=(1.14,0.42),
                   ncol=1)
    # sns.despine(trim=True)  # 去掉右侧和顶部边框

    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.8, top=0.99)   # 纵向间距 ← 调小
    # 显示图表
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig(f'./plot_source/finetune.jpg', dpi=300)
    plt.show()


def plot_lm_eval():
    fontsize = 16
    # 1. 读 CSV
    df = pd.read_csv('./plot_source/lm_eval.csv', index_col=0)  # 第一列是方法名，作为行索引

    # 2. 提取数据
    methods = df.index.tolist()  # ['Clean', 'Wordbkd', 'Sentbkd', 'BadEdit', 'BadArch']
    metrics = df.columns.tolist()  # 8 个任务名
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 3. 画图
    plt.figure(figsize=(5, 4))
    ax = plt.subplot(111, polar=True)

    # 颜色循环
    palette = sns.color_palette("coolwarm", len(methods))

    for i, method in enumerate(methods):
        values = df.loc[method].values.tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=palette[i])
        ax.fill(angles, values, alpha=0.08, color=palette[i])

    # 再把每个标签沿径向往外推 25%（可调）
    radial_offset = -0.01  # 单位是百分比，越大越远
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        label.set_horizontalalignment('center')
        label.set_rotation(np.degrees(angle))
        # matplotlib 3.3+ 可用 transform + radial offset
        label.set_position((angle, radial_offset))

    # 4. 美化
    ax.grid(True)
    ax.set_theta_offset(np.pi / 2)
    # ax.set_theta_offset(0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=fontsize*0.8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.05, 0.2))
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=fontsize*0.7)

    plt.legend(loc='center right', bbox_to_anchor=(1.5,0.9),
                   ncol=1, fontsize=fontsize*0.8, handletextpad=0.4,
           handlelength=1.0,
           columnspacing=1)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.7, top=0.9)   # 纵向间距 ← 调小

    # 6. 把 θ 标签重新画到最顶层
    # 先拿到角度和标签文本
    angles_label = angles[:-1]
    labels_text = metrics

    # 手动再写一遍，zorder 设得比任何元素都高
    for angle, txt in zip(angles_label, labels_text):
        ax.text(x=angle,
                y=1.2,  # 半径位置，和原来平齐
                s=txt,
                fontsize=fontsize*0.8,
                ha='center',
                va='center',
                zorder=10)  # 10 足够高，保证压在最上面

    # 然后把原来的那一圈标签隐藏掉，避免重复
    ax.set_xticklabels([])

    # 5. 保存 & 显示
    plt.savefig(f'./plot_source/lm_eval.jpg', dpi=300)
    plt.show()


def plot_badarch(task='classification'):
    fontsize = 25
    df = pd.read_csv(f'./plot_source/{task}.csv')

    # 2. 高级相似配色
    base_hue = 0.58  # 青蓝色相
    clean_rgb = mcolors.hsv_to_rgb([base_hue, 0.35, 0.90])  # 淡一点
    cacc_rgb = mcolors.hsv_to_rgb([base_hue, 0.55, 0.65])  # 深一点
    asr_rgb = mcolors.hsv_to_rgb([0.05, 0.75, 0.85])  # 暖橙对比
    custom_pal = {'CACC-Clean': clean_rgb, 'CACC-BadArch': cacc_rgb, 'ASR-BadArch': asr_rgb}

    # 2. 把三列指标“融化”成长格式
    id_vars = ['dataset', 'model']
    value_vars = ['CACC-Clean', 'CACC-BadArch', 'ASR-BadArch']
    df_long = df.melt(id_vars=id_vars,
                      value_vars=value_vars,
                      var_name='metric',
                      value_name='score')

    # 3. 画图
    sns.set_style("whitegrid")

    g = sns.catplot(
        data=df_long,
        kind='bar',
        x='dataset',  # 大组：LLama / Deepseek / Mistral
        y='score',  # 柱子高度
        hue='metric',  # 颜色区分 Clean-CACC / CACC / ASR
        col='model',  # 每个 dataset 一列小图
        col_wrap=3,  # 3 列排成一行
        height=4,
        aspect=1.5,
        palette=custom_pal,
        legend_out=False
    )

    g.set_axis_labels("", "Score", fontsize=fontsize)
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.title.set_fontsize(fontsize)  # 这里才会真正生效
    g.set_xticklabels(fontsize=fontsize)   #
    # g.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fontsize*0.8)   #
    g.tick_params(axis='y', labelsize=fontsize*0.8)

    # ---------- 新增 ----------
    g._legend.remove()  # 去掉默认 legend
    g.fig.legend(
        handles=list(g._legend_data.values()),  # 真正用过的色块
        labels=list(g._legend_data.keys()),
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.),
        fontsize=fontsize,
        title='')  # 去掉“Metric”
    # --------------------------
    plt.subplots_adjust(left=0.05, bottom=0.32, right=0.98, top=0.91)
    plt.savefig(f'./plot_source/{task}.jpg', dpi=300)
    plt.show()


def plot_prune():
    from sklearn.preprocessing import MinMaxScaler
    fontsize = 24

    df = pd.read_csv('./plot_source/prune.csv', index_col=0)  # 第一列是方法名
    df = df*100
    df_delta = df.subtract(df.loc['Orig'], axis=1)
    df_delta = df_delta.drop('Orig')
    # 4. 绘制热力图
    plt.figure(figsize=(10, 4))
    # 使用 RdBu_r 调色板（红色表示下降，蓝色表示上升，白色表示无变化）
    # center=0 确保 0 值对应中性色
    # ax =sns.heatmap(df_delta, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    #             linewidths=.5, cbar_kws={'label': 'ΔASR'}, annot_kws={"size": fontsize*0.7})
    ax = sns.heatmap(df_delta,
                     annot=True,
                     cmap="coolwarm",  # 1. 颜色统一
                     fmt=".2f",
                     linewidths=1.2,  # 2. 增加分割线
                     annot_kws={"weight": "bold", "size": fontsize*0.7},  # 3. 数字加粗
                     cbar_kws={'label': 'ΔASR'},
                     center=0)

    plt.xlabel('', fontsize=fontsize)
    plt.ylabel('', fontsize=fontsize)
    # 修改 Y 轴刻度 (Ablate, Sparsegpt, Wanda, Magnitude)
    plt.yticks(fontsize=fontsize*0.8, rotation=0)

    # 修改 X 轴刻度 (Wordbkd, Sentbkd, BadEdit, BadArch)
    plt.xticks(fontsize=fontsize*0.8)

    cbar_axes = ax.collections[0].colorbar.ax
    cbar_axes.yaxis.label.set_size(fontsize*0.8)
    # 如果你还想修改颜色条上的刻度数字大小：
    cbar_axes.tick_params(labelsize=fontsize*0.8)

    plt.tight_layout()
    plt.savefig(f'./plot_source/prune.jpg', dpi=300)
    plt.show()

    # metrics = df.columns
    # fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True)
    # axes = axes.ravel()
    #
    # for ax, m in zip(axes, metrics):
    #     bars = ax.bar(df.index, df[m], color='steelblue')
    #     bars[0].set_color('lightgray')  # Orig 单独上色
    #     ax.axhline(df.loc['Orig', m], ls='--', color='gray')
    #     ax.set(ylim=(0, 1.05), title=m)
    #     ax.tick_params(axis='x', rotation=45)
    #
    # plt.suptitle('Small multiples: per-metric comparison')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    plot_kl()
    # plot_keywords()
    # plot_finetune()
    # plot_lm_eval()
    # plot_badarch('classification')
    # plot_badarch('jailbreak')
    # plot_prune()