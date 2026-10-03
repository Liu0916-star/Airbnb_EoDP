"""
COMP20008 A2 — 相关性分析（A 板块第 2 个文件，先跑 preprocess.py）
输入: data/clean.parquet（25,728 行；筛掉 4,477 个零评论房源，只用 21,251 行——它们的天数是人为填的 4,594，会把相关系数拉偏）
输出: outputs/correlation_matrix.csv（报告 Table 1：10 对变量 × 4 种方法）
      outputs/fig_hlc_vs_y.png（报告 Figure 1）、outputs/fig_dsl_vs_dist.png（报告 Figure 2）
      outputs/evidence_correlation.json（报告相关性部分引用的所有数字）
运行: python3 correlation.py（从哪个目录运行都可以）；每个 "# %%" 对应最终 notebook 的一个 cell
"""

# %% Section 0. 准备：导入工具库、读数据、筛出有评论的房源、定义 5 个变量和分箱工具
import json  # 保存 evidence_correlation.json
import os  # 拼接文件路径
from itertools import combinations  # 生成两两组合：5 个变量 → 10 对

import matplotlib.pyplot as plt  # 画图
import pandas as pd  # 表格；Pearson / Spearman 也直接用它的 .corr() 算
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score  # 互信息 MI（任意依赖，没有上限）、归一化互信息 NMI（缩放到 0~1，可跨变量对比较）

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else os.getcwd()   # 项目根目录（本文件在脚本文件夹里，往上一层）；转成 notebook 后没有 __file__，退回当前工作目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")                           # 输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)                                       # 没有就建，已有不报错
round_value = lambda value, digits=4: round(float(value), digits)            # 小工具：转成普通小数并保留 digits 位，才能存进 json

listings = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "clean.parquet"))  # 读预处理结果（25,728 行 × 98 列）
listings = listings[listings["has_review"] == 1]                             # 只留有评论的 21,251 行（零评论房源的天数是填充值，不是真实值）
CORRELATION_VARIABLES = ["host_listings_count",                              # HLC：房东房源数（运营：规模）
                         "days_since_last_review",                           # DSLR：距上次评论天数（运营：活跃度）
                         "availability_365",                                 # AV365：未来一年可订天数（运营）
                         "dist_cbd",                                         # DIST：到市中心距离（基本面，唯一的一个）
                         "y"]                                                # SH：是不是超级房东（目标，作业要求必须包含）
equal_frequency_bin = lambda column, n_bins=10: (                            # 分箱工具：MI/NMI 只能处理离散类别，连续数字要先切成"第几箱"
    column if column.nunique() <= 2                                          #   y 本来就只有 0/1，原样返回
    else pd.qcut(column, n_bins, labels=False, duplicates="drop"))           #   连续变量等频切 n_bins 箱（每箱房源数大致相同）；labels=False 返回箱号 0,1,2…；duplicates="drop" 合并重复边界（AV365 有 21.8% 是 0，同一个值没法拆进两个箱）
evidence = {"n_rows": len(listings)}                                         # 记账本：先记用了多少行 21,251

# %% Section 1. 分箱决策的证据：为什么等频、为什么 10 箱、为什么跨对比较只看 NMI
# ---- 1.1 分布描述（Methodology 引用） ----
evidence["percentiles"] = listings[["host_listings_count", "days_since_last_review"]].quantile([0.5, 0.75, 1]).to_dict()  # HLC 中位数 2 / 75% 分位 14 / 最大 667；DSLR 84 / 448 / 4,593 → 都是严重右偏
evidence["av365_share_0"] = round_value((listings["availability_365"] == 0).mean())      # 可订 0 天的占 21.76%（大量相同值 → 等频分箱时边界会重复、被合并）
evidence["av365_share_365"] = round_value((listings["availability_365"] == 365).mean())  # 可订 365 天的只占 1.56%

# ---- 1.2 为什么不用等宽分箱 ----
evidence["hlc_equal_width_max_bin"] = {                                      # 等宽分箱（把数值范围平均切开）时最挤的那个箱装了多少比例的房源
    n_bins: round_value(pd.cut(listings["host_listings_count"], n_bins).value_counts(normalize=True).max())
    for n_bins in (10, 20)}                                                  # 10 箱 91.4%、20 箱 84.6% → 绝大部分房源挤在第一个箱，MI 算不出区分度，所以用等频

# ---- 1.3 为什么跨对比较只看 NMI ----
dsl_column, availability_column = listings["days_since_last_review"], listings["availability_365"]  # 用相关最强的一对来测箱数敏感度
evidence["mi_bin_sensitivity"] = {                                           # 同一对变量分别切 5 / 10 / 20 箱
    n_bins: {"mi": round_value(mutual_info_score(equal_frequency_bin(dsl_column, n_bins), equal_frequency_bin(availability_column, n_bins))),
             "nmi": round_value(normalized_mutual_info_score(equal_frequency_bin(dsl_column, n_bins), equal_frequency_bin(availability_column, n_bins)))}
    for n_bins in (5, 10, 20)}                                               # 5→20 箱：MI 涨 113%，NMI 只涨 12% → MI 的大小主要取决于箱数，不能跨对比较

# %% Section 2. Table 1：10 对变量 × 4 种方法
table_rows = []                                                              # 每对变量一行
for variable_a, variable_b in combinations(CORRELATION_VARIABLES, 2):        # 5 选 2，共 10 对
    binned_a, binned_b = equal_frequency_bin(listings[variable_a]), equal_frequency_bin(listings[variable_b])  # 两边各分 10 箱，给 MI/NMI 用
    table_rows.append({
        "pair": f"{variable_a} ~ {variable_b}",                              # 这一对的名字
        "pearson": round_value(listings[variable_a].corr(listings[variable_b]), 3),                       # Pearson：线性关系（对极端值敏感）；对 0/1 的 y 就是点二列相关
        "spearman": round_value(listings[variable_a].corr(listings[variable_b], method="spearman"), 3),    # Spearman：先换成排名再算，测单调关系，极端值影响被压小
        "mi": round_value(mutual_info_score(binned_a, binned_b)),            # MI：任意形式的依赖（倒 U 型也能测到）
        "nmi": round_value(normalized_mutual_info_score(binned_a, binned_b)),  # NMI：缩放到 0~1，可跨对比较
        "bins": f"{binned_a.nunique()}×{binned_b.nunique()}"})               # 实际箱数（重复边界合并后可能少于 10），如 "7×10"
correlation_table = pd.DataFrame(table_rows).sort_values("nmi", ascending=False)  # 按 NMI 从大到小排，和报告 Table 1 顺序一致

# %% Section 3. 找分歧：Pearson 和 Spearman 意见不一致的变量对
pearson_spearman_gap = (correlation_table["spearman"] - correlation_table["pearson"]).abs()  # 每对两个系数差多少（取绝对值）
evidence["max_gap"] = {correlation_table.loc[pearson_spearman_gap.idxmax(), "pair"]: round_value(pearson_spearman_gap.max(), 3)}  # 差距最大的一对：HLC ~ AV365，0.166（方向相同）
evidence["sign_flip_pairs"] = correlation_table.loc[correlation_table["pearson"] * correlation_table["spearman"] < 0, "pair"].tolist()  # 一正一负（相乘 < 0）：HLC ~ y、DSLR ~ DIST → 下面两张图确认原因

# %% Section 4. Figure 1：按房东规模分档看超级房东率（倒 U 型）
# ---- 4.1 分档统计 ----
overall_superhost_rate = listings["y"].mean()                                # 整体超级房东率 0.357（有评论的 21,251 行）
host_size_bands = (listings.groupby(pd.cut(listings["host_listings_count"], [0, 1, 2, 5, 14, 50, 100, 700], include_lowest=True), observed=True)["y"]  # 按房源数切 7 档（2 = 中位数，14 = 75% 分位，700 盖住最大值 667）；include_lowest 让 0 也落进第一档；observed=True 只保留有数据的档
                   .agg(n="size", rate="mean"))                              # 每档：房源个数、超级房东率
host_size_bands.index = host_size_bands.index.astype(str)                    # 区间对象转成文字，才能存进 json
evidence["hlc_band"] = host_size_bands.round(4).to_dict("index")             # 0–1 套 28.48%（n=8,103）→ 3–5 套峰值 46.08% → 超过 100 套断崖 16.92%（n=1,259）

# ---- 4.2 画柱状图 ----
figure, axis = plt.subplots(figsize=(7.5, 4.5))                              # 一张图
axis.bar(range(len(host_size_bands)), host_size_bands["rate"], edgecolor="black", linewidth=0.6)  # 柱子 = 每档超级房东率
axis.axhline(overall_superhost_rate, color="red", linestyle="--", label=f"Overall {overall_superhost_rate:.3f}")  # 红色虚线 = 整体水平
for bar_position, band_rate in enumerate(host_size_bands["rate"]):
    axis.text(bar_position, band_rate + 0.012, f"{band_rate:.3f}", ha="center", fontsize=9)  # 柱顶标数值
axis.set_xticks(range(len(host_size_bands)))
axis.set_xticklabels([f"{band}\nn={count:,}" for band, count in zip(host_size_bands.index, host_size_bands["n"])], rotation=45, ha="right", fontsize=8)  # x 轴：档位 + 样本量（证明每档都上千，不是小样本噪音）
axis.set(ylim=(0, 0.55), xlabel="host_listings_count band", ylabel="Superhost rate",
         title=f"Superhost rate by host_listings_count band (n={len(listings):,})")
axis.legend(loc="upper right")
axis.spines[["top", "right"]].set_visible(False)                             # 去掉上、右边框，图更干净
plt.tight_layout()                                                           # 自动调整边距，标签不被裁掉
plt.savefig(os.path.join(OUTPUT_DIR, "fig_hlc_vs_y.png"), dpi=150)
plt.close(figure)                                                            # 关掉图，释放内存

# %% Section 5. Figure 2：按到市中心距离分档看评论新近度（先升后降）
# ---- 5.1 分档统计 ----
distance_trend = listings.groupby(pd.qcut(listings["dist_cbd"], 20, labels=False)).agg(  # 距离等频切 20 档，每档约 1,063 个房源
    dist_km=("dist_cbd", "median"),                                          # 每档的中位距离
    dsl=("days_since_last_review", "median"))                                # 每档的中位天数（用中位数：天数最大 4,593，平均数会被拉偏）
evidence["dist_band_trend"] = distance_trend.round(2).to_dict("index")       # 1.36 km→51 天，1.71 km→111 天（350 米内翻倍），约 5 km 峰值 122，50.6 km→47 天

# ---- 5.2 画折线图 ----
figure, axis = plt.subplots(figsize=(7.5, 4.5))
axis.plot(distance_trend["dist_km"], distance_trend["dsl"], marker="o", linewidth=1.5, markersize=5)  # x = 距离，y = 天数，每个点是一档
axis.set(xlabel="Distance from CBD (km, band median)", ylabel="Median days_since_last_review",
         title="Review recency vs distance from CBD (20 equal-frequency bands)")
axis.grid(alpha=0.3)                                                         # 淡网格线，方便读数
axis.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_dsl_vs_dist.png"), dpi=150)
plt.close(figure)

# %% Section 6. 保存
correlation_table.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"), index=False)  # Table 1（行号没意义，不存）
with open(os.path.join(OUTPUT_DIR, "evidence_correlation.json"), "w") as evidence_file:
    json.dump(evidence, evidence_file, indent=2, ensure_ascii=False, default=str)  # default=str 处理区间等特殊类型
print(correlation_table.to_string(index=False))                              # 打印 Table 1 核对
