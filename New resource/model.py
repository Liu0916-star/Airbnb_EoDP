"""
COMP20008 A2 — 监督学习 + 特征选择
输入: data/clean.parquet（先跑 preprocess.py；25,728 行，全部使用，不删零评论房源）
输出: outputs/evidence_model.json（报告 B 部分引用的所有数字）
      outputs/cv_results.csv（调参表）、rq_experiments.csv（RQ 实验表）
      outputs/feature_ranking.csv（特征重要性 + 排名）、fs_validation.csv（前 3 特征验证表）
      outputs/fig_confusion.png（混淆矩阵）、fig_importance.png（置换重要性）
运行: python3 model.py（从哪里运行都可以；KNN 较慢，全程几分钟）；每个 "# %%" 对应最终 notebook 的一个 cell
"""

# %% 0. 准备数据：X、y、groups
import json                                                     # 保存 evidence
import os                                                       # 拼接路径
import matplotlib.pyplot as plt                                 # 画图
import numpy as np                                              # 数组、随机数、分位数
import pandas as pd                                             # 表格
from matplotlib.patches import Patch                            # 图例色块
from sklearn.compose import ColumnTransformer                   # 不同的列走不同的预处理
from sklearn.dummy import DummyClassifier                       # 基线：永远猜人数最多的类
from sklearn.feature_selection import mutual_info_classif       # Filter 特征选择：互信息
from sklearn.impute import SimpleImputer                        # 填空值
from sklearn.inspection import permutation_importance           # 置换重要性
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score  # 混淆矩阵图、每类指标、macro-F1
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score  # 分层 + 分组切分、交叉验证
from sklearn.neighbors import KNeighborsClassifier              # KNN
from sklearn.pipeline import Pipeline                           # 把预处理和模型串成一个整体
from sklearn.preprocessing import OneHotEncoder, StandardScaler # 独热编码、标准化
from sklearn.tree import DecisionTreeClassifier                 # 决策树

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录（本文件在"新代码"里，往上一层）
OUT = os.path.join(ROOT, "outputs")                             # 输出文件夹
os.makedirs(OUT, exist_ok=True)                                 # 没有就建
r = lambda x, d=4: round(float(x), d)                           # 小工具：转成普通小数并保留 d 位，才能存进 json
ev = {}                                                         # 记账本：报告要引用的数字都存这里

df = pd.read_parquet(os.path.join(ROOT, "data", "clean.parquet"))  # 读预处理结果
df["room_type_3"] = df["room_type"].replace({"Shared room": "Other", "Hotel room": "Other"})  # 合租和酒店合计仅 283 行，并成 Other → 三类
fundamentals = ["room_type_3", "dist_cbd", "accommodates"]      # 房源基本面：房东基本改不了的条件
operationals = ["availability_365", "amenity_count", "days_since_last_review", "has_review",
                "price_num", "price_missing", "minimum_nights", "host_listings_count"]  # 运营策略：房东怎么经营
ALL = fundamentals + operationals                               # 全部 11 个特征（白名单：官方评选标准和评论衍生列一律不进）
X, y, groups = df[ALL], df["y"], df["host_id"]                  # 特征 / 答案 / 房东编号（只用来切分，不进模型）
# 价格用 price_num（带空值），不用 price_filled：后者用全体数据的中位数补的，含测试集信息；这里改在 Pipeline 里只用训练集补

# %% 1. 按房东分组切分：80% 训练、20% 测试
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)  # 分 5 份 → 1 份 = 20%；分层（正类率接近）+ 分组（同一房东不跨两边）
train_idx, test_idx = next(sgkf.split(X, y, groups=groups))     # 只取第一种切法：1 份测试、4 份训练（返回的是行号）
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]           # 按行号取特征
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]           # 按行号取答案
g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx] # 按行号取房东：训练集的给交叉验证用，测试集的给 bootstrap 用
ev["split"] = {"train_rows": len(X_train), "test_rows": len(X_test),      # 20,783 / 4,945
               "test_share": r(len(X_test) / len(X)),                     # 0.1922（房东整体移动，不会刚好 0.20）
               "train_hosts": int(g_train.nunique()), "test_hosts": int(g_test.nunique()),  # 11,293 / 2,820
               "train_pos_rate": r(y_train.mean()), "test_pos_rate": r(y_test.mean()),     # 0.3119 / 0.2835（大房东落在哪边会拉偏）
               "host_overlap": len(set(g_train) & set(g_test))}           # 两边房东交集，必须是 0
assert ev["split"]["host_overlap"] == 0                         # 分组切分生效，否则直接停
print("split:", ev["split"])

# %% 2. 预处理流水线 + 两个小工具
def get_preprocessor(feats, scale=False):
    """给定特征名单，造一个预处理器：数字列补中位数（KNN 还要标准化），房型独热编码。只造不 fit。"""
    num = [c for c in feats if c != "room_type_3"]              # 数字列 = 除房型外的全部
    steps = [("impute", SimpleImputer(strategy="median"))]      # 数字路第 1 步：补中位数（fit 时只用训练数据算）
    if scale:
        steps.append(("scale", StandardScaler()))               # 第 2 步（仅 KNN）：减均值除标准差，让每个特征发言权相当
    parts = [("num", Pipeline(steps), num)] if num else []      # 分配器的第一条路
    if "room_type_3" in feats:                                  # 名单里有房型才加文字路（只用运营特征时没有）
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["room_type_3"]))  # 拆成 3 列 0/1
    return ColumnTransformer(parts)                             # 没列进来的列会被丢掉，所以名单必须写全

def make_model(name, feats, param):
    """造一个完整模型 = 预处理器 + 分类器。KNN 的 param 是邻居数，DT 的 param 是最大深度。"""
    if name == "KNN":
        return Pipeline([("prep", get_preprocessor(feats, scale=True)), ("model", KNeighborsClassifier(n_neighbors=param))])  # KNN 算距离，必须标准化
    return Pipeline([("prep", get_preprocessor(feats)), ("model", DecisionTreeClassifier(max_depth=param, random_state=42))])  # 树只问大于小于，不标准化，切点保留真实单位

def cv(model, feats):
    """在训练集上做 5 折交叉验证（同样按房东分组），返回 5 个 macro-F1。"""
    return cross_val_score(model, X_train[feats], y_train, groups=g_train, cv=sgkf, scoring="f1_macro", n_jobs=-1)  # 漏传 groups 不报错但会泄漏

# %% 3. 基线 + 交叉验证调参
rows = []                                                       # 每个候选值一行
s = cv(DummyClassifier(strategy="most_frequent"), ALL)          # 基线：全猜 0，macro-F1 约 0.41
rows.append({"model": "Baseline", "param": "-", "cv_mean": s.mean(), "cv_std": s.std()})
GRID = {"KNN": [1, 5, 11, 21, 51, 101, 151, 201, 301],        # 邻居数：太小过拟合、太大欠拟合；奇数避免投票打平
        "DT": [2, 3, 4, 5, 6, 8, 10, 15, None]}                 # 深度：None = 不限（等于背答案）
for name, grid in GRID.items():
    for p in grid:
        s = cv(make_model(name, ALL, p), ALL)                   # 这个候选值的 5 折成绩
        rows.append({"model": name, "param": p, "cv_mean": s.mean(), "cv_std": s.std()})
cv_df = pd.DataFrame(rows)                                      # 19 行调参表（报告要列出每个值的成绩）
best = {n: max((row for row in rows if row["model"] == n), key=lambda row: row["cv_mean"])["param"] for n in GRID}  # 各自 CV 平均分最高的超参数（直接从列表取，避免 pandas 把整数变小数）
BEST = max(GRID, key=lambda n: cv_df.loc[cv_df["model"] == n, "cv_mean"].max())  # 按 CV（不看测试集）选出更好的模型，第 5、8 段用它
ev["best_param"], ev["best_model"] = best, BEST
print(cv_df.round(4).to_string(index=False), "\nbest:", best, "→", BEST)

# %% 4. 测试集最终评估（只考一次）
final = {"Baseline": DummyClassifier(strategy="most_frequent"),  # 三个正式模型：用最佳超参数在整个训练集上训练
         "KNN": make_model("KNN", ALL, best["KNN"]),
         "DT": make_model("DT", ALL, best["DT"])}
preds, ev["test"] = {}, {}                                      # 每个模型的测试集预测、成绩
for name, m in final.items():
    preds[name] = m.fit(X_train, y_train).predict(X_test)       # .fit = 机器学习；.predict = 考试
    rep = classification_report(y_test, preds[name], digits=3, zero_division=0, output_dict=True)  # 每类 P/R/F1；zero_division 给基线用（它从不猜 1）
    ev["test"][name] = {"macro_f1": r(rep["macro avg"]["f1-score"]), "accuracy": r(rep["accuracy"]),  # 主指标 macro-F1，准确率仅作参考
                        **{f"class{c}": {k: r(rep[c][k]) for k in ("precision", "recall", "f1-score")} for c in ("0", "1")}}
    print(f"\n==== {name} ====\n" + classification_report(y_test, preds[name], digits=3, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))                 # 混淆矩阵：只画 KNN 和 DT（基线一整列是 0）
for ax, name in zip(axes, ["KNN", "DT"]):
    ConfusionMatrixDisplay.from_predictions(y_test, preds[name], ax=ax, colorbar=False,
                                            display_labels=["Non-superhost", "Superhost"])  # 行 = 真实，列 = 预测
    ax.set_title(f"{name} (test macro-F1 = {ev['test'][name]['macro_f1']:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_confusion.png"), dpi=150)
plt.close(fig)

y_te = y_test.to_numpy()                                        # 转成数组才能按位置取
host_rows = list(g_test.reset_index(drop=True).groupby(g_test.to_numpy()).indices.values())  # 每个测试房东在测试集里的行号（共 2,820 组）
rng = np.random.default_rng(42)                                 # 固定种子，可复现

def boot(pa, pb, n=1000):
    """按房东有放回抽样 n 次，返回模型 a、b 的 macro-F1 95% 区间和 b − a 的区间（配对：每次同一份样本给两边打分）。"""
    sa, sb = [], []
    for _ in range(n):
        pick = rng.integers(len(host_rows), size=len(host_rows))  # 有放回抽 2,820 个房东（有的抽到两次，有的没抽到）
        idx = np.concatenate([host_rows[i] for i in pick])      # 拼出这些房东的全部房源
        sa.append(f1_score(y_te[idx], pa[idx], average="macro"))  # 同一份样本上给 a 打分
        sb.append(f1_score(y_te[idx], pb[idx], average="macro"))  # 同一份样本上给 b 打分
    ci = lambda v: [r(q) for q in np.percentile(v, [2.5, 97.5])]  # 去掉两头各 2.5% → 95% 区间
    return {"a": ci(sa), "b": ci(sb), "b_minus_a": ci(np.array(sb) - np.array(sa))}  # 差值区间不跨 0 → 差距不是运气

ev["boot_KNN_vs_DT"] = boot(preds["KNN"], preds["DT"])          # a = KNN，b = DT
print("\nbootstrap KNN vs DT:", ev["boot_KNN_vs_DT"])

# %% 5. RQ 实验：只给模型看一部分特征
REV = ["days_since_last_review", "has_review"]                  # 评论新近度那一对（一起去掉：has_review 就是在说明 dsl 是不是填的）
SETS = {"A_fundamentals": fundamentals,                         # 3 个：只看基本面
        "B_operational": operationals,                          # 8 个：只看运营
        "C_all": ALL,                                           # 11 个：对照组（= 第 4 段）
        "D_all_no_review": [c for c in ALL if c not in REV],    # 9 个：结果是不是全靠评论新近度？
        "E_all_no_avail": [c for c in ALL if c != "availability_365"],  # 10 个：填报告的 [TO BE CONFIRMED]
        "F_operational_no_review": [c for c in operationals if c not in REV]}  # 6 个：去掉评论后运营还比基本面强吗？

def run_sets(sets):
    """每组特征 × 两个模型：CV 成绩（主要依据）+ 测试成绩（确认）。超参数沿用第 3 段。"""
    out, p = [], {}
    for exp, feats in sets.items():
        for name in ("KNN", "DT"):
            m = make_model(name, feats, best[name])             # 只用这组特征的模型
            s = cv(m, feats)                                    # 训练集 5 折
            p[exp, name] = m.fit(X_train[feats], y_train).predict(X_test[feats])  # 只取这组列训练、预测
            out.append({"experiment": exp, "model": name, "n_features": len(feats), "cv_mean": s.mean(),
                        "cv_std": s.std(), "test_macro_f1": f1_score(y_test, p[exp, name], average="macro")})
    return pd.DataFrame(out), p

rq_df, rq_preds = run_sets(SETS)                                # 12 行
assert all((rq_preds["C_all", n] == preds[n]).all() for n in ("KNN", "DT"))  # 自检：C 组必须和第 4 段完全一样
ev["rq"] = rq_df.round(4).to_dict("records")
ev["boot_A_vs_B"] = boot(rq_preds["A_fundamentals", BEST], rq_preds["B_operational", BEST])  # RQ 关键：运营 − 基本面
ev["boot_A_vs_F"] = boot(rq_preds["A_fundamentals", BEST], rq_preds["F_operational_no_review", BEST])  # 稳健性：去掉评论后的运营 − 基本面
print("\n", rq_df.pivot(index="experiment", columns="model", values="cv_mean").round(4))
print("bootstrap A vs B:", ev["boot_A_vs_B"], "\nbootstrap A vs F:", ev["boot_A_vs_F"])

# %% 6. 特征影响力：决策树自带重要性 + 两个模型的置换重要性
dt = final["DT"]                                                # 第 4 段训练好的正式决策树
names = dt.named_steps["prep"].get_feature_names_out()          # 预处理后的 13 个列名：num__xxx、cat__room_type_3_xxx
dt_builtin = pd.Series(dt.named_steps["model"].feature_importances_,  # 13 个重要性（训练集上的纯度提升，偏爱连续特征）
                       index=["room_type_3" if n.startswith("cat__") else n[5:] for n in names]  # 去掉 "num__" 前缀；3 个房型列同名
                       ).groupby(level=0).sum()                 # 同名相加 → 11 个，总和 = 1
perm = {n: pd.Series(permutation_importance(final[n], X_test, y_test, scoring="f1_macro", n_repeats=10,
                                            random_state=42, n_jobs=-1).importances_mean, index=ALL)
        for n in ("KNN", "DT")}                                 # 测试集上每列打乱 10 次，macro-F1 平均掉多少；打乱的是原始 11 列
rank_df = pd.DataFrame({"group": ["fundamental" if c in fundamentals else "operational" for c in ALL],
                        "DT_builtin": dt_builtin, "DT_perm": perm["DT"], "KNN_perm": perm["KNN"]}, index=ALL)  # 按特征名自动对齐

COLORS = {"fundamental": "#4C72B0", "operational": "#DD8452"}   # 两组两种颜色
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ["KNN_perm", "DT_perm"]):
    s = rank_df.sort_values(col)                                # 升序 → 最重要的画在最上面
    ax.barh(s.index, s[col], color=s["group"].map(COLORS))      # 横向条形图，特征名在 y 轴
    ax.axvline(0, color="grey", linewidth=0.8)                  # 0 线：小负数 = 没用
    ax.set(title=f"{col[:-5]} permutation importance (test set)", xlabel="Mean drop in macro-F1")
axes[1].legend(handles=[Patch(color=v, label=k) for k, v in COLORS.items()], loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_importance.png"), dpi=150)
plt.close(fig)

# %% 7. 特征选择：Filter（MI）vs Embedded（决策树重要性），各选前 3
Xm = X_train.copy()                                             # 训练集副本（只用训练集选特征，避免偷看）
Xm["room_type_3"] = Xm["room_type_3"].astype("category").cat.codes  # 房型 → 0/1/2（MI 只看是否同类，不在乎大小）
Xm = Xm.fillna(Xm.median())                                     # MI 不接受空值：用训练集中位数补
rank_df["filter_MI"] = mutual_info_classif(Xm, y_train, random_state=42,  # 每个特征单独和 y 算 MI（内部有随机成分）
                                           discrete_features=[c in ("room_type_3", "has_review", "price_missing") for c in ALL])
rank_df["filter_rank"] = rank_df["filter_MI"].rank(ascending=False, method="min").astype(int)      # Filter 排名
rank_df["embedded_rank"] = rank_df["DT_builtin"].rank(ascending=False, method="min").astype(int)   # Embedded 排名
top3 = {"filter_top3": rank_df["filter_MI"].nlargest(3).index.tolist(),       # Filter：单独看每个特征，重复信息会一起入选
        "embedded_top3": rank_df["DT_builtin"].nlargest(3).index.tolist()}    # Embedded：看在树里的额外贡献，重复的会被压低
fs_df, _ = run_sets(top3)                                       # 只用前 3 个特征重新打分
fs_df = pd.concat([fs_df, rq_df[rq_df["experiment"] == "C_all"]], ignore_index=True)  # 附上全部特征作对照
ev["top3"] = top3
ev["top3_common"] = sorted(set(top3["filter_top3"]) & set(top3["embedded_top3"]))  # 两种方法都选中的
ev["feature_ranking"] = rank_df.round(4).to_dict("index")
ev["fs_validation"] = fs_df.round(4).to_dict("records")
print("\n", rank_df.sort_values("filter_rank").round(4), "\n", top3, "\n", fs_df.round(4).to_string(index=False))

# %% 8. 难例分析：模型最有把握判错的超级房东（FN）
detail = X_test.assign(y_true=y_test, y_pred=preds[BEST],       # 测试集明细：真实、预测
                       proba=final[BEST].predict_proba(X_test)[:, 1],  # 模型认为是超级房东的概率
                       id=df.loc[X_test.index, "id"], host_id=g_test)  # 房源 ID、房东 ID（X_test 保留了原 df 的行标签）
fn = detail[(detail["y_true"] == 1) & (detail["y_pred"] == 0)].sort_values("proba", kind="stable")  # 所有漏掉的超级房东，概率从小到大
case = fn.iloc[0]                                               # 第一行 = 最有把握判错的
ref = X_train.groupby(y_train).median(numeric_only=True).T      # 训练集两类的中位数（房型是文字，自动跳过）
ref.columns = ["non_superhost_median", "superhost_median"]
ref["this_listing"] = case[ref.index].astype(float)             # 这个房源的实际数值，放在一起比
host = df[df["host_id"] == case["host_id"]]                     # 这个房东在数据集里的全部房源（查背景，不当特征）
ev["hard_case"] = {"listing_id": int(case["id"]), "host_id": int(case["host_id"]), "model": BEST,
                   "proba_superhost": r(case["proba"]), "room_type": case["room_type_3"], "fn_total": len(fn),
                   "listing_number_of_reviews": int(df.loc[case.name, "number_of_reviews"]),  # 被排除的列，只用来解释
                   "host_n_listings": len(host), "host_listings_with_reviews": int(host["has_review"].sum()),
                   "comparison": ref.round(2).to_dict("index")}
print("\n", {k: v for k, v in ev["hard_case"].items() if k != "comparison"}, "\n", ref.round(2))

# %% 9. 保存
cv_df.round(4).to_csv(os.path.join(OUT, "cv_results.csv"), index=False)         # 调参表 17 行
rq_df.round(4).to_csv(os.path.join(OUT, "rq_experiments.csv"), index=False)     # RQ 实验表 12 行
rank_df.round(4).to_csv(os.path.join(OUT, "feature_ranking.csv"))               # 特征表 11 行（index = 特征名，要保留）
fs_df.round(4).to_csv(os.path.join(OUT, "fs_validation.csv"), index=False)      # 前 3 验证表 6 行
with open(os.path.join(OUT, "evidence_model.json"), "w") as f:
    json.dump(ev, f, indent=2, ensure_ascii=False, default=str)                 # default=str 兜底 numpy 类型
print("\n→ saved: evidence_model.json, 4 csv, fig_confusion.png, fig_importance.png")
