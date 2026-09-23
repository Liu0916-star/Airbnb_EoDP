"""
COMP20008 A2 — 预处理管线 (v2: 原始 Inside Airbnb 数据集)
=========================================================
把原始 listings.csv 变成下游全部分析共用的 clean.parquet,
同时把报告要引用的 before/after 数字记进 evidence.json。

运行:  python preprocess.py
输入:  data/listings.csv          Inside Airbnb 原始文件 (25,728 行 × 90 列)
产出:  data/clean.parquet         下游所有板块都读这一个文件
       outputs/evidence.json      报告 Methodology + Impact of Preprocessing 引用

--------------------------------------------------------------------
v2 相对 v1 的变化 (数据源换成原始数据集之后)
--------------------------------------------------------------------
1. 日期是 ISO 格式 (2026-05-04), 默认解析即正确。
   ⚠ v1 的 dayfirst=True 在新数据上反而是 bug: 会把 16,570 行弄成 NaT。
   → 删除。v1 里"日期格式陷阱"那段论证整段作废。
2. 原始数据里有 4,477 条零评论房源 (A1 数据集把它们删掉了)。
   它们的 last_review 为空 → days_since_last_review 无定义。
   → 新增 has_review 指示变量 + dsl 填充 (见 step1)。
3. 步骤 1 的"受评内容"从"修日期格式"换成"零评论房源的缺失策略"。

--------------------------------------------------------------------
为什么要有这个文件
--------------------------------------------------------------------
rubric 有 1 分是 Cross-Section Consistency。保证方式是"清洗只有一份、
只跑一次、结果落盘" —— 下游全部从 clean.parquet 读。

--------------------------------------------------------------------
这个文件绝对不做的三件事
--------------------------------------------------------------------
✗ 不做标准化       → StandardScaler 必须在建模时只 fit 训练集
✗ 不做 train/test 切分 → 建模那一步做, 且必须按 host_id 分组
✗ 不删任何行       → 零评论房源和 price 缺失房源的超级房东率都显著
                     偏低, 删了会系统性推高正类率 (这正是 A1 数据集的问题)
"""

import json
import os

import numpy as np
import pandas as pd

RAW_PATH = "data/listings.csv"
OUT_PARQUET = "data/clean.parquet"
OUT_EVIDENCE = "outputs/evidence.json"

# 距离原点: Flinders Street Station。这是"我们的选择"不是客观事实 ——
# 选它是因为它是墨尔本主要交通枢纽。报告里要为这个选择辩护。
CBD_LAT, CBD_LON = -37.8183, 144.9671

# 1 度纬度 ≈ 111 km (处处成立); 1 度经度要乘 cos(纬度), 见 derive_dist_cbd
KM_PER_DEG = 111.0

# 全项目唯一种子。固定种子保证可复现, 但可复现 ≠ 稳健。
RANDOM_SEED = 42


# ====================================================================
# 第 0 层 — 解析 (复用 A1 task1, 不算受评的"3 个预处理步骤")
# ====================================================================

def parse_price(price):
    """'$1,306.99' -> 1306.99, 缺失 -> np.nan。取自 A1 parse_price_correct。
    关键: 删掉逗号和 $ 后整串转 float, 不要 re.search 截片段 (会把
    1,306.99 截成 306.99)。"""
    if pd.isna(price):
        return np.nan
    return float(price.replace(",", "").replace("$", ""))


def count_amenities(amenities):
    """'["Wifi", "HDTV, DVD player"]' -> 2。取自 A1 count_amenities_correct。
    关键: json.loads 解析, 不要按逗号 split (设施名本身含逗号)。
    缺失返回 0 —— 原始数据里 amenities 无缺失, 所以 0 都是真实的空列表。"""
    if pd.isna(amenities):
        return 0
    return len(json.loads(amenities.strip()))


# ====================================================================
# 第 1 层 — 三个受评的预处理步骤
# ====================================================================

def step1_review_recency(df, ev):
    """
    步骤 1: 派生 days_since_last_review + 零评论房源的缺失策略

    dsl = 参考日 - last_review (天)。测的是"这个房源最近有没有在运营" ——
    评论只在入住后产生, 最后一条评论的时间是近期成交的代理变量。
    RQ 二分法里属于【运营策略】侧。

    参考日 = last_review 的最大值 (2026-06-28), 不用 today:
      数据是快照, 最大值≈抓取日; 用 today 会引入与数据无关的偏移。

    ⚠ 零评论房源 (4,477 行, 17.4%) 的 dsl 无定义, 模型又吃不了 NaN。
      三个选项:
        删行        → 重蹈 A1 覆辙: 这批超级房东率仅 6.59%, 删了推高正类率
        填中位数    → 等于告诉模型"这些房源最近很活跃", 方向完全反了
        填 max+1 + has_review 指示变量  ← 采用
          语义: "比最久没评论的还久"; has_review 把"从未有评论"这个
          信息显式保留, 跟 price_missing 是同一套"缺失本身是信号"方法论。

    ⚠ RQ 张力 (报告必须正面处理):
      has_review ≡ number_of_reviews > 0, 是被剔除的官方指标的粗化形式。
      辩护: 官方门槛是"近一年约 10 次入住 + 约 4.8 分", "有没有过任何
      评论"粗得多; 且零评论房源里仍有 295 个超级房东 (房东其他房源达标),
      所以它不是标签的替身。
      但 dsl 本身与近 12 月评论数 Spearman −0.816 —— 是评论量的强代理。
      这一条进 Limitations, 并在 B 板块做"有/无 dsl+has_review"对照。

    ⚠ 填充后 dsl 有 4,477 行堆在 max+1。所以:
      - 两组均值只在 has_review == 1 的行上算, 否则被填充值扭曲
      - correlation.py 只在 has_review == 1 的行上算相关性
    """
    # 新数据是 ISO 格式, 默认解析即正确。NaT 数 = 零评论房源数。
    # errors="coerce": 解析失败返回 NaT 而不是抛异常。
    lr = pd.to_datetime(df["last_review"], errors="coerce")

    # has_review 必须在填充之前造 —— 填完就分不出哪些原本是缺失的
    df["has_review"] = lr.notna().astype(int)

    ref = lr.max()                          # .max() 自动跳过 NaT
    ev["ref_date"] = str(ref.date())

    # (ref - lr) 是 Timedelta 序列, 必须走 .dt 访问器取天数
    df["days_since_last_review"] = (ref - lr).dt.days

    # ---- 记账: 填充前, 只在有评论的行上 ----
    rev = df["has_review"] == 1
    ev["dsl_median"] = float(df.loc[rev, "days_since_last_review"].median())
    ev["dsl_max"] = int(df.loc[rev, "days_since_last_review"].max())
    ev["dsl_mean_super"] = float(round(
        df.loc[rev & (df["y"] == 1), "days_since_last_review"].mean(), 2))
    ev["dsl_mean_nonsuper"] = float(round(
        df.loc[rev & (df["y"] == 0), "days_since_last_review"].mean(), 2))

    # ---- 记账: 零评论房源 —— 这一步的"受评证据" ----
    ev["no_review_n"] = int((~rev).sum())
    ev["no_review_pct"] = float(round((~rev).mean() * 100, 2))
    ev["superhost_rate_no_review"] = float(round(df.loc[~rev, "y"].mean(), 4))
    ev["superhost_rate_has_review"] = float(round(df.loc[rev, "y"].mean(), 4))
    # 零评论房源里的超级房东数 —— has_review 不是标签替身的证据
    ev["superhost_n_in_no_review"] = int(df.loc[~rev, "y"].sum())

    # 零评论房源与其他缺失的重叠 (说明它们多是新上线/下架的房源)
    ev["no_review_av0_share"] = float(round(
        (df.loc[~rev, "availability_365"] == 0).mean(), 4))
    ev["no_review_price_missing_share"] = float(round(
        df.loc[~rev, "price_num"].isna().mean(), 4))

    # dsl 与近 12 月评论数的相关 —— Limitations 里"部分代理"的量化证据
    ev["dsl_vs_reviews_ltm_spearman"] = float(round(
        df.loc[rev, "days_since_last_review"].corr(
            df.loc[rev, "number_of_reviews_ltm"], method="spearman"), 4))

    # ---- 填充 ----
    fill_value = int(ev["dsl_max"] + 1)
    ev["dsl_fill_value"] = fill_value
    df["days_since_last_review"] = df["days_since_last_review"].fillna(fill_value)

    return df


def step2_amenity_count(df, ev):
    """
    步骤 2: amenity_count (第 0 层已算好, 这里只记账)

    RQ 二分法归入【运营策略】—— 房东可随时增减设施, 区位户型改不了。
    这个判定理由要写进报告, 是二分法唯一的模糊点。

    朴素实现 (按逗号 split) vs 正确实现 (json.loads) 的对照也在这里算:
    那是这一步真正的 before/after 证据。
    """
    ev["amenity_count_median"] = float(df["amenity_count"].median())
    ev["amenity_count_max"] = int(df["amenity_count"].max())
    ev["amenity_count_mean_super"] = float(round(
        df.loc[df["y"] == 1, "amenity_count"].mean(), 2))
    ev["amenity_count_mean_nonsuper"] = float(round(
        df.loc[df["y"] == 0, "amenity_count"].mean(), 2))
    ev["amen_zero_n"] = int((df["amenity_count"] == 0).sum())

    # before/after: A1 的朴素实现 (去掉首尾方括号后按逗号 split)
    def naive(a):
        if pd.isna(a):
            return 0
        t = a.strip()
        if t.startswith("[") and t.endswith("]"):
            t = t[1:-1]
        return 0 if t == "" else len(t.split(","))

    naive_cnt = df["amenities"].apply(naive)
    diff = naive_cnt != df["amenity_count"]
    ev["amen_naive_diff_n"] = int(diff.sum())
    ev["amen_naive_diff_pct"] = float(round(diff.mean() * 100, 2))
    ev["amen_naive_max_diff"] = int((naive_cnt - df["amenity_count"]).abs().max())

    return df


def step3_price_missing(df, ev):
    """
    步骤 3: price 缺失处理 (中位数填补 + 缺失指示变量)

    缺失是【结构性】的: availability_365 == 0 (下架) 的房源不显示价格。

    ⚠ 三个条件概率是三个不同的量, 别混:
        P(缺失 | av=0)   下架的几乎都没价格
        P(缺失 | av>0)   在售的几乎都有价格   ← 上面那个的对照组
        P(av=0 | 缺失)   没价格的多半是下架的

    不能删行: av=0 组超级房东率远低于 av>0 组, 删了系统性剔除一类房源。

    泄漏辩护: 填补值严格应只用训练集算。10 次 75% 子样本的中位数波动
    量化了"全量 vs 训练集"的差异。进 Limitations 预处理栏。

    与 bedrooms 的对比: 同为结构性缺失, 应对相反 ——
      price 填补+指示 (缺失携带"下架"信号, 且特征集必需);
      bedrooms 不用 (缺失随 room_type 剧变, 是语义性缺失, 可被
      accommodates 替代)。
    """
    miss = df["price_num"].isna()
    av0 = df["availability_365"] == 0

    ev["price_missing_n"] = int(miss.sum())
    # 单位约定: _pct 一律 0-100; p_ 和 _rate 一律 0-1
    ev["price_missing_pct"] = float(round(miss.mean() * 100, 2))
    ev["av0_n"] = int(av0.sum())
    # 布尔序列的布尔索引 + .mean() = 条件概率
    ev["p_miss_given_av0"] = float(round(miss[av0].mean(), 4))
    ev["p_miss_given_avpos"] = float(round(miss[~av0].mean(), 4))
    ev["p_av0_given_miss"] = float(round(av0[miss].mean(), 4))   # ⚠ 方向相反

    ev["superhost_rate_av0"] = float(round(df.loc[av0, "y"].mean(), 4))
    ev["superhost_rate_avpos"] = float(round(df.loc[~av0, "y"].mean(), 4))

    price_median_full = float(df["price_num"].median())
    ev["price_median_full"] = price_median_full

    # 中位数稳定性: 10 次 75% 随机子样本。rng.choice 返回位置, 用 .iloc
    rng = np.random.default_rng(RANDOM_SEED)
    sub_medians = []
    for _ in range(10):
        idx = rng.choice(len(df), size=int(len(df) * 0.75), replace=False)
        sub_medians.append(df.iloc[idx]["price_num"].median())
    ev["price_median_sub_min"] = float(min(sub_medians))
    ev["price_median_sub_max"] = float(max(sub_medians))

    # price_num 保留原始 NaN —— check_structural 靠它交叉验证
    df["price_missing"] = miss.astype(int)
    df["price_filled"] = df["price_num"].fillna(price_median_full)

    return df


# ====================================================================
# 第 2 层 — 建模需要的派生列 (不占"3 个受评步骤"名额)
# ====================================================================

def derive_dist_cbd(df, ev):
    """
    到 CBD 的平面近似距离 (km)。RQ 二分法【硬实力】侧最纯粹的代表 ——
    房东完全无法改变 (price 其实是定价决策, 更接近运营策略)。

    ⚠ 纬度修正: 墨尔本 cos(37.8°) ≈ 0.79, 不修正则【经度方向】高估约 26%;
      但合成距离的高估远小于此 —— 见 dist_inflation_pct。两个数别混。
      np.cos 吃弧度, 必须 np.radians(), 漏了不报错但结果全错。

    这是等距圆柱近似, 房源纬度跨度 <1 度, 误差远 <1%。进 Limitations。
    """
    dlat = (df["latitude"] - CBD_LAT) * KM_PER_DEG
    dlon = (df["longitude"] - CBD_LON) * KM_PER_DEG * np.cos(np.radians(CBD_LAT))
    df["dist_cbd"] = np.sqrt(dlat**2 + dlon**2)

    # 自检: 不做修正的版本, 只为得到一个数字
    dlon_naive = (df["longitude"] - CBD_LON) * KM_PER_DEG
    dist_naive = np.sqrt(dlat**2 + dlon_naive**2)

    ev["dist_median"] = float(round(df["dist_cbd"].median(), 2))
    ev["dist_max"] = float(round(df["dist_cbd"].max(), 2))
    ev["dist_inflation_pct"] = float(round(
        (dist_naive.median() / df["dist_cbd"].median() - 1) * 100, 2))

    return df


def derive_misc(df, ev):
    """
    (a) room_type_entire —— 二元编码。Hotel room 样本极少, 四类 one-hot
        会造出近乎全零的列。理由 (带 room_type_counts 里的数字) 进报告。

    (b) host_n_listings_actual —— 本数据集内该房东的真实房源数。
        host_listings_count 是房东【自报】的跨平台总数, 不等于本数据集内
        的持有量。作为排序特征可靠 (看 Spearman), 作为计数不准。
        解释时不能说"持有 100 套房的房东"。进 Limitations。
    """
    ev["room_type_counts"] = {k: int(v) for k, v in df["room_type"].value_counts().items()}
    df["room_type_entire"] = (df["room_type"] == "Entire home/apt").astype(int)
    ev["entire_share"] = float(round(df["room_type_entire"].mean(), 4))

    # groupby().size() -> {host_id: 房源数}, 再 map 回原表
    actual = df.groupby("host_id").size()
    df["host_n_listings_actual"] = df["host_id"].map(actual)

    ev["hlc_max"] = int(df["host_listings_count"].max())
    ev["actual_max"] = int(df["host_n_listings_actual"].max())
    ev["hlc_vs_actual_spearman"] = float(round(
        df["host_listings_count"].corr(df["host_n_listings_actual"], method="spearman"), 4))
    ev["hlc_gt_actual_pct"] = float(round(
        (df["host_listings_count"] > df["host_n_listings_actual"]).mean() * 100, 2))

    return df


# ====================================================================
# 主流程
# ====================================================================

def load_clean(path=RAW_PATH):
    # low_memory=False: 原始文件 90 列, 分块读会导致混合类型警告
    df = pd.read_csv(path, low_memory=False)
    ev = {}

    ev["n_rows_raw"] = len(df)
    ev["n_cols_raw"] = int(df.shape[1])
    ev["n_hosts"] = int(df["host_id"].nunique())

    # 第 0 层
    df["price_num"] = df["price"].apply(parse_price)
    df["amenity_count"] = df["amenities"].apply(count_amenities)
    df["y"] = (df["host_is_superhost"] == "t").astype(int)

    ev["superhost_raw_missing"] = int(df["host_is_superhost"].isna().sum())
    ev["y_pos"] = int(df["y"].sum())
    ev["y_neg"] = int((1 - df["y"]).sum())
    ev["y_rate"] = round(float(df["y"].mean()), 4)

    # 分组切分的核心前提: 标签在房东内是否恒定?
    # 目标是房东级属性、在房源级建模 → 随机切分会泄漏 → 必须按 host_id 分组。
    ev["label_const_max"] = int(df.groupby("host_id")["y"].nunique().max())

    # 第 1 层
    df = step1_review_recency(df, ev)
    df = step2_amenity_count(df, ev)
    df = step3_price_missing(df, ev)

    # 第 2 层
    df = derive_dist_cbd(df, ev)
    df = derive_misc(df, ev)

    return df, ev


# ====================================================================
# 第 3 层 — 检查
# ====================================================================

def check_structural(df, ev):
    """硬断言, 只查不变量不查具体数值。挂了一定有事。"""
    assert len(df) == ev["n_rows_raw"], \
        f"行数变了 ({ev['n_rows_raw']} -> {len(df)}) — 不应删任何行"
    assert ev["superhost_raw_missing"] == 0, \
        "host_is_superhost 有缺失 — 被静默当成了非超级房东, 需单独处理"
    assert ev["label_const_max"] == 1, \
        "标签在房东内不恒定 — 分组切分的核心前提不成立"

    # 步骤 1
    assert df["days_since_last_review"].isna().sum() == 0, "dsl 填充后仍有缺失"
    assert (df["days_since_last_review"] >= 0).all(), "dsl 出现负数 — 参考日期错了?"
    assert set(df["has_review"].unique()) <= {0, 1}, "has_review 应为 0/1"
    assert (df["has_review"] == 0).sum() == df["last_review"].isna().sum(), \
        "has_review 的 0 的个数应等于 last_review 的缺失数"
    assert (df.loc[df["has_review"] == 0, "days_since_last_review"]
            == ev["dsl_fill_value"]).all(), "零评论房源的 dsl 应全等于填充值"

    # 步骤 3
    assert df["price_filled"].isna().sum() == 0, "price_filled 不应有缺失"
    assert df["price_missing"].sum() == df["price_num"].isna().sum(), \
        "price_missing 的 1 的个数应等于 price_num 的缺失数"

    # 第 2 层
    assert (df["dist_cbd"] >= 0).all(), "距离出现负数"
    assert ev["dist_inflation_pct"] > 0, "cos 修正可能写反了 (应该乘, 不是除)"

    for col in ["days_since_last_review", "has_review", "amenity_count",
                "price_filled", "price_missing", "dist_cbd",
                "room_type_entire", "host_n_listings_actual", "y"]:
        assert col in df.columns, f"缺少列: {col}"

    print("✅ 结构性断言全部通过")


def report_observed(ev):
    """只打印不卡。加新记账项时把键名加进 groups, 否则漏了不会被发现。"""
    groups = [
        ("规模",       ["n_rows_raw", "n_cols_raw", "n_hosts", "y_pos", "y_neg",
                        "y_rate", "label_const_max"]),
        ("步骤1 评论",  ["ref_date", "dsl_median", "dsl_max", "dsl_mean_super",
                        "dsl_mean_nonsuper", "no_review_n", "no_review_pct",
                        "superhost_rate_no_review", "superhost_rate_has_review",
                        "superhost_n_in_no_review", "no_review_av0_share",
                        "no_review_price_missing_share",
                        "dsl_vs_reviews_ltm_spearman", "dsl_fill_value"]),
        ("步骤2 设施",  ["amenity_count_median", "amenity_count_max",
                        "amenity_count_mean_super", "amenity_count_mean_nonsuper",
                        "amen_zero_n", "amen_naive_diff_n", "amen_naive_diff_pct",
                        "amen_naive_max_diff"]),
        ("步骤3 价格",  ["price_missing_n", "price_missing_pct", "av0_n",
                        "p_miss_given_av0", "p_miss_given_avpos", "p_av0_given_miss",
                        "superhost_rate_av0", "superhost_rate_avpos",
                        "price_median_full", "price_median_sub_min",
                        "price_median_sub_max"]),
        ("区位",       ["dist_median", "dist_max", "dist_inflation_pct"]),
        ("杂项",       ["entire_share", "room_type_counts", "hlc_max", "actual_max",
                        "hlc_vs_actual_spearman", "hlc_gt_actual_pct"]),
    ]
    print("\n" + "=" * 62)
    for name, keys in groups:
        print(f"\n[{name}]")
        for k in keys:
            print(f"  {k:32s} {ev.get(k, '  ← 没记!')}")
    missing = [k for _, ks in groups for k in ks if k not in ev]
    if missing:
        print(f"\n⚠ 漏记 {len(missing)} 项: {missing}")
    print("=" * 62)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    df, ev = load_clean()
    check_structural(df, ev)
    report_observed(ev)

    df.to_parquet(OUT_PARQUET, index=False)
    with open(OUT_EVIDENCE, "w") as f:
        json.dump(ev, f, indent=2, ensure_ascii=False)

    print(f"\n→ {OUT_PARQUET}  ({df.shape[0]} 行 × {df.shape[1]} 列)")
    print(f"→ {OUT_EVIDENCE}")
