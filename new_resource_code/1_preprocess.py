"""
COMP20008 A2 — 预处理（A 板块第 1 个文件，必须最先运行）
输入: data/listings.csv（Inside Airbnb 墨尔本原始数据，25,728 行 × 90 列；Amendment #1：只能用原始数据，不能用 A1 数据集）
输出: data/clean.parquet（25,728 行 × 98 列 = 原始 90 列 + 新增 8 列，全组后面所有分析都只读这张表）
      outputs/evidence.json（报告预处理部分引用的所有数字，写报告时一律从这里抄）
运行: python3 preprocess.py（从哪个目录运行都可以）；每个 "# %%" 是一段，可在 VS Code 里单独运行，也对应最终 notebook 的一个 cell
"""

# %% Section 0. 准备：导入工具库、找到项目根目录、读原始数据、建记账本
import json                                                                  # 两个用途：① 解析 amenities 那串文字 ② 把记账本存成 evidence.json
import os                                                                    # 拼接文件路径（Windows / Mac 的路径写法不同，用它就不用管）
import numpy as np                                                           # 数学工具：三角函数（算距离）、随机抽样（验证中位数稳定性）
import pandas as pd                                                          # 表格工具：读 csv、筛选、分组统计、存 parquet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录：本文件在"新代码"文件夹里，往上走一层就是根目录；这样从哪里运行都能找到 data/ 和 outputs/
round_value = lambda value, digits=4: round(float(value), digits)            # 小工具：把 numpy 数字转成 Python 普通小数并保留 digits 位小数（numpy 类型直接存 json 会报错）

listings = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "listings.csv"), low_memory=False)  # 读原始数据：一行 = 一个房源；low_memory=False 让 pandas 一次读完整列再判断类型，避免混合类型警告
evidence = {"n_rows": len(listings),                                         # 记账本（字典）：后面每算出一个报告要用的数字就往里记一笔；先记总行数 25,728
            "n_hosts": listings["host_id"].nunique()}                        # 不同房东的个数 14,113（一个房东可以有多套房，所以房东数 < 房源数）

# %% Section 1. 解析：把文字列变成数字列（做法沿用 A1）
# ---- 1.1 价格：文字 → 数字 ----
listings["price_num"] = (listings["price"]                                   # 原始价格是文字，比如 "$1,306.99"
                         .str.replace(r"[$,]", "", regex=True)              # 删掉美元符号和千位逗号 → "1306.99"（整串替换，不能用正则"找一段数字"，否则遇到千位逗号只会抓到 306.99）
                         .astype(float))                                     # 文字 → 小数 1306.99；原本缺价格的仍然是空值 NaN（第 4 段再处理）

# ---- 1.2 设施个数：文字列表 → 个数 ----
listings["amenity_count"] = listings["amenities"].apply(                     # amenities 看起来像列表，其实是一串文字：'["Wifi", "Kitchen", "HDTV with Chromecast, DVD player"]'
    lambda amenities_text: len(json.loads(amenities_text)))                  # json.loads 把文字解析成真正的 Python 列表，再用 len 数个数 → 3（直接按逗号切会把第三项拆开，数成 4）

# ---- 1.3 目标变量 y ----
listings["y"] = (listings["host_is_superhost"] == "t").astype(int)           # 原本是 "t"/"f" 文字 → 超级房东 1，否则 0；这就是模型要预测的答案
evidence["y_rate"] = round_value(listings["y"].mean())                       # 记：整体超级房东率 30.65%（0/1 列的平均值 = 1 的比例）
is_superhost = listings["y"] == 1                                            # 筛选条件（True/False 列）：这个房源是不是超级房东的；后面算"超级房东 vs 非超级房东"的对比时反复用

# ---- 1.4 一个房东多套房：说明 B 板块为什么必须按房东分组切分 ----
listings_per_host = listings.groupby("host_id").size()                       # 每个房东名下有几套房（一个 Series：索引是房东 ID，值是房源数）
evidence["single_listing_hosts"] = int((listings_per_host == 1).sum())       # 记：只有 1 套房的房东 11,383 个
evidence["multi_listing_hosts"] = int((listings_per_host > 1).sum())         # 记：有 2 套以上的房东 2,730 个
evidence["multi_listing_listings"] = int(listings_per_host[listings_per_host > 1].sum())  # 记：这 2,730 个房东名下一共 14,345 个房源
evidence["multi_listing_share"] = round_value(evidence["multi_listing_listings"] / len(listings))  # 记：占全部房源 55.76% → 超过一半的房源和别的房源同属一个房东；而超级房东是给房东的称号，同一房东名下标签都一样，随机切分会让同一房东同时出现在训练集和测试集（模型靠"认出老熟人"答对），所以 B 板块必须按 host_id 分组切分

# %% Section 2. 预处理步骤 1：距上次评论天数 + 零评论房源怎么处理
# ---- 2.1 算天数 ----
last_review_date = pd.to_datetime(listings["last_review"], errors="coerce")  # 最后一条评论的日期：文字 → 日期；从来没有评论的房源变成空值 NaT（errors="coerce" = 转不了就变空，不报错）
listings["has_review"] = last_review_date.notna().astype(int)                # 新列：有过评论 1，从来没有 0。必须在填充之前做，填完就分不出哪些原本是空的了
has_review_mask = listings["has_review"] == 1                                # 筛选条件：有评论的房源（下面算平均数、中位数只用这些，否则 4,477 个填充值会把结果拉偏）
days_since_review = (last_review_date.max() - last_review_date).dt.days      # 天数 = 参考日 − 最后评论日期；参考日 = 整个数据集里最晚的评论日期 2026-06-28（不用今天：数据是快照，用今天的话每天跑结果都不一样，无法复现）

# ---- 2.2 记账：报告步骤 1 要用的数字 ----
evidence.update({
    "ref_date": str(last_review_date.max().date()),                          # 参考日 2026-06-28
    "dsl_median": round_value(days_since_review[has_review_mask].median()),  # 有评论房源的天数中位数 84
    "dsl_max": int(days_since_review.max()),                                 # 最大天数 4,593（最久没评论的那个房源）
    "dsl_mean_super": round_value(days_since_review[has_review_mask & is_superhost].mean(), 2),       # 超级房东平均 93.10 天 → 更活跃
    "dsl_mean_nonsuper": round_value(days_since_review[has_review_mask & ~is_superhost].mean(), 2),   # 非超级房东平均 727.03 天（~ 表示"取反"，即不是超级房东）
    "no_review_n": int((~has_review_mask).sum()),                            # 零评论房源 4,477 个
    "no_review_pct": round_value((~has_review_mask).mean() * 100, 2),        # 占全部 17.4%
    "superhost_rate_no_review": round_value(listings.loc[~has_review_mask, "y"].mean()),   # 零评论组的超级房东率 6.59%
    "superhost_rate_has_review": round_value(listings.loc[has_review_mask, "y"].mean()),   # 有评论组 35.72% → 删掉零评论房源会把正类率从 30.65% 推高到 35.72%，这就是 A1 数据集的偏差，所以不能删行
    "superhost_n_in_no_review": int((~has_review_mask & is_superhost).sum()),             # 零评论房源里仍有 295 个超级房东 → has_review 不是标签的替身
    "superhost_n_in_no_review_host_reviewed": int(                           # 其中 271 个的房东另有带评论的房源（超级房东身份来自房东的其他房子）
        listings.loc[~has_review_mask & is_superhost, "host_id"]             #   取出这 295 个房源的房东 ID……
        .isin(listings.loc[has_review_mask, "host_id"]).sum()),              #   ……看有几个出现在"有评论房源"的房东名单里
    "dsl_vs_reviews_ltm_spearman": round_value(                              # 天数和"近 12 个月评论数"的 Spearman 相关 −0.8155：关系很强，说明天数部分代替了被剔除的评论数（写进 Limitations）
        days_since_review[has_review_mask].corr(listings.loc[has_review_mask, "number_of_reviews_ltm"], method="spearman")),
})

# ---- 2.3 填充零评论房源 ----
evidence["dsl_fill_value"] = evidence["dsl_max"] + 1                         # 填充值 4,594 = 最大天数 + 1，意思是"比最久没评论的还要久"（填中位数 84 意思就反了：84 天代表"最近挺活跃"）
listings["days_since_last_review"] = days_since_review.fillna(evidence["dsl_fill_value"])  # 新列：有评论的是真实天数，零评论的填 4,594；配合 has_review 列，模型能分清哪些是填的

# %% Section 3. 预处理步骤 2：设施个数（第 1 段已算好，这里证明修正效果 + 看区分度）
# ---- 3.1 before / after：A1 的错误数法会数错多少 ----
amenities_inner_text = listings["amenities"].str.strip().str[1:-1]           # A1 的错误数法第 1 步：去掉首尾空格，再去掉最外面的方括号 [ ]
naive_amenity_count = (amenities_inner_text.str.split(",").str.len()         # 第 2 步：按逗号切开数段数（名字里自带逗号的设施会被多数）
                       .where(amenities_inner_text != "", 0))                # 空列表 "[]" 去掉括号后是空字符串，切开会算成 1 段，这里改回 0
amenity_count_differs = naive_amenity_count != listings["amenity_count"]     # 两种数法结果不一样的行（True = 被 A1 数法数错了）

# ---- 3.2 记账 ----
evidence.update({
    "amen_naive_diff_n": int(amenity_count_differs.sum()),                   # before/after：3,096 行会被数错
    "amen_naive_diff_pct": round_value(amenity_count_differs.mean() * 100, 2),  # 占 12.03%
    "amen_naive_max_diff": int((naive_amenity_count - listings["amenity_count"]).abs().max()),  # 单个房源最多差 21 项
    "amen_zero_n": int((listings["amenity_count"] == 0).sum()),              # 真正一项设施都没有的房源 29 个（真实空列表，不是解析出错）
    "amen_zero_pct": round_value((listings["amenity_count"] == 0).mean() * 100, 2),  # 占 0.11%
    "amenity_count_mean_super": round_value(listings.loc[is_superhost, "amenity_count"].mean(), 2),     # 超级房东平均 44.08 项
    "amenity_count_mean_nonsuper": round_value(listings.loc[~is_superhost, "amenity_count"].mean(), 2), # 非超级房东平均 31.03 项 → 有区分度
})

# %% Section 4. 预处理步骤 3：价格缺失 → 中位数填补 + 缺失标记
# ---- 4.1 两个筛选条件 + 中位数 ----
price_is_missing = listings["price_num"].isna()                              # 筛选条件：缺价格的房源（6,553 个）
is_unavailable_all_year = listings["availability_365"] == 0                  # 筛选条件：未来一年可订 0 天 = 下架了
price_median = listings["price_num"].median()                                # 全部房源价格的中位数 $243.67（空值自动跳过；用中位数不用平均数，因为有一晚几万的极端价格）

# ---- 4.2 验证"用全量数据算中位数"影响可以忽略 ----
random_generator = np.random.default_rng(42)                                 # 固定种子 42 的随机数生成器 → 每次运行抽到的样本一样，结果可复现
subsample_medians = [                                                        # 随机抽 75% 的房源（约等于训练集大小）算中位数，重复 10 次
    listings["price_num"].iloc[random_generator.choice(len(listings), int(len(listings) * 0.75), replace=False)].median()  # choice：从 0~25727 里不放回地抽 19,296 个行号
    for _ in range(10)]                                                      # _ 表示循环变量用不到

# ---- 4.3 记账 ----
evidence.update({
    "price_missing_n": int(price_is_missing.sum()),                          # 缺价格 6,553 行
    "price_missing_pct": round_value(price_is_missing.mean() * 100, 2),      # 占 25.47%
    "p_miss_given_av0": round_value(price_is_missing[is_unavailable_all_year].mean()),   # 下架房源中缺价格的比例 0.9985
    "p_miss_given_avpos": round_value(price_is_missing[~is_unavailable_all_year].mean()),# 在售房源中缺价格的比例 0.0222 → 缺不缺几乎完全由下架决定，不是随机缺的
    "p_av0_given_miss": round_value(is_unavailable_all_year[price_is_missing].mean()),   # 反过来问：缺价格房源中下架的比例 0.9335（分子分母都换了，写报告别和第一个搞混）
    "superhost_rate_av0": round_value(listings.loc[is_unavailable_all_year, "y"].mean()),    # 下架组超级房东率 8.13%
    "superhost_rate_avpos": round_value(listings.loc[~is_unavailable_all_year, "y"].mean()), # 在售组 37.68% → 删掉缺价格的行 ≈ 删掉下架房源，数据会偏，所以不能删
    "price_median_full": round_value(price_median, 2),                       # 填补用的中位数 $243.67
    "price_median_sub_min": round_value(min(subsample_medians), 2),          # 10 次抽样最小 $243.0
    "price_median_sub_max": round_value(max(subsample_medians), 2),          # 最大 $245.0 → 和全量相差不到 1%，用全量算影响可忽略（B 板块建模时改在 Pipeline 里只用训练集补，彻底避免）
})

# ---- 4.4 生成两个新列 ----
listings["price_missing"] = price_is_missing.astype(int)                     # 新列：原本缺价格 1，否则 0（告诉模型"这里的价格是补的，而且这个房源大概率下架了"）
listings["price_filled"] = listings["price_num"].fillna(price_median)        # 新列：缺的价格补成 $243.67（price_num 保留带空值的原版，B 板块用 price_num）

# %% Section 5. 派生列：到墨尔本市中心的距离（建模要用，不算三个受评预处理步骤）
north_south_km = (listings["latitude"] + 37.8183) * 111                      # 南北方向差多少公里：原点 Flinders Street 火车站（纬度 −37.8183）；纬度差 1 度 ≈ 111 公里
east_west_km = (listings["longitude"] - 144.9671) * 111 * np.cos(np.radians(37.8183))  # 东西方向差多少公里：经度差还要乘 cos(纬度) 修正（越靠近南北极经线越密，墨尔本经度 1 度约 88 公里）；np.cos 只认弧度，先用 np.radians 把角度转成弧度
listings["dist_cbd"] = np.sqrt(north_south_km ** 2 + east_west_km ** 2)      # 新列：勾股定理合成直线距离（公里）；平面近似，墨尔本范围不大，误差可忽略

# %% Section 6. 三个弃用候选步骤的证据（这些步骤没做，只算报告里"为什么不做"的数字，不修改任何数据）
bedrooms_is_missing = listings["bedrooms"].isna()                            # 筛选条件：缺卧室数
property_type_counts = listings["property_type"].value_counts()              # 每种房源类型各有多少个（从多到少排）
evidence.update({
    # ---- 6.1 候选：补 bedrooms / beds 缺失值（弃用） ----
    "bedrooms_missing_pct": round_value(bedrooms_is_missing.mean() * 100, 2),            # 卧室数缺失 18.19%
    "beds_missing_pct": round_value(listings["beds"].isna().mean() * 100, 2),            # 床数缺失 27.44%
    "bedrooms_missing_by_room_type": bedrooms_is_missing.groupby(listings["room_type"]).mean().round(4).to_dict(),  # 按房型拆开：整套 2.47% vs 单间 61.28%，差约 25 倍 → 租单间时房东不填卧室数，缺失本身有含义，硬补是造假
    # ---- 6.2 候选：合并稀有的 property_type（弃用） ----
    "property_type_n_categories": len(property_type_counts),                 # 一共 82 种房源类型
    "property_type_n_under_10": int((property_type_counts < 10).sum()),      # 其中 41 种不到 10 个房源
    "property_type_rare_rows": int(property_type_counts[property_type_counts < 10].sum()),  # 这 41 种加起来只有 111 行
    "property_type_rare_row_pct": round_value(property_type_counts[property_type_counts < 10].sum() / len(listings) * 100, 2),  # 占 0.43% → 合并最多影响 0.43% 的数据，收益很小
    "property_type_top1_share_within_room_type": listings.groupby("room_type")["property_type"]   # 每种房型里，最多的那种房源类型占几成
        .agg(lambda types: types.value_counts(normalize=True).iloc[0]).round(4).to_dict(),       #   value_counts(normalize=True) 给出比例，iloc[0] 取最大的那个；四类里三类过半 → 主要信息已被 room_type 覆盖
    "entire_share": round_value((listings["room_type"] == "Entire home/apt").mean()),    # 整套房占全部房源 73.18%
    # ---- 6.3 候选：minimum_nights 分成短租 / 长租（弃用） ----
    "min_nights_ge30_n": int((listings["minimum_nights"] >= 30).sum()),      # 最少要住 30 晚以上的房源只有 426 个
    "min_nights_ge30_pct": round_value((listings["minimum_nights"] >= 30).mean() * 100, 2),  # 占 1.66% → 分出来的长租组太小，模型学不到东西
})

# %% Section 7. 检查并保存
assert len(listings) == evidence["n_rows"]                                   # 检查 1：行数没变（确认没有删任何一行）
assert listings.groupby("host_id")["y"].nunique().max() == 1                 # 检查 2：同一个房东名下所有房源的标签都一样（每个房东名下不同 y 值的个数最多是 1）；这是 B 板块按房东分组切分的前提
assert listings[["days_since_last_review", "price_filled"]].notna().all().all()  # 检查 3：填充后这两列没有空值（第一个 .all() 按列判断，第二个汇总成一个 True/False）
listings.to_parquet(os.path.join(PROJECT_ROOT, "data", "clean.parquet"), index=False)  # 保存干净数据（parquet 比 csv 小、读得快，还能保留每列的数据类型）
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)            # 没有 outputs 文件夹就建一个（exist_ok=True：已经存在也不报错）
with open(os.path.join(PROJECT_ROOT, "outputs", "evidence.json"), "w") as evidence_file:  # 打开（新建）evidence.json 准备写入
    json.dump(evidence, evidence_file, indent=2, ensure_ascii=False, default=int)  # 保存记账本：indent=2 缩进好读；ensure_ascii=False 中文不转码；default=int 处理 numpy 整数
print(json.dumps(evidence, indent=2, ensure_ascii=False, default=int), f"\n→ clean.parquet {listings.shape}")  # 打印记账本和最终表格大小，方便核对
