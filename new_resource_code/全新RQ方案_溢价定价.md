# 全新 RQ 方案：What You Rent or Who You Rent From?

> 只写设计，不写任何数据结果。所有方括号 `[…]` 的地方都要跑完代码以后才能填。
> 数据源仍然只能用 Inside Airbnb 墨尔本**原始**数据（Amendment #1），不能用 A1 数据集。

---

## 1. 研究问题

**RQ**：*Can a listing's price tier be predicted from its property attributes and its host's operating profile, and is premium pricing more closely associated with what is rented (property and location) or with who rents it out (host experience and operating practices)?*

**标题**：*What You Rent or Who You Rent From? Explaining Premium Pricing in Melbourne's Short-Term Rental Market*

**中文意思**：一个房源属于便宜、中档还是高价，能不能从房子本身和房东的情况预测出来？高价主要和"租的是什么"（户型、位置）有关，还是和"跟谁租"（房东经验、经营方式）有关？

### 1.1 为什么值得研究

- **对房东**：如果溢价主要来自经营方式（经验、响应速度、评分），新房东可以通过改进经营来提价；如果主要来自房子本身，提价空间就有限
- **对政策**：维多利亚州对短租征收 Short Stay Levy，了解价格由什么构成，有助于评估这项政策对不同类型房东的影响
- **方法上**：和"超级房东"那个 RQ 一样，有一个清楚的二分框架（房产 vs 房东），每一部分分析都能落到这个框架上

### 1.2 两组特征（整个方案的骨架）

| 组 | 含义 | 候选列 |
|---|---|---|
| **Property（租的是什么）** | 房东基本改不了 | room_type、bedrooms、bathrooms（沿用 A1 的解析代码）、property_type（合并后）、dist_cbd（由经纬度算）。accommodates **不进模型**，原因见 2.2 |
| **Host（跟谁租）** | 房东的经验和经营方式 | host_tenure（由 host_since 算）、host_is_superhost、host_listings_count、host_response_rate、host_acceptance_rate、instant_bookable、minimum_nights、availability_365、number_of_reviews、review_scores_rating、amenity_count |

amenity_count 放哪一组有争议：它是房东可以随时增减的投入，所以放 Host 组。报告里要说明这个归类理由。

---

## 2. 目标变量

### 2.1 定义

**三分类价格档位**：budget / mid / premium，按价格的三分位数切。

- 用三分类而不是直接预测价格（回归），是因为作业要求报告每一类的 precision / recall / F1，而且 KNN 和决策树做分类更直观
- 用三分位数而不是固定金额，是因为每一档样本数大致相同，避免类别不平衡

### 2.2 两个必须注意的地方

1. **切分点只用训练集算**：三分位数是从数据里算出来的，在全量上算就会用到测试集的价格（泄漏）。正确做法：先切训练集 / 测试集，在训练集上算两个分位点，再用同样的分位点给测试集打标签
2. **价格要按什么口径比较**：原始价格受人数影响很大，一个住 8 人的整套房当然比单间贵。两个选择：
   - (a) 直接用每晚价格：问题变成"什么房子贵"，accommodates 会压倒一切
   - (b) **用每人每晚价格**（price / accommodates）：问题变成"同样的容量下，什么因素让房子更贵"，**推荐**，更能体现房东因素
   - 报告里两个都提，选 (b) 并说明理由
   - **(b) 带来一个必须处理的问题：数学耦合**。目标是 price ÷ accommodates，如果 accommodates 同时当特征，两者之间会出现一个**纯粹由除法造成**的负相关（分母越大，比值越小），不代表真实规律。处理办法：accommodates **不进模型**，容量信息由 bedrooms、bathrooms 代表；在相关性部分专门指出这一点；在 Limitations 里承认 bedrooms 和 accommodates 高度相关，耦合不能完全消除

### 2.3 没有价格的房源

没有价格的房源**没有标签**，只能排除。这不是删数据偏好，而是标签本身不存在。但要在报告里说明：缺价格的房源大部分是下架房源，所以研究范围是"**在售房源**"，结论不能推广到下架房源。

---

## 3. 预处理：6 个候选，做 3 个，弃 3 个

模板要求：列出 6 个候选，做其中 3 个，用数据集里的实际数字说明另外 3 个为什么不做，每个做了的步骤都要给出 before / after 的数字。

### 3.1 做的 3 个

spec 要求：每个做了的步骤，都要写出**考虑过的替代做法**以及**为什么没选**，并引用数据集里的实际数字；还要给出 before / after 的数字。

| 步骤 | 做法 | 考虑过的替代做法（为什么没选） | before / after 要报告什么 | 和 RQ 的关系 |
|---|---|---|---|---|
| **1. property_type 合并** | 把原始的几十种类型按"房产性质"合并成少数几组（比如 apartment / house / unique stay / hotel-like），合并规则写成明确的映射表 | (a) 直接独热编码全部类型：类别太多、大量类别不到 10 个房源，模型学不到；(b) 只保留 room_type：丢掉了"公寓 vs 独栋 vs 特色住宿"这种和价格直接相关的区分 | 类别数从 [n] 降到 [m]；被合并的行数和占比；合并前后各组的价格档位分布 | Property 组的核心特征。**spec 列表里的现成候选**，在价格 RQ 里比在超级房东 RQ 里更有意义 |
| **2. host_since → host_tenure** | 参考日（数据里最晚的日期）减 host_since，换算成年 | (a) 用天数：数值太大、不直观，和年的信息量一样；(b) 分成新手 / 老手两档：丢失连续信息，切点人为 | 原本是日期文字 → 数值；缺失行数（host_since 为空的房东） | Host 组的核心特征："经验"的代理 |
| **3. host_response_rate / acceptance_rate** | "95%" → 0.95；"N/A" → 空值，加一个 `response_rate_missing` 标记，空值在 Pipeline 里用训练集中位数补 | (a) 删掉缺失行：缺失可能集中在不活跃的房东，删掉会偏；(b) 只补中位数不加标记：抹掉"从不回复"这个信号 | 缺失比例；有 / 无响应率的房源在三个价格档位上的分布差别 | 经营方式的代理；缺失本身可能有含义 |

**bathrooms 怎么办**：spec 的相关性 Set A 原文写着 bathrooms 是 *"the numeric variable your Assignment 1 pipeline already derived from 'bathrooms_text' — reuse it here rather than re-parsing"*。也就是说，解析 bathrooms_text 被视为 A1 已经做过的事，**不要把它算作这次的 3 个步骤之一**，直接沿用 A1 的解析代码生成这一列（只用代码，不用 A1 的数据集）。

### 3.2 弃的 3 个（要用实际数字说明）

| 候选 | 弃用理由的方向（跑数据后补具体数字） |
|---|---|
| **价格离群值截尾** | 价格是目标变量的来源，用三分位数切档位后，极端值只是落进 premium 档，不影响标签。截尾反而改变真实的价格分布 |
| **neighbourhood_cleansed 做 target encoding** | 用"这个区的平均价格档位"当特征，标签信息会直接漏进特征里；就算只在训练集上算，样本少的区也会过拟合。用 dist_cbd 这个连续变量代替区位 |
| **补 bedrooms 缺失值** | 先查缺失是否随房型变化。如果单间缺得多、整套缺得少，缺失就是"语义性"的，硬补等于造数据。容量由 bathrooms 和 room_type 一起代表 |

---

## 4. 相关性分析

### 4.1 变量（5 个，两边各有代表 + 目标）

| 变量 | 组 | 理由 |
|---|---|---|
| dist_cbd | Property | 最纯的区位变量，房东改不了 |
| ~~accommodates~~ | — | **不选**：它是目标（每人每晚价格）的分母，和目标的相关是除法造成的（见 2.2） |
| bedrooms 或 bathrooms | Property | 户型 |
| host_tenure | Host | 经验 |
| review_scores_rating | Host | 服务质量的代理 |
| price tier（目标） | — | 作业要求必须包含 |

### 4.2 方法和要注意的点

- 4 种方法：Pearson、Spearman、MI、NMI，和原方案一样
- **目标是有序的三分类**（budget < mid < premium），所以 **Spearman 最适合**目标相关的那几对；Pearson 把档位当作等距的 0/1/2，要在报告里说明这个假设
- review_scores_rating 极度左偏（大部分都很高），Pearson 会被少数低分拉偏，正好可以拿来讨论 Pearson 和 Spearman 的分歧
- 没有评分的房源（零评论）要单独处理：要么只在有评分的子集上算，要么说明排除了多少

---

## 5. 监督学习

### 5.1 设计

| 项 | 做法 |
|---|---|
| 模型 | 多数类基线 + KNN + 决策树 |
| 切分 | **按 host_id 分组 + 分层**（StratifiedGroupKFold）。多房房东往往统一定价，随机切分会让同一房东同时出现在训练集和测试集 |
| 标签 | 在训练集上算三分位数，再给训练集、测试集打标签（见 2.2） |
| 预处理 | 放进 Pipeline：中位数补空值（只用训练集）、类别变量独热编码、只给 KNN 标准化 |
| 调参 | 训练集上 5 折分组交叉验证；KNN 调 k，决策树调深度；候选值要扩到分数开始下降为止 |
| 主指标 | **macro-F1**（三类各占三分之一权重）；同时报告每类 P / R / F1 和 3×3 混淆矩阵 |
| 置信区间 | 按房东做配对 bootstrap。**要写清楚为什么选它而不是其他方法**（rubric 原文要求 *justify why that method was chosen over the alternatives*）：测试集只能用一次，要给最终测试成绩本身算不确定性，只能对测试集重抽样，所以不用 repeated CV；多房房东的房源不独立，所以按房东抽；配对才能得到两个模型差值的区间 |
| 分层的理由 | 三分位切出来三类各约三分之一，类别是平衡的，但仍然分层，保证训练集和测试集里三类比例一致。**理由要对着实际的类别比例写**（rubric 原文：*justified against your target's actual class balance*） |
| 改过默认值的超参数 | rubric 原文：*For every hyperparameter changed from its default: state the default, your chosen value, and its specific effect on your reported metric.* 候选列表里**一定要包含默认值**（KNN 的 k=5，决策树的 max_depth=None），这样报告里才能写"从默认值 X 改成 Y，CV macro-F1 从 a 变成 b" |
| 和基线比较 | 基线用同样的指标报告，**写出绝对提升**（比如"+0.30 macro-F1"），不能只写"明显高于基线" |
| 每类指标 | **所有模型、所有三类**都要报告 precision / recall / F1，包括基线 |

### 5.2 有序分类特有的分析

三分类是有序的，所以**错误也有轻重**：把 premium 判成 mid 是"差一档"，判成 budget 是"差两档"。混淆矩阵里要单独看"差两档"的比例。如果大部分错误都只差一档，说明模型抓到了价格的方向，只是边界模糊。这一点是原 RQ 没有、可以加分的讨论点。

### 5.3 RQ 实验（和原方案同样的逻辑）

| 组 | 特征 | 检验什么 |
|---|---|---|
| A | 只用 Property | RQ 后半句 |
| B | 只用 Host | RQ 后半句 |
| C | 全部 | 对照组 |
| D | 全部，去掉 review_scores_rating 和 number_of_reviews | 结论是不是全靠评论撑着（评论需要先有人入住，和价格可能互相影响） |
| E | 全部，去掉 dist_cbd | 区位到底贡献多少 |
| F | Property 去掉 bedrooms 和 bathrooms | 容量信息拿掉以后，房型和区位还剩多少解释力（accommodates 已因数学耦合不进模型，见 2.2） |

超参数固定为全特征下调出的值，主要比较 CV 分数，关键对比（A vs B）用配对 bootstrap。

---

## 6. 特征选择

- **Filter**：`mutual_info_classif`，每个特征单独和价格档位算互信息
- **Embedded**：调好的决策树的 impurity importance（独热编码的列加回原特征）
- 两种方法都只用训练集，各取前 3
- **验证**：只用每份前 3 重新训练，CV 分数和全特征比较
- **交叉核对**：测试集上做置换重要性
- **难例**：rubric 只认两种情况，二选一，要明确写出属于哪一种：
  - (a) **两种方法排第一的特征会导致不同预测的房源**。如果 Filter 和 Embedded 排第一的特征不一样（比如 Filter 第一是 bedrooms，Embedded 第一是 dist_cbd），找一个"按 bedrooms 看像 premium、按 dist_cbd 看像 budget"的房源
  - (b) **在排名第一的特征上是离群值的房源**。如果两种方法排第一的特征相同，就走这条路：找一个在该特征上远离同档位其他房源的房源，用**百分位**说明它有多极端（比如"在 premium 档里排第 99 百分位"）
  - 两种情况都要：给出房源 ID，**引用它的实际数值**，和各档中位数对比，解释它为什么难判断。优先挑"差两档"的错误（真实 premium、预测 budget），讨论价值最大

**预期的讨论点**（跑完再确认）：
- bedrooms 和 bathrooms 高度相关（都代表容量），Filter 可能把两者都选进前 3，Embedded 往往只用其中一个。这就是"Filter 忽略冗余"的例子
- dist_cbd 是连续变量，impurity importance 可能会高估它

---

## 7. PCA + 聚类

### 7.1 三个候选特征集

| 候选 | 特征 | 评价 |
|---|---|---|
| S1 只用 Property | accommodates、bedrooms、bathrooms、dist_cbd（聚类不涉及目标，accommodates 可以用） | 簇 = 房型 / 区位类型，好解释，但看不出房东的作用 |
| S2 只用 Host | host_tenure、host_listings_count、response_rate、availability_365、amenity_count | 簇 = 房东类型（新手 / 老手 / 商业运营） |
| **S3 混合（推荐）** | S1 + S2 中的连续变量 | PCA 载荷能直接显示主要差异轴是房产还是房东，直接回答 RQ |

### 7.2 做法要点

- 只用连续变量（K-Means 和 Ward 基于欧氏距离），偏态变量先 log1p，再标准化
- **价格档位不参与聚类**，只在事后给每个簇算 premium 占比
- 层次聚类在**全部数据**上做（用 `fastcluster.linkage_vector(X, method="ward")`，只占和行数成正比的内存；scipy 的标准 linkage 会内存溢出），切成和 K-Means **相同的 k**（spec 硬性要求）
- 两种方法的比较要用**各簇大小 + 交叉表**（rubric 原文：*evidenced by actual cluster assignments and sizes*），最后给出"是否实质不同"的明确结论
- K-Means 的 k 用 elbow 法选（spec 原文：*using the elbow method or a value tied to the research question*）
- 每个簇报告：占比、各特征中位数、三个价格档位的占比
- **PCA**（spec 原文要求）：报告**前 2–3 个主成分**的解释方差比例；每个主成分绝对值最大的 3 个载荷，并根据这 3 个载荷解释它代表什么；加一张 **PC1 × PC2 散点图，按价格档位着色**

---

## 8. Limitations（每部分都要有具体的点）

| 部分 | 局限 |
|---|---|
| **数据本身** | 挂牌价 ≠ 成交价（可能有折扣、动态定价）；价格不含清洁费；**快照只反映抓取那几周的价格**，墨尔本的旅游淡旺季（比如澳网、F1）会让价格大幅波动，结论只适用于抓取时的季节 |
| **研究范围** | 没有价格的房源被排除，而它们大部分是下架房源，所以结论只适用于在售房源 |
| **目标定义** | 三分位切点是人为的，切点附近的房源只差几块钱就被分到不同档；可以改进的方法：做回归，或者试四分位 / 五分位看结论是否稳定 |
| **预处理** | bathrooms_text 有少量无法解析的格式；host_tenure 只代表在平台上的时长，不代表实际的房东经验 |
| **相关性** | Pearson 把有序档位当作等距；样本大，几乎所有系数都"显著"，只能看效应大小 |
| **目标与特征的数学耦合** | 目标是每人每晚价格，accommodates 虽然不进模型，但 bedrooms、bathrooms 和它高度相关，残余的耦合可能让 Property 组的作用被高估 |
| **监督学习** | review_scores 和价格可能互相影响（高价房客的期望更高，反过来影响评分），所以只能说"相关"；Property 和 Host 两组特征数不对等 |
| **特征选择** | 户型三个特征高度相关，置换重要性会互相掩护；impurity importance 偏爱连续特征 |
| **聚类** | 未解释的方差；对 log 变换和标准化敏感；簇数没有唯一正确答案；层次聚类只用了样本 |
| **独立性** | 多房房东的房源定价高度一致，样本不独立；已用分组切分处理建模部分，但相关性和聚类部分没有处理 |

---

## 9. 和原 RQ（超级房东）比较

| | 原 RQ：超级房东 | 新 RQ：价格档位 |
|---|---|---|
| 目标 | 二分类、房东级（同一房东标签一样） | 三分类有序、房源级（同一房东不同房子价格可以不同） |
| 分组切分 | 必须（标签在房东内恒定） | 仍然需要（多房房东统一定价） |
| 最大的泄漏风险 | 官方评选标准、评论衍生指标 | 价格衍生指标（estimated_revenue_l365d = 价格 × 入住率，**必须排除**）；三分位切点在全量上算 |
| 特殊加分点 | 零评论房源的选择偏差 | 有序分类的"差一档 vs 差两档"错误分析 |
| 特有的陷阱 | 评论新近度是被剔除指标的近亲 | 目标是比值，分母不能当特征（数学耦合） |
| 可以复用的代码 | — | 预处理里的距离计算、设施解析；A1 的 bathrooms 解析；model.py 的 Pipeline、分组切分、bootstrap、特征选择框架几乎可以原样复用 |

---

## 10. 执行顺序

1. 预处理（3 个步骤 + 3 个弃用证据 + 距离、设施等派生列 + 沿用 A1 代码生成 bathrooms）→ `clean.parquet`
2. 相关性（5 个变量 × 4 种方法 + 分歧分析 + 图）
3. 监督学习（切分 → 在训练集上算三分位、打标签 → 调参 → 测试 → RQ 实验）
4. 特征选择（Filter vs Embedded + 置换重要性 + 难例）
5. PCA + 聚类
6. 每完成一步，把报告要引用的数字存进 evidence json，写报告时一律从这里抄
7. 代码最后合并成**一个 `code.ipynb` + `README.txt`**（spec 第 2 节）；注意 Jupyter 里没有 `__file__`，路径要用 notebook 所在目录；README 和报告里**声明 GenAI 的使用**（spec 8.4）
