# Stage C 交接：PCA + 聚类（K-Means & Hierarchical）

> 给负责 PCA / 聚类的队员。读完这份文件就能直接开工：数据从哪里读、用哪些特征、每一步怎么做、为什么这么做、报告每一节写什么、Limitations 怎么写，全在这里。
> A（预处理 + 相关性）和 B（监督学习 + 特征选择）的代码、结果、报告已经全部完成。

---

## 0. 一页速览

| 项 | 决定 |
|---|---|
| 数据 | `data/clean.parquet`，筛 `has_review == 1` → **21,251 行**（11,474 个房东） |
| 聚类特征（6 个连续变量） | 基本面：`dist_cbd`、`accommodates`；运营：`availability_365`、`amenity_count`、`days_since_last_review`、`host_listings_count` |
| 变换 | `days_since_last_review`、`host_listings_count`、`dist_cbd` 先 `log1p`，然后 6 列全部 `StandardScaler` |
| PCA | 对这 6 列（变换后）做 PCA。spec 要求：**报告前 2–3 个主成分**的解释方差比例，每个主成分绝对值最大的 3 个载荷，再加一张 **PC1 × PC2 散点图**（可以按目标变量 y 着色） |
| K-Means | k = 2…8，用 elbow（inertia）+ silhouette 选 k；`n_init=10, random_state=42` |
| Hierarchical | Ward linkage，**在全部 21,251 行上做**：用 `fastcluster.linkage_vector`（`pip install fastcluster`），实测 1.3 秒，结果和 scipy 完全一致；最终切成和 K-Means 相同的 k（spec 硬性要求） |
| 两种方法比较 | 在全部 21,251 行上算 Adjusted Rand Index（ARI）+ 交叉表 + 各簇大小（rubric 要求用 "actual cluster assignments and sizes" 来证明两者是否不同） |
| 和 RQ 的连接 | **y（超级房东）绝不参与聚类**，只在聚类完成后用来给每个簇算超级房东率 |
| 报告篇幅 | **最多 1 页**（全部四个大节加起来），1 张表 + 1 张图 |
| 代码文件 | `新代码/pca_cluster.py`，风格和另外三个文件一致（见第 7 节） |

---

## 1. 背景：我们的 RQ 和前面已经得到的结论

**RQ**：*Can superhost status be inferred from listing attributes and host operating behaviour once Airbnb's own criteria are excluded, and is it more closely associated with listing fundamentals or operational strategy?*

**标题**：*Operational Strategy or Listing Fundamentals? Predicting Superhost Status in the Melbourne Short-Term Rental Market*

**措辞规矩（全组统一）**：只能说 *associated with*，不能说 *driven by / caused by / leads to*。

**A、B 已经得到的结论**（你的 Discussion 要和它们呼应）：

| 来源 | 结论 |
|---|---|
| Correlation | 和超级房东关联最强的是 days_since_last_review（NMI 0.0785）；唯一的基本面变量 dist_cbd 全表最低（NMI 0.0047）；host_listings_count 是倒 U 型（3–5 套最高 46.08%，超过 100 套只有 16.92%） |
| 监督学习 | 只用基本面 macro-F1 0.485（基线 0.408），只用运营 0.701；去掉评论特征后运营仍有 0.670 |
| 特征选择 | 两种方法的前 3 全是运营特征；days_since_last_review 和 host_listings_count 两种方法都选中 |

**你这部分的角色**：前面两部分都是**有监督**的，用了 y。PCA 和聚类是**无监督**的，不看 y。它们回答的是一个互补的问题：

> 不告诉算法谁是超级房东，房源**自然**分成的几类，主要是按运营方式分开的，还是按基本面分开的？这些自然形成的类别之间，超级房东率差别大不大？

如果聚出来的簇主要沿运营维度分开，而且簇之间的超级房东率差别很大，就从无监督的角度支持了 B 的结论；如果不是，那也是一个值得讨论的发现。**不要预设结论，按数据写。**

---

## 2. 数据：读什么、用哪些行

```
data/clean.parquet    25,728 行 × 98 列，全组唯一的数据入口（preprocess.py 生成）
```

- 先运行 `新代码/preprocess.py` 生成它（大约 10 秒）
- **只用 `has_review == 1` 的 21,251 行**。原因：4,477 个零评论房源的 `days_since_last_review` 是人为填的 4,594，全部堆在同一个值上，聚类会把它们单独聚成一个**假的簇**。Correlation 部分也是同样的口径
- 这 21,251 行的超级房东率是 **35.72%**（全样本是 30.65%），报告里引用时注意口径

**关键数字**（引用时必须一致）：

| | 行数 | 房东数 | 超级房东率 |
|---|---|---|---|
| 全样本 | 25,728 | 14,113 | 30.65% |
| 有评论子集（你用这个） | 21,251 | 11,474 | 35.72% |
| 零评论房源（排除） | 4,477 | — | 6.59% |

---

## 3. 特征集：三个候选，选哪个、为什么（模板要求写）

模板原文：*Name 3 candidate feature sets for clustering, justify your choice, and apply it to both K-Means and Hierarchical clustering. Apply PCA to the same feature set.*

### 3.1 三个候选

| 候选 | 特征 | 优点 | 缺点 |
|---|---|---|---|
| **S1 只用运营** | availability_365、amenity_count、days_since_last_review、host_listings_count | 簇可以直接解读成"经营模式"（活跃 / 休眠 / 大型商业运营等） | 没有基本面，**回答不了"按什么分开"这个问题** |
| **S2 只用基本面** | dist_cbd、accommodates | 簇直接对应区位和房型规模 | 只有 2 个连续变量，PCA 没有意义（最多 2 个主成分）；B 部分已经显示基本面信号很弱 |
| **S3 混合（选这个）** | S1 + S2，共 6 个 | PCA 载荷能直接看出**主要差异轴是运营还是基本面**；簇的画像可以同时比较两类特征 | 运营 4 个对基本面 2 个，不对等（写进 Limitations） |

**选 S3 的理由**（报告里一两句）：只有同时包含两类特征，PCA 载荷和簇中心才能显示数据的主要结构是沿哪一类特征展开的，这正是 RQ 的核心问题。

### 3.2 为什么排除这些列（报告里一句话带过）

| 排除 | 原因 |
|---|---|
| `y` / `host_is_superhost` | 目标变量。聚类必须是无监督的，y 只在**事后**给簇算超级房东率 |
| `price_num` | 子集里缺失 **4,876 个（22.9%）**。用中位数补会在同一个值上堆出一个假簇；而且偏度极高（42.35，最大一晚 $50,094） |
| `minimum_nights` | **87%** 的值 ≤ 3 晚，偏度 22.22，最大 1,000，几乎没有可用的变化，只会贡献离群点 |
| `has_review` | 子集里全是 1，没有变化 |
| `price_missing`、`room_type_3` | 二元或类别变量。K-Means 和 Ward 都基于欧氏距离，0/1 变量和连续变量混在一起，距离没有意义 |
| 官方标准、评论衍生列（review_scores_*、number_of_reviews 等） | 和全组一致，一律不用 |

---

## 4. 预处理（聚类前）

### 4.1 log 变换

PCA 和 K-Means 都是基于方差或距离的，**极端值会主导结果**。子集里的偏度：

| 特征 | 偏度 | 最大值 | 处理 |
|---|---|---|---|
| host_listings_count | 4.44 | 667 | `np.log1p` |
| days_since_last_review | 2.19 | 4,593 | `np.log1p` |
| dist_cbd | 2.02 | 79.5 km | `np.log1p` |
| accommodates | 1.64 | 16 | 不变换（整数，范围小） |
| availability_365 | 0.13 | 365 | 不变换 |
| amenity_count | −0.12 | 92 | 不变换 |

`log1p(x) = log(1 + x)`，x = 0 时不会出错。

### 4.2 标准化

6 列全部 `StandardScaler()`：每列减均值、除标准差。不标准化的话，availability_365（0–365）的方差会压倒 accommodates（1–16），PCA 的第一主成分就只是"可订天数"。

**注意**：这里**不需要**训练集 / 测试集切分，聚类是描述性分析，没有"测试"这个概念，所以直接在 21,251 行上 fit 就行。

---

## 5. 分析步骤

### 5.1 PCA

spec 原文：*Apply PCA to the same feature set you used for clustering above and report the explained variance ratio for the first 2–3 components. For each of these components, identify the top 3 original variables contributing to it (by loading magnitude) and briefly interpret what that component appears to represent… Add a 2D scatter plot of the first two components, optionally coloured by your research question's target variable.*

1. `PCA()` 不限主成分数，在标准化后的 6 列上 fit
2. **报告前 3 个主成分**的 `explained_variance_ratio_` 和累计值（spec 要求前 2–3 个，6 个特征报 3 个最稳妥）
3. 每个主成分列出**绝对值最大的 3 个载荷**（`components_`），带正负号
4. 根据这 3 个载荷给主成分起名字，比如"活跃度轴"、"规模轴"、"区位轴"。rubric 要求解释"strictly grounded in its actual top 3 loading variables (not a guessed label)"，**名字必须能从那 3 个载荷直接看出来**
5. **正负号是任意的**：PCA 的方向可以整体翻转。解释时看哪些特征同号、哪些异号，不要说"正方向更好"
6. **散点图**：PC1 × PC2，**按 y（超级房东 / 非超级房东）着色**。这是 spec 建议的做法，能直接看出超级房东在主成分空间里是否集中在某个区域，和 RQ 连接最直接

**这一步对 RQ 最关键的观察**：PC1（解释方差最多的那个轴）的前 3 个载荷，主要是运营特征还是基本面特征？超级房东在散点图上是否沿某个轴分开？

### 5.2 K-Means

1. 在**标准化后的 6 列**上聚类（不是在主成分上；主成分只用来画图）
2. k 从 2 试到 8：`KMeans(n_clusters=k, n_init=10, random_state=42)`
3. 每个 k 记录 inertia（elbow 法）和 silhouette 分数
   - silhouette 在全量上是 O(n²)，很慢，用 `silhouette_score(..., sample_size=5000, random_state=42)`
4. 选 k：综合 elbow 拐点和 silhouette 最高点。**如果两者不一致，选更好解释的那个，并在报告里说明**
5. 最终的 k 用全部 21,251 行重新 fit，得到每个房源的簇标签

### 5.3 Hierarchical（层次聚类）

1. **在全部 21,251 行上做**，不要抽样。rubric 要求 "correctly applied to your chosen feature set"，只用样本有被扣分的风险
2. scipy 的标准 `linkage` 要先存一张两两距离表（约 1.8 GB），内存小的电脑会被系统杀掉（我在 3 GB 内存的环境里实测被杀了）。所以用 **`fastcluster.linkage_vector(X, method="ward")`**：
   - 内存只和行数成正比，实测 21,251 行 **1.3 秒**跑完
   - 在 2,000 行上和 scipy 的结果比较，**ARI = 1.0，完全一致**
   - 安装：`pip install fastcluster`；README 里要写上这个依赖
3. 画 dendrogram：`scipy.cluster.hierarchy.dendrogram(Z, truncate_mode="lastp", p=30)`，否则 2 万片叶子画不清
4. 从 dendrogram 上看最大的几次合并跳跃，作为选 k 的参考
5. **最终切成和 K-Means 相同的 k**：`fcluster(Z, t=k, criterion="maxclust")`。spec 硬性要求两种方法用相同的 k
6. 选 Ward 的理由：它和 K-Means 一样最小化簇内方差，比较两者时差异来自"层次式 vs 划分式"，而不是目标函数不同。single linkage 容易产生链式的长条簇，complete / average 对离群点更敏感

### 5.4 两种方法的比较

rubric 原文：*Explicit comparison of whether K-Means and Hierarchical clustering produced materially different groupings, evidenced by actual cluster assignments and sizes.*

- 两种方法都在全部 21,251 行上，直接比较标签
- **各簇大小**：两种方法的每个簇各有多少房源（rubric 点名要 sizes）
- **交叉表** `pd.crosstab(kmeans_labels, hierarchical_labels)`：看哪些簇对得上、哪些被拆开或合并了（rubric 点名要 assignments）
- **ARI**（`adjusted_rand_score`）：1 = 完全一样，0 = 和随机分组差不多
- 最后要有一句**明确的结论**："materially different" 还是 "largely the same"，rubric 要求 "a clear verdict"
- 簇编号是任意的（K-Means 的第 0 簇不一定对应层次聚类的第 1 簇），比较时按交叉表配对，不要按编号直接对比

### 5.5 簇的画像（cluster summary，模板要求）

对每种方法的每个簇，算：
- 房源数和占比
- 6 个特征的**中位数（用原始单位，不是标准化后的值）**：天、公里、套、项
- **超级房东率**（y 在这里第一次出现，只用来描述，不参与聚类）
- 给每个簇起一个描述性的名字，比如"活跃的中等规模房东"、"休眠房源"、"大型商业运营"

**和 RQ 连接的两个问题**：
1. 簇之间的超级房东率差别有多大？（最高的簇对最低的簇）
2. 区分簇的主要是哪类特征？看簇中心在标准化空间里哪些特征偏离 0 最多

### 5.6 图（只放 1 张）

**PC1 × PC2 散点图，按 y（超级房东）着色**（spec 要求必须有这张图）。标题里写出 PC1、PC2 各自解释了多少方差，两个轴标签写上主成分的名字。点太多就抽样 5,000 个画，`alpha=0.3`。可以再把 K-Means 的簇中心投影到 PC 空间、用大号标记画在同一张图上，一张图同时展示目标和簇。

scree plot 和 dendrogram 存成文件即可，不放进报告（篇幅不够）。

---

## 6. 报告怎么写（全部加起来 ≤ 1 页）

模板里 "Dimensionality Reduction & Clustering (groups of four only)" 在四个大节各出现一次，灰色提示文字写完以后要删掉。

**格式规矩（全组统一）**：Calibri 11、单倍行距；表格编号从 **Table 6** 开始，图从 **Figure 3** 开始（Table 1–5、Figure 1–2 已经被 A 和 B 用了）；每个下结论的段落都要带具体数字。

### 6.1 Methodology（约 130 词）

- 三个候选特征集各一句，选 S3 的理由一句
- 排除的列一句（价格缺失 22.9%、minimum_nights 87% ≤ 3、二元 / 类别变量不适合欧氏距离、y 只用于事后描述）
- 只用有评论的 21,251 行，理由一句
- log1p 哪三列 + 标准化，理由一句
- PCA：保留主成分的标准
- K-Means：k 的范围、选 k 的两个标准、n_init
- Hierarchical：Ward、全部 21,251 行（fastcluster 实现）、切成相同的 k
- 比较：各簇大小 + 交叉表 + ARI
- **只写方法，不写结果**

### 6.2 Results（约 120 词 + 1 张表 + 1 张图）

模板原文：*Present explained variance per component, top-3 loadings per component, and cluster summaries from K-Means and Hierarchical clustering.*

**Table 6 建议的结构**（把 PCA 和簇画像压进一张表）：

- 上半部分 PCA：每个主成分一行，列出解释方差、累计解释方差、前 3 个载荷（特征名 + 带符号的数值）
- 下半部分簇画像：每个簇一行（K-Means 和 Hierarchical 都列），列出占比、6 个特征的中位数、超级房东率

**Figure 3**：PC1 × PC2 散点图

正文只陈述数字：前 3 个主成分各自和累计解释了多少；选的 k 和理由（silhouette 值、elbow 拐点）；两种方法各簇大小和 ARI；超级房东率最高和最低的簇各是多少。**不写原因。**

### 6.3 Discussion（约 200 词）

模板原文：*Interpret what each component and cluster represents, and whether K-Means and Hierarchical produced materially different groupings.*

三段：
1. **每个主成分代表什么**：根据载荷命名；PC1 主要是运营特征还是基本面特征，以及这对 RQ 意味着什么
2. **每个簇代表什么**：用中位数描述每个簇是什么样的房源；超级房东率在簇之间差多少；哪类特征把簇分开。和 B 的结论（运营 > 基本面）、Correlation 的 host_listings_count 倒 U 型对照：簇的结果支持、补充还是挑战了它们
3. **两种方法有没有实质差别**：引用 ARI 和交叉表。差别大的话，可能的解释是 K-Means 假设簇是球形、大小相近，Ward 层次聚类是逐步合并、不能回头修正；差别小就说明聚类结构比较稳

全程用 *associated with*。**y 是事后描述**，簇的超级房东率高，不代表"属于这个簇导致成为超级房东"。

### 6.4 Limitations（约 100 词）——模板提示：*unexplained variance, scaling sensitivity, number of clusters*

**写具体的、带数字的局限，不写"样本小"这种空话。** 下面每条一两句，按实际结果把方括号里的数字填进去：

| # | 局限 | 怎么写 |
|---|---|---|
| 1 | **未解释的方差** | 保留的 [n] 个主成分只解释了 [x]% 的方差，剩下的 [100 − x]% 没有体现在二维图和解释里。如果 PC1 解释得不多，说明这 6 个特征之间没有一个占主导的结构 |
| 2 | **对缩放和变换敏感** | 结果依赖于 log1p 和标准化的选择。不做 log 的话，host_listings_count（最大 667）的离群点会单独占一个簇。可以加一个敏感性检查：不做 log 跑一次，报告 ARI |
| 3 | **簇数不确定** | elbow 拐点不明显、silhouette 只有 [s]（< 0.5 属于结构较弱），说明房源更像是连续分布，而不是界限清楚的几群。k 是"便于解释"的选择，不是数据里唯一正确的数量 |
| 4 | **Ward 的贪心合并** | 层次聚类一旦合并就不能撤销，早期合并的错误会一直保留；K-Means 对初始中心敏感（用 n_init=10 缓解）。两者 ARI 只有 [x] 的话，说明分组结构不唯一 |
| 5 | **只覆盖有评论的房源** | 排除了 4,477 个零评论房源（17.4%，超级房东率 6.59%），所以聚类结果不描述"从没被评论过"的那一群，簇的超级房东率也比全样本偏高（35.72% vs 30.65%） |
| 6 | **特征不对等、类别变量缺席** | 运营 4 个对基本面 2 个；房型（room type）因为是类别变量被排除，基本面被低估。可以改进的方法：用 Gower 距离或 k-prototypes 处理混合类型变量 |
| 7 | **房东不独立** | 12% 的行来自拥有 50 套以上房源的房东，同一房东的房源特征几乎一样，可能形成紧密的小簇、放大某些簇的规模。可以改进的方法：每个房东只取一套房，或者按房东聚合后再聚类 |
| 8 | **K-Means 的形状假设** | 假设簇是球形、大小相近；数据里如果有长条形或密度不同的群，会被强行切开 |

**考虑过但没用的方法**（写一句）：DBSCAN（对参数敏感，而且 6 维里密度难以定义）；Gaussian Mixture（允许椭圆形簇，但篇幅不够）。

---

## 7. 代码规范（和另外三个文件保持一致）

文件：`新代码/pca_cluster.py`

```
"""
COMP20008 A2 — PCA + 聚类（C 板块，先跑 preprocess.py）
输入: data/clean.parquet（只用 has_review == 1 的 21,251 行）
输出: outputs/evidence_cluster.json、outputs/pca_table.csv、outputs/cluster_profiles.csv
      outputs/fig_pca_clusters.png（报告 Figure 3）、outputs/fig_scree.png、outputs/fig_dendrogram.png
运行: python3 pca_cluster.py；每个 "# %%" 对应最终 notebook 的一个 cell
"""
```

规矩：
1. **路径**：用 `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`，不要写相对路径 `"outputs"`（从别的目录运行会报 `Read-only file system`，我们踩过这个坑）
2. **分段**：每段开头 `# %% Section N. xxx`，段内用 `# ---- N.1 xxx ----` 小标题
3. **变量名用全称**：`listings`、`scaled_features`、`kmeans_labels`，不要用 `df`、`X2`、`km`
4. **每行后面写注释**
5. **记账本**：报告里引用的每个数字都存进 `evidence` 字典，最后存成 `outputs/evidence_cluster.json`；写报告时一律从这里抄
6. **所有随机过程**都用 `random_state=42`
7. **图上的文字用英文**：matplotlib 默认字体显示不了中文，会变成方框
8. 存图用 `plt.savefig(...)` + `plt.close()`，**不要用 `plt.show()`**（终端运行会卡住）
9. 写完后**在终端里完整运行一次**（`python3 pca_cluster.py`），不要只在 VS Code 里一段一段跑：之前留在内存里的变量会掩盖错误
10. 写一份 `README_cluster.md`，格式参考 `README_model.md`

可以参考的现成写法：`model.py` 第 0 段（读数据、路径、记账本）和第 6 段（画横向条形图、存图）。

---

## 8. 不要做的事

- ❌ 用 y 参与聚类或 PCA
- ❌ 用全部 25,728 行（零评论房源会形成假簇）
- ❌ 用 A1 的数据集（Amendment #1 禁止）
- ❌ 用 `price_filled` 或 `price_num` 作聚类特征
- ❌ 用 scipy 的 `linkage` 在全量上做层次聚类（会内存溢出），用 fastcluster
- ❌ 两种方法用不同的 k
- ❌ 写 *superhost status is caused by cluster membership* 这类因果句
- ❌ 报告超过 1 页

---

## 9. 交付清单

| 交付 | 给谁 |
|---|---|
| `新代码/pca_cluster.py`（能从头到尾跑通） | 全组 |
| `outputs/evidence_cluster.json` + 3 张图 + 2 个 csv | 全组 |
| `新代码/README_cluster.md` | 全组 |
| 报告四个小节的正文 + Table 6 + Figure 3，直接写进 `A2_Report_MASTER.docx` | 全组 |

**时间**：报告和代码 10 月 9 日截止。**建议 10 月 6 日前交初稿**，留 2 到 3 天给全组统一格式、压页数。整份报告现在约 8.5 页，加上 Introduction、Conclusion 后只剩约 1 页给你，篇幅是硬约束。

有任何数字或口径对不上，以 `outputs/` 里的 json / csv 为准。
