# 基于 Spark MLlib 的电影推荐与用户画像系统

本项目提供可以独立运行的 PySpark ALS（交替最小二乘）推荐流水线和 Streamlit 数据看板。数据源为 **GroupLens MovieLens Latest Small**：

- 官方页面：https://grouplens.org/datasets/movielens/latest/
- 官方直链：https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- 数据文档：https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
- Spark ALS 文档：https://spark.apache.org/docs/3.5.7/ml-collaborative-filtering.html

## 环境与运行

推荐 Python 3.10 / 3.11、JDK 17、Apache Spark 3.5.x，内存至少 8 GB，Linux / macOS / Windows + WSL2 均可。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用 .venv\\Scripts\\activate
pip install -r requirements.txt
python download_data.py
spark-submit --master 'local[*]' train_spark.py --data-root data --out outputs
streamlit run app.py
```

若没有安装 `spark-submit` 命令，也可在已正确安装 PySpark 和 Java 的环境运行：

```bash
python train_spark.py --master 'local[*]' --data-root data --out outputs
```

## 工作流程

`ratings.csv`、`movies.csv` → Spark SQL 清洗及 8:1:1 训练/验证/测试划分 → 对照组（电影均值 / 用户与电影平滑偏置）→ 原始评分 ALS、偏置残差 ALS → 验证集调参 → 训练+验证重新拟合 → 测试集 RMSE、MAE、覆盖率 → 候选召回与 Top-10 过滤已看电影 → 用户类型喜好画像 → Streamlit 可视化。

特点：真实分割、防止测试泄漏、ALS冷启动处理、基线比较、可解释用户偏好、模型持久化和可交互展示。**没有使用年龄、性别等人口统计属性**。

## 运行后产物

- `outputs/results.json`：数据统计、验证超参结果、最终测试 RMSE/MAE/覆盖率。
- `outputs/genre_profiles/`：匿名用户电影类型画像。
- `outputs/recommendations/`：匿名用户 Top-10 电影推荐。
- `outputs/popular_movies/`：电影热度统计。
- `outputs/als_residual_model/`：Spark MLlib 模型。

> **重要：** 配套 PPT 中的预先计算实验来自 **对同一 MovieLens 原始数据的独立 ALS-WR 数值参考实现（JavaScript）**，不是 Spark 本机运行结果，也不具有相同的训练集随机划分方式。必须在提交课程项目前在自己的 Spark 环境运行此程序，并以 `results.json` 的测量值替换 PPT 中的“参考复算”页，附上 Spark UI 与 Streamlit 真实截图。这里不应将参考分数当成已运行的 Spark 得分。

## 成员分工

课程允许单人小组；本项目 PPT 的署名处需要提交者自行填写姓名、学号及实际分工。建议单人写“数据处理、特征工程、模型训练、可视化、实验分析、汇报制作”。

## 数据使用与引用

MovieLens 由 University of Minnesota GroupLens Research 维护。请遵守原始文件中的非商业、署名及再分发条款；代码包**不包含原始 MovieLens 数据文件**。学术引用：Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM TiiS. https://doi.org/10.1145/2827872

## 独立数值参考复算（非 Spark）

可执行 `node reference_benchmark.js data/ml-latest-small/ratings.csv > reference_results.json`，用于复核 PPT 中的离线模型评分和迭代收敛数据。该脚本采用独立 ALS-WR 实现，与 PySpark 的随机拆分和数值求解不同，因此不可将其输出说成 Spark 实测。

## 提交与结果真实性说明

本仓库提供源码与复现步骤，不附带虚构的 Spark 实验运行记录。
完成真实 Spark 运行后，建议将输出中的 `outputs/results.json` 另存为 `spark_results.json` 并提交到仓库，作为可查验的实验指标。
`data/` 与 `outputs/` 已纳入 `.gitignore`，避免将原始数据、模型文件或大体积运行产物误提交。

请注意：`reference_benchmark.js` 是**独立非 Spark** 的参考计算，与 Spark 训练结果不能混为一谈。
