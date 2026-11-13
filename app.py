"""Interactive Streamlit dashboard for Spark-generated artifacts."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='MovieLens 推荐与用户画像',page_icon='🎬',layout='wide')
st.title('🎬 电影智能推荐与用户画像分析')
st.caption('MovieLens Latest Small | Spark MLlib ALS | 非人口学属性的行为画像')

OUT=Path('outputs')
if not (OUT/'results.json').exists():
    st.error('尚无 Spark 实验输出。请先运行 spark-submit --master "local[*]" train_spark.py')
    st.stop()

stats=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
def read_csv_dir(p):
    files=list(p.glob('part-*.csv'))
    return pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame()

rec=read_csv_dir(OUT/'recommendations')
profile=read_csv_dir(OUT/'genre_profiles')
pop=read_csv_dir(OUT/'popular_movies')
a,b,c,d=st.columns(4)
a.metric('评分记录',f"{stats['data']['ratings']:,}")
b.metric('匿名用户',f"{stats['data']['users']:,}")
c.metric('电影数',f"{stats['data']['movies']:,}")
d.metric('测试 RMSE',f"{stats['test']['rmse']:.4f}")

t1,t2,t3=st.tabs(['个性化推荐','用户画像','模型与全局统计'])
with t1:
    if rec.empty:st.warning('推荐清单尚未输出。')
    else:
        uid=st.selectbox('用户ID（匿名）',sorted(rec.userId.unique()),index=0)
        st.dataframe(rec.loc[rec.userId==uid,['position','title','genres','estimated_rating']].sort_values('position'),hide_index=True,use_container_width=True)
        st.info('已看电影剔除基于训练与验证集的已知历史；当前仅支持已有评分记录的用户，尚未实现新用户冷启动。')
with t2:
    if not profile.empty:
        uid=st.selectbox('查看画像的用户ID',sorted(profile.userId.unique()),index=0)
        usr=profile[profile.userId==uid].query('rated_count>=5').sort_values('positive_ratio',ascending=False)
        st.subheader('类型喜好占比（评分≥4.0）')
        st.bar_chart(usr.set_index('genre')['positive_ratio'])
        st.dataframe(usr[['genre','rated_count','avg_rating','positive_ratio']],hide_index=True,use_container_width=True)
        st.caption('画像只依据匿名评分和电影类型计算，数据集不包含年龄、性别等人口统计信息。')
with t3:
    st.subheader('评分分布')
    df=pd.DataFrame(stats['rating_histogram']).set_index('rating')
    st.bar_chart(df['count'])
    st.subheader('模型验证指标')
    exp=pd.DataFrame(stats['hyperparameters'])
    st.dataframe(exp,hide_index=True,use_container_width=True)
    st.write('最终测试集',stats['test'],'测试覆盖率',f"{stats['test_coverage']:.2%}")
    if not pop.empty:
        st.subheader('最常被评分的电影')
        st.dataframe(pop[['title','rating_count','mean_rating']].head(15),hide_index=True,use_container_width=True)
