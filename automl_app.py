import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, io, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
    mean_squared_error, classification_report, confusion_matrix,
    silhouette_score, davies_bouldin_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy import stats

st.set_page_config(page_title="AutoML Pro", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background:#04060d;color:#d4d8f0;font-family:'Outfit',sans-serif}
[data-testid="stAppViewContainer"]{background:#04060d;
  background-image:radial-gradient(ellipse 60% 40% at 20% 0%,rgba(0,200,150,.08) 0%,transparent 60%),
                   radial-gradient(ellipse 50% 30% at 80% 100%,rgba(0,120,255,.08) 0%,transparent 60%)}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none}
.block-container{padding:2rem 2.5rem 4rem;max-width:1200px;margin:auto}
.hero{padding:3rem 0 1.5rem;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:2.5rem}
.hero-top{display:flex;align-items:center;gap:1.5rem;margin-bottom:1.2rem}
.hero-icon{font-size:3rem;background:linear-gradient(135deg,#00c896,#0078ff,#a855f7);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero h1{font-family:'Outfit',sans-serif;font-size:2.6rem;font-weight:800;
  background:linear-gradient(135deg,#fff 0%,#00c896 50%,#0078ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1}
.hero-sub{font-size:.9rem;color:#4b5563;font-weight:300}
.feature-pills{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.8rem}
.fpill{display:inline-block;padding:.3rem .8rem;border-radius:50px;font-size:.72rem;font-weight:600}
.fpill.green{background:rgba(0,200,150,.12);border:1px solid rgba(0,200,150,.3);color:#00c896}
.fpill.blue{background:rgba(0,120,255,.12);border:1px solid rgba(0,120,255,.3);color:#60a5fa}
.fpill.purple{background:rgba(168,85,247,.12);border:1px solid rgba(168,85,247,.3);color:#c084fc}
.fpill.yellow{background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.3);color:#fbbf24}
.fpill.red{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:#f87171}
.fpill.indigo{background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.3);color:#818cf8}
.fpill.teal{background:rgba(20,184,166,.12);border:1px solid rgba(20,184,166,.3);color:#2dd4bf}
.fpill.orange{background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.3);color:#fb923c}
.pipeline{display:flex;margin-bottom:2.5rem;background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.06);border-radius:16px;overflow:hidden}
.pipe-step{flex:1;padding:.8rem;text-align:center;border-right:1px solid rgba(255,255,255,.06)}
.pipe-step:last-child{border-right:none}
.pipe-step.active{background:rgba(0,200,150,.08)}
.pipe-step.done{background:rgba(0,200,150,.04)}
.pipe-num{font-family:'Space Mono',monospace;font-size:.58rem;color:#374151;margin-bottom:.2rem}
.pipe-step.active .pipe-num,.pipe-step.done .pipe-num{color:#00c896}
.pipe-label{font-size:.65rem;font-weight:600;color:#4b5563}
.pipe-step.active .pipe-label{color:#d4d8f0}
.pipe-step.done .pipe-label{color:#6b7280}
.card{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);
      border-radius:16px;padding:1.5rem;margin-bottom:1.5rem}
.card-title{font-family:'Space Mono',monospace;font-size:.68rem;color:#00c896;
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:1rem}
.section-sep{height:1px;background:linear-gradient(90deg,transparent,rgba(0,200,150,.25),transparent);margin:1.5rem 0}
.section-head{font-family:'Space Mono',monospace;font-size:.63rem;color:#6b7280;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem}
.metrics-row{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem}
.metric-box{flex:1;min-width:100px;background:rgba(0,200,150,.06);
  border:1px solid rgba(0,200,150,.2);border-radius:12px;padding:.9rem;text-align:center}
.metric-val{font-family:'Space Mono',monospace;font-size:1.4rem;font-weight:700;color:#00c896}
.metric-key{font-size:.68rem;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-top:.2rem}
.model-row{display:flex;align-items:center;gap:1rem;padding:.65rem .5rem;
  border-bottom:1px solid rgba(255,255,255,.04);border-radius:8px}
.model-row.best{background:rgba(0,200,150,.07);border:1px solid rgba(0,200,150,.2);
  border-radius:10px;margin-bottom:4px}
.model-name{flex:2;font-size:.85rem;font-weight:500;color:#d4d8f0}
.model-score{flex:1;font-family:'Space Mono',monospace;font-size:.8rem;color:#00c896;text-align:right}
.model-bar-wrap{flex:3;height:5px;background:rgba(255,255,255,.06);border-radius:10px;overflow:hidden}
.model-bar{height:100%;background:linear-gradient(90deg,#00c896,#0078ff);border-radius:10px}
.best-badge{background:linear-gradient(135deg,#00c896,#0078ff);color:#000;
  font-size:.58rem;font-weight:700;padding:.12rem .45rem;border-radius:50px}
.tuned-badge{background:rgba(251,191,36,.2);border:1px solid rgba(251,191,36,.4);
  color:#fbbf24;font-size:.58rem;font-weight:700;padding:.12rem .45rem;border-radius:50px}
.chip{display:inline-block;background:rgba(0,120,255,.1);border:1px solid rgba(0,120,255,.25);
  border-radius:50px;padding:.22rem .7rem;font-size:.75rem;color:#60a5fa;margin:.2rem}
.chip.green{background:rgba(0,200,150,.1);border-color:rgba(0,200,150,.25);color:#00c896}
.chip.red{background:rgba(239,68,68,.1);border-color:rgba(239,68,68,.25);color:#f87171}
.chip.purple{background:rgba(168,85,247,.1);border-color:rgba(168,85,247,.25);color:#c084fc}
.nlp-detect-box{background:rgba(168,85,247,.06);border:1px solid rgba(168,85,247,.25);
  border-radius:14px;padding:1.2rem 1.5rem;margin:1rem 0}
.nlp-detect-title{font-family:'Space Mono',monospace;font-size:.68rem;color:#c084fc;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem}
/* Business insight cards */
.biz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;margin-top:1rem}
.biz-card{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);
  border-radius:14px;padding:1.2rem;transition:border-color .3s}
.biz-card:hover{border-color:rgba(0,200,150,.3)}
.biz-icon{font-size:1.8rem;margin-bottom:.6rem}
.biz-title{font-family:'Space Mono',monospace;font-size:.7rem;color:#00c896;
  letter-spacing:.08em;text-transform:uppercase;margin-bottom:.4rem}
.biz-desc{font-size:.82rem;color:#6b7280;line-height:1.6}
.biz-tag{display:inline-block;background:rgba(0,200,150,.1);border:1px solid rgba(0,200,150,.2);
  border-radius:50px;padding:.15rem .6rem;font-size:.68rem;color:#00c896;margin-top:.6rem;margin-right:.3rem}
.insight-box{background:linear-gradient(135deg,rgba(0,200,150,.06),rgba(0,120,255,.06));
  border:1px solid rgba(0,200,150,.2);border-radius:16px;padding:1.5rem;margin-top:1rem}
.insight-title{font-family:'Space Mono',monospace;font-size:.72rem;color:#00c896;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem}
div[data-testid="stFileUploader"]{background:rgba(0,200,150,.03);
  border:2px dashed rgba(0,200,150,.25);border-radius:16px;padding:1rem}
.stButton>button{background:linear-gradient(135deg,#00c896,#0078ff);color:#000;
  font-family:'Outfit',sans-serif;font-weight:700;font-size:.92rem;border:none;
  border-radius:12px;padding:.6rem 1.8rem;width:100%;transition:all .25s;
  box-shadow:0 4px 20px rgba(0,200,150,.25)}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,200,150,.4)}
.stProgress>div>div{background:linear-gradient(90deg,#00c896,#0078ff)!important}
[data-testid="stExpander"]{background:rgba(255,255,255,.02)!important;
  border:1px solid rgba(255,255,255,.07)!important;border-radius:12px!important}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.fadein{animation:fadeIn .4s ease}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-top"><div class="hero-icon">⚡</div>
    <div><h1>AutoML Pro</h1><div class="hero-sub">Enterprise-grade automated machine learning for everyone</div></div>
  </div>
  <div class="feature-pills">
    <span class="fpill green">✓ Auto EDA</span>
    <span class="fpill red">✓ Outlier Detection</span>
    <span class="fpill teal">✓ Word Cloud</span>
    <span class="fpill blue">✓ ML + Clustering + DR</span>
    <span class="fpill yellow">✓ Hyperparameter Tuning</span>
    <span class="fpill purple">✓ SHAP Explainability</span>
    <span class="fpill orange">✓ Business Insights</span>
    <span class="fpill indigo">✓ PDF Report</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session defaults ───────────────────────────────────────────────────────
defs={'step':0,'df':None,'df_clean':None,'target':None,'results':None,'best_model':None,
      'problem_type':None,'nlp_mode':False,'text_col':None,'text_cols_detected':[],
      'df_proc':None,'scaler':None,'feat_cols':[],'X_test':None,'X_test_sc':None,
      'y_test':None,'le_target':None,'tfidf':None,'best_name':'','best_scaled':False,
      'metric_name':'Accuracy','sort_col':'Accuracy','cluster_results':[],'dr_results':[],
      'outlier_report':{},'tuning_results':{},'pdf_buf':None}
for k,v in defs.items():
    if k not in st.session_state: st.session_state[k]=v

step=st.session_state.step
pipe_labels=["Upload","Clean","EDA","Preprocess","Train","Results","Insights","Report"]
ph='<div class="pipeline">'
for i,lbl in enumerate(pipe_labels):
    cls="active" if i==step else ("done" if i<step else "")
    icon="✓" if i<step else str(i+1)
    ph+=f'<div class="pipe-step {cls}"><div class="pipe-num">{icon}</div><div class="pipe-label">{lbl}</div></div>'
ph+='</div>'
st.markdown(ph,unsafe_allow_html=True)

def nav(s): st.session_state.step=s; st.rerun()
def dark_fig(w=8,h=4):
    fig,ax=plt.subplots(figsize=(w,h),facecolor='#04060d'); ax.set_facecolor('#0d1117'); return fig,ax
def style_ax(ax):
    ax.tick_params(colors='#6b7280',labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#1f2937')

def add_bar_labels(ax, fmt='{:.0f}', color='#9ca3af', fontsize=7, horizontal=False):
    """Add value labels to bar charts"""
    for patch in ax.patches:
        if horizontal:
            w=patch.get_width()
            if w>0:
                ax.text(w+max(0.001,w*0.02), patch.get_y()+patch.get_height()/2,
                        fmt.format(w), va='center', ha='left', color=color, fontsize=fontsize)
        else:
            h=patch.get_height()
            if h>0:
                ax.text(patch.get_x()+patch.get_width()/2, h+max(0.001,h*0.02),
                        fmt.format(h), ha='center', va='bottom', color=color, fontsize=fontsize)

def get_business_insights(problem_type, best_name, best_score, feat_cols, target, df):
    """Generate business use-case recommendations based on the model results"""
    insights = []
    score_pct = f"{best_score*100:.1f}%" if best_score <= 1 else f"{best_score:.2f}"

    if problem_type == 'classification':
        n_classes = df[target].nunique() if target else 2
        insights = [
            {"icon":"🎯","title":"Customer Churn Prediction",
             "desc":f"Use this {best_name} model ({score_pct} accuracy) to predict which customers are likely to leave. Enables proactive retention campaigns saving thousands in acquisition costs.",
             "tags":["CRM","Retention","Marketing"]},
            {"icon":"🔍","title":"Fraud Detection",
             "desc":f"Deploy as a real-time fraud screening system. Flag suspicious transactions automatically with {score_pct} accuracy, reducing financial losses.",
             "tags":["FinTech","Risk","Security"]},
            {"icon":"🏥","title":"Medical Diagnosis Support",
             "desc":f"Assist doctors in early diagnosis by classifying patient symptoms or test results into risk categories with {score_pct} accuracy.",
             "tags":["Healthcare","Diagnostics","AI"]},
            {"icon":"📧","title":"Email / Spam Filtering",
             "desc":f"Automatically classify incoming messages by priority, category or spam status — keeping inboxes clean at scale.",
             "tags":["Productivity","SaaS","Automation"]},
            {"icon":"🛒","title":"Product Recommendation",
             "desc":f"Classify customer segments and serve personalized product categories, boosting conversion rates and average order value.",
             "tags":["E-commerce","Personalization","Revenue"]},
            {"icon":"📊","title":"Lead Scoring",
             "desc":f"Score incoming sales leads as Hot / Warm / Cold automatically, letting your sales team focus on the highest-value prospects.",
             "tags":["Sales","B2B","CRM"]},
        ]
    elif problem_type == 'regression':
        insights = [
            {"icon":"🏠","title":"Property Price Estimation",
             "desc":f"Predict real estate prices based on features like location, size and amenities. Powers automated valuation tools for banks and agents.",
             "tags":["Real Estate","FinTech","Valuation"]},
            {"icon":"📈","title":"Sales Forecasting",
             "desc":f"Forecast future sales revenue with {score_pct} R² accuracy, enabling better inventory planning and budget allocation.",
             "tags":["Retail","Finance","Planning"]},
            {"icon":"⚡","title":"Energy Demand Prediction",
             "desc":f"Predict electricity or gas consumption to optimize grid management and reduce waste in utility companies.",
             "tags":["Energy","IoT","Sustainability"]},
            {"icon":"💰","title":"Dynamic Pricing",
             "desc":f"Predict optimal price points based on demand, competition and seasonality — maximize revenue automatically.",
             "tags":["E-commerce","Strategy","Revenue"]},
            {"icon":"🚗","title":"Vehicle Maintenance Cost",
             "desc":f"Predict maintenance and repair costs for fleet management, reducing downtime and surprise expenses.",
             "tags":["Logistics","Fleet","Operations"]},
            {"icon":"👥","title":"Employee Attrition Risk Score",
             "desc":f"Predict employee tenure or satisfaction scores, helping HR teams intervene before key staff leave.",
             "tags":["HR","Retention","Culture"]},
        ]
    else:  # clustering
        insights = [
            {"icon":"👥","title":"Customer Segmentation",
             "desc":f"Group customers into distinct segments based on behaviour, demographics or purchase history — enabling targeted marketing for each group.",
             "tags":["Marketing","CRM","Personalization"]},
            {"icon":"🗺","title":"Market Basket Analysis",
             "desc":f"Identify natural product groupings that customers buy together — power recommendation engines and store layouts.",
             "tags":["Retail","E-commerce","Strategy"]},
            {"icon":"🔬","title":"Anomaly & Fraud Detection",
             "desc":f"Isolate unusual data points that don't fit any cluster — potential fraud, system errors or rare medical cases.",
             "tags":["Security","FinTech","QA"]},
            {"icon":"🏭","title":"Predictive Maintenance Grouping",
             "desc":f"Group machines by usage patterns and failure signatures — schedule preventative maintenance before breakdowns occur.",
             "tags":["Manufacturing","IoT","Operations"]},
            {"icon":"📰","title":"Content & Topic Discovery",
             "desc":f"Automatically group articles, reviews or social media posts by topic — power content recommendation without labels.",
             "tags":["Media","NLP","Publishing"]},
            {"icon":"🧬","title":"Patient Cohort Analysis",
             "desc":f"Group patients by symptom profiles or treatment responses to discover new disease subtypes or treatment pathways.",
             "tags":["Healthcare","Research","Biotech"]},
        ]
    return insights

# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════
if step==0:
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 1 — Upload Dataset</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload CSV",type=["csv"],label_visibility="collapsed")
    if uploaded:
        df=pd.read_csv(uploaded)
        st.session_state.df=df; st.session_state.df_clean=df.copy()
        st.markdown(f"""<div class="metrics-row">
          <div class="metric-box"><div class="metric-val">{df.shape[0]:,}</div><div class="metric-key">Rows</div></div>
          <div class="metric-box"><div class="metric-val">{df.shape[1]}</div><div class="metric-key">Columns</div></div>
          <div class="metric-box"><div class="metric-val">{df.isnull().sum().sum()}</div><div class="metric-key">Missing</div></div>
          <div class="metric-box"><div class="metric-val">{df.dtypes[df.dtypes=='object'].count()}</div><div class="metric-key">Categorical</div></div>
        </div>""",unsafe_allow_html=True)
        st.dataframe(df.head(),use_container_width=True)
        target=st.selectbox("🎯 Select Target Column",["-- No target (Clustering) --"]+df.columns.tolist())
        st.session_state.target=None if target.startswith("--") else target
        if st.button("Continue →"): nav(1)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — CLEAN + OUTLIERS
# ══════════════════════════════════════════════════════════════════════════
elif step==1:
    df=st.session_state.df; target=st.session_state.target
    num_cols=df.select_dtypes(include=np.number).columns.tolist()
    feat_num=[c for c in num_cols if c!=target]
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 2 — Data Cleaning & Outlier Detection</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-head">🧹 Missing Value Analysis</div>',unsafe_allow_html=True)
    miss=df.isnull().sum(); miss=miss[miss>0]
    if len(miss):
        st.dataframe(pd.DataFrame({'Column':miss.index,'Missing':miss.values,
            'Pct':(miss.values/len(df)*100).round(1),
            'Suggestion':['Drop column' if p>50 else 'Fill median' if df[c].dtype!=object else 'Fill mode'
                          for c,p in zip(miss.index,miss.values/len(df)*100)]}),use_container_width=True)
    else: st.success("✅ No missing values!")

    outlier_report={}
    if feat_num:
        st.markdown('<div class="section-sep"></div><div class="section-head">🔴 Outlier Detection</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Z-Score (|z|>3)**")
            z_out={col:int((np.abs(stats.zscore(df[col].dropna()))>3).sum()) for col in feat_num[:8]}
            z_out={k:v for k,v in z_out.items() if v>0}
            if z_out:
                st.dataframe(pd.DataFrame({'Col':list(z_out.keys()),'Count':list(z_out.values()),
                    'Pct':[round(v/len(df)*100,1) for v in z_out.values()]}),use_container_width=True)
                outlier_report['zscore']=z_out
            else: st.success("✅ No Z-score outliers")
        with c2:
            st.markdown("**IQR Method**")
            iqr_out={}
            for col in feat_num[:8]:
                Q1,Q3=df[col].quantile(.25),df[col].quantile(.75); IQR=Q3-Q1
                n=int(((df[col]<Q1-1.5*IQR)|(df[col]>Q3+1.5*IQR)).sum())
                if n>0: iqr_out[col]=n
            if iqr_out:
                st.dataframe(pd.DataFrame({'Col':list(iqr_out.keys()),'Count':list(iqr_out.values())}),use_container_width=True)
                outlier_report['iqr']=iqr_out
            else: st.success("✅ No IQR outliers")

        if len(feat_num)>=2:
            try:
                iso=IsolationForest(contamination=0.05,random_state=42)
                X_iso=df[feat_num].fillna(df[feat_num].median())
                preds=iso.fit_predict(X_iso); n_anom=int((preds==-1).sum())
                st.markdown(f'<span class="chip red">🌲 Isolation Forest: {n_anom} anomalies ({round(n_anom/len(df)*100,1)}%)</span>',unsafe_allow_html=True)
                outlier_report['isolation_forest']=n_anom
                top_cols=(list(z_out.keys()) if z_out else feat_num)[:4]
                fig,axes=plt.subplots(1,len(top_cols),figsize=(min(12,len(top_cols)*3),3),facecolor='#04060d')
                if len(top_cols)==1: axes=[axes]
                for ax,col in zip(axes,top_cols):
                    ax.set_facecolor('#0d1117')
                    bp=ax.boxplot(df[col].dropna(),patch_artist=True,
                        boxprops=dict(facecolor='rgba(0,120,255,0.15)',color='#0078ff'),
                        medianprops=dict(color='#00c896',linewidth=2),
                        whiskerprops=dict(color='#374151'),capprops=dict(color='#374151'),
                        flierprops=dict(marker='o',markerfacecolor='#f87171',markersize=4,alpha=.7))
                    ax.set_title(col[:12],color='#9ca3af',fontsize=7); style_ax(ax)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            except: pass

        remove_out=st.toggle("Remove Z-score outliers (|z|>3) before training",value=False)
        if remove_out:
            df_cl=df.copy()
            for col in feat_num:
                z=np.abs(stats.zscore(df_cl[col].fillna(df_cl[col].median())))
                df_cl=df_cl[z<=3]
            st.session_state.df_clean=df_cl
            st.markdown(f'<span class="chip green">✓ {len(df)-len(df_cl)} rows removed. New size: {len(df_cl)}</span>',unsafe_allow_html=True)
        else: st.session_state.df_clean=df.copy()

    st.session_state.outlier_report=outlier_report
    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(0)
    with cb:
        if st.button("Continue to EDA →"): nav(2)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA + WORD CLOUD
# ══════════════════════════════════════════════════════════════════════════
elif step==2:
    df=st.session_state.df_clean; target=st.session_state.target
    num_cols=df.select_dtypes(include=np.number).columns.tolist()
    cat_cols=df.select_dtypes(include='object').columns.tolist()
    txt_det=[c for c in cat_cols if c!=target and df[c].dropna().apply(lambda x:len(str(x).split())).mean()>5]
    st.session_state.text_cols_detected=txt_det
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 3 — EDA & Word Cloud</div>',unsafe_allow_html=True)

    if txt_det:
        st.markdown(f'<div class="nlp-detect-box"><div class="nlp-detect-title">🔍 NLP Text Columns Detected</div>'
                    f'<div style="font-size:.85rem;color:#9ca3af;">{len(txt_det)} long-text column(s) found — Word Cloud available below.</div></div>',unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown("**📊 Numeric Summary**")
        if num_cols: st.dataframe(df[num_cols].describe().round(2),use_container_width=True)
    with c2:
        st.markdown("**❓ Missing Values**")
        miss=df.isnull().sum(); miss=miss[miss>0]
        if len(miss): st.dataframe(pd.DataFrame({'Col':miss.index,'#':miss.values,'%':(miss.values/len(df)*100).round(1)}),use_container_width=True)
        else: st.success("✅ None!")

    # ── Target Distribution with values ─────────────────────────────────
    if target:
        st.markdown("**🎯 Target Distribution**")
        fig,ax=dark_fig(8,3)
        if df[target].dtype=='object' or df[target].nunique()<=15:
            counts=df[target].value_counts()
            palette=['#00c896','#0078ff','#a855f7','#f59e0b','#ef4444']
            bars=ax.bar(counts.index.astype(str),counts.values,color=palette[:len(counts)])
            # Add values on bars
            for bar,val in zip(bars,counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(1,bar.get_height()*0.02),
                        f'{val:,}', ha='center', va='bottom', color='#9ca3af', fontsize=8, fontweight='bold')
            # Add percentages
            total=counts.sum()
            for bar,val in zip(bars,counts.values):
                pct=val/total*100
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                        f'{pct:.1f}%', ha='center', va='center', color='white', fontsize=7, alpha=0.8)
        else:
            ax.hist(df[target].dropna(),bins=30,color='#00c896',edgecolor='#04060d',linewidth=.5)
            ax.set_xlabel(target,color='#6b7280',fontsize=9)
            mean_val=df[target].mean(); ax.axvline(mean_val,color='#fbbf24',linestyle='--',linewidth=1.5)
            ax.text(mean_val,ax.get_ylim()[1]*0.95,f'Mean: {mean_val:.2f}',color='#fbbf24',fontsize=8,ha='center')
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Correlation heatmap ──────────────────────────────────────────────
    if len(num_cols)>1:
        st.markdown("**🔥 Correlation Heatmap**")
        fig2,ax2=dark_fig(10,4)
        corr=df[num_cols].corr(); mask=np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr,mask=mask,cmap=sns.diverging_palette(150,220,as_cmap=True),
                    annot=True,fmt='.2f',linewidths=.5,ax=ax2,annot_kws={'size':7},cbar_kws={'shrink':.8})
        ax2.tick_params(colors='#6b7280',labelsize=7)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Feature distributions with mean line ────────────────────────────
    plot_cols=[c for c in num_cols if c!=target][:6]
    if plot_cols:
        st.markdown("**📈 Feature Distributions**")
        fig3,axes=plt.subplots(1,len(plot_cols),figsize=(min(14,len(plot_cols)*2.5),3),facecolor='#04060d')
        if len(plot_cols)==1: axes=[axes]
        for ax3,col in zip(axes,plot_cols):
            ax3.set_facecolor('#0d1117')
            ax3.hist(df[col].dropna(),bins=20,color='#0078ff',edgecolor='#04060d',linewidth=.3,alpha=.8)
            mean_v=df[col].mean()
            ax3.axvline(mean_v,color='#00c896',linestyle='--',linewidth=1.2)
            ax3.text(mean_v,ax3.get_ylim()[1]*0.9,f'{mean_v:.1f}',color='#00c896',fontsize=6,ha='center')
            ax3.set_title(col[:12],color='#9ca3af',fontsize=7)
            ax3.tick_params(colors='#4b5563',labelsize=6)
            for sp in ax3.spines.values(): sp.set_color('#1f2937')
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    # ── Categorical value counts with labels ─────────────────────────────
    short_cats=[c for c in cat_cols if c!=target and c not in txt_det and df[c].nunique()<=10][:3]
    if short_cats:
        st.markdown("**📦 Categorical Columns**")
        fig4,axes4=plt.subplots(1,len(short_cats),figsize=(min(12,len(short_cats)*4),3),facecolor='#04060d')
        if len(short_cats)==1: axes4=[axes4]
        for ax4,col in zip(axes4,short_cats):
            ax4.set_facecolor('#0d1117')
            counts=df[col].value_counts()[:8]
            bars=ax4.barh(counts.index.astype(str)[::-1],counts.values[::-1],color='#a855f7',alpha=.8)
            for bar,val in zip(bars,counts.values[::-1]):
                ax4.text(val+max(0.1,val*0.02),bar.get_y()+bar.get_height()/2,
                         f'{val:,}',va='center',ha='left',color='#9ca3af',fontsize=7)
            ax4.set_title(col[:15],color='#9ca3af',fontsize=8)
            ax4.tick_params(colors='#4b5563',labelsize=7)
            for sp in ax4.spines.values(): sp.set_color('#1f2937')
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    # ── Word Cloud ───────────────────────────────────────────────────────
    if txt_det:
        st.markdown('<div class="section-sep"></div>',unsafe_allow_html=True)
        st.markdown("**☁️ Word Cloud**")
        wc_col=st.selectbox("Select text column for Word Cloud",txt_det,key='wc_col')
        try:
            from wordcloud import WordCloud
            text_data=' '.join(df[wc_col].dropna().astype(str).tolist())
            wc=WordCloud(width=900,height=400,background_color='#0d1117',
                         colormap='cool',max_words=200,
                         stopwords=None,collocations=False,
                         contour_width=1,contour_color='#00c896').generate(text_data)
            fig_wc,ax_wc=plt.subplots(figsize=(10,4),facecolor='#0d1117')
            ax_wc.set_facecolor('#0d1117')
            ax_wc.imshow(wc,interpolation='bilinear')
            ax_wc.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc); plt.close()

            # Top words bar chart with values
            st.markdown("**🔤 Top Words Frequency**")
            from collections import Counter
            import re
            words=re.findall(r'\b[a-zA-Z]{3,}\b',text_data.lower())
            stopwords_basic={'the','and','for','are','but','not','you','all','can','had','her','was',
                             'one','our','out','day','get','has','him','his','how','its','may','new',
                             'now','old','see','two','way','who','boy','did','man','end','put','say',
                             'she','too','use','with','that','this','from','they','been','have','more',
                             'will','than','then','when','your','also','into','over','such','only',
                             'come','some','time','very','what','which','their','there','these','those'}
            words=[w for w in words if w not in stopwords_basic]
            top_words=Counter(words).most_common(15)
            if top_words:
                words_list,counts_list=zip(*top_words)
                fig_tw,ax_tw=dark_fig(9,3)
                bars_tw=ax_tw.bar(words_list,counts_list,
                                  color=['#00c896' if i==0 else '#0078ff' if i<3 else '#a855f7' if i<6 else '#374151'
                                         for i in range(len(words_list))])
                for bar,val in zip(bars_tw,counts_list):
                    ax_tw.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(0.5,bar.get_height()*0.02),
                               f'{val:,}',ha='center',va='bottom',color='#9ca3af',fontsize=7,fontweight='bold')
                ax_tw.set_ylabel('Frequency',color='#6b7280',fontsize=9)
                ax_tw.tick_params(axis='x',rotation=30,colors='#6b7280',labelsize=8)
                ax_tw.tick_params(axis='y',colors='#6b7280',labelsize=8)
                for sp in ax_tw.spines.values(): sp.set_color('#1f2937')
                plt.tight_layout(); st.pyplot(fig_tw); plt.close()
        except ImportError:
            st.info("Add `wordcloud` to requirements.txt to enable Word Cloud.")

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(1)
    with cb:
        if st.button("Continue to Preprocessing →"): nav(3)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════
elif step==3:
    df=st.session_state.df_clean; target=st.session_state.target
    txt_det=st.session_state.text_cols_detected
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 4 — Preprocessing</div>',unsafe_allow_html=True)

    nlp_mode=False; text_col=None
    if txt_det:
        st.markdown(f'<div class="nlp-detect-box"><div class="nlp-detect-title">🧠 NLP Mode</div>'
                    f'<div style="font-size:.85rem;color:#9ca3af;">Enable TF-IDF on text column.</div></div>',unsafe_allow_html=True)
        nlp_mode=st.toggle("Enable NLP Mode (TF-IDF)",value=True)
        if nlp_mode: text_col=st.selectbox("Text column",txt_det)
    st.session_state.nlp_mode=nlp_mode; st.session_state.text_col=text_col

    with st.spinner("Preprocessing..."):
        df_proc=df.copy()
        high_miss=[c for c in df_proc.columns if df_proc[c].isnull().mean()>0.5 and c!=target]
        if high_miss: df_proc.drop(columns=high_miss,inplace=True)
        if target:
            if df_proc[target].dtype=='object' or df_proc[target].nunique()<=20:
                problem_type='classification'; le_t=LabelEncoder(); df_proc[target]=le_t.fit_transform(df_proc[target].astype(str))
            else: problem_type='regression'; le_t=None
        else: problem_type='clustering'; le_t=None
        st.session_state.problem_type=problem_type; st.session_state.le_target=le_t
        cat_cols=df_proc.select_dtypes(include='object').columns.tolist()
        non_txt=[c for c in cat_cols if c!=target and c!=text_col]
        for col in non_txt:
            le=LabelEncoder(); df_proc[col]=le.fit_transform(df_proc[col].astype(str))
        num_cols=[c for c in df_proc.select_dtypes(include=np.number).columns if c!=target]
        if df_proc[num_cols].isnull().sum().sum()>0:
            imp=SimpleImputer(strategy='median'); df_proc[num_cols]=imp.fit_transform(df_proc[num_cols])
        tfidf=None; tfidf_cols=[]
        if nlp_mode and text_col and text_col in df_proc.columns:
            tfidf=TfidfVectorizer(max_features=300,stop_words='english')
            tm=tfidf.fit_transform(df_proc[text_col].fillna('').astype(str))
            tdf=pd.DataFrame(tm.toarray(),columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()],index=df_proc.index)
            df_proc=pd.concat([df_proc.drop(columns=[text_col]),tdf],axis=1); tfidf_cols=list(tdf.columns)
        st.session_state.tfidf=tfidf; st.session_state.df_proc=df_proc

    chips=[f'<span class="chip green">✓ {problem_type.title()}</span>',
           f'<span class="chip green">✓ {len(non_txt)} cats encoded</span>',
           f'<span class="chip green">✓ Missing imputed</span>']
    if high_miss: chips.append(f'<span class="chip red">✗ {len(high_miss)} cols dropped</span>')
    if nlp_mode and text_col: chips.append(f'<span class="chip purple">🧠 TF-IDF {len(tfidf_cols)} feats</span>')
    st.markdown("".join(chips),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.dataframe(df_proc.head(),use_container_width=True)
    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(2)
    with cb:
        if st.button("Continue to Training →"): nav(4)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN + TUNE
# ══════════════════════════════════════════════════════════════════════════
elif step==4:
    df_proc=st.session_state.df_proc; target=st.session_state.target
    problem_type=st.session_state.problem_type
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 5 — Training + Tuning</div>',unsafe_allow_html=True)

    feat_cols=[c for c in df_proc.columns if c!=target]
    X=df_proc[feat_cols].values; y=df_proc[target].values if target else None
    scaler=StandardScaler(); X_sc=scaler.fit_transform(X)
    st.session_state.scaler=scaler; st.session_state.feat_cols=feat_cols

    if problem_type!='clustering':
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        Xtr_sc=scaler.transform(X_train); Xte_sc=scaler.transform(X_test)
        st.session_state.X_test=X_test; st.session_state.X_test_sc=Xte_sc; st.session_state.y_test=y_test

    results=[]; tuning_results={}
    enable_tuning=st.toggle("🔧 Enable Hyperparameter Tuning (GridSearchCV)",value=True)

    if problem_type=='classification':
        ml_models={"Random Forest":(RandomForestClassifier(n_estimators=100,random_state=42),False),
                   "Gradient Boosting":(GradientBoostingClassifier(random_state=42),False),
                   "Logistic Regression":(LogisticRegression(max_iter=1000),True),
                   "SVM":(SVC(),True),"KNN":(KNeighborsClassifier(),True),
                   "Decision Tree":(DecisionTreeClassifier(random_state=42),False)}
        param_grids={"Random Forest":{'n_estimators':[50,100],'max_depth':[None,10]},
                     "Logistic Regression":{'C':[0.1,1,10]}}
        metric_name="Accuracy"; sort_col="Accuracy"
    elif problem_type=='regression':
        ml_models={"Random Forest":(RandomForestRegressor(n_estimators=100,random_state=42),False),
                   "Gradient Boosting":(GradientBoostingRegressor(random_state=42),False),
                   "Linear Regression":(LinearRegression(),True),
                   "Ridge":(Ridge(),True),"SVR":(SVR(),True),
                   "KNN":(KNeighborsRegressor(),True),
                   "Decision Tree":(DecisionTreeRegressor(random_state=42),False)}
        param_grids={"Random Forest":{'n_estimators':[50,100],'max_depth':[None,10]},
                     "Ridge":{'alpha':[0.1,1,10,100]}}
        metric_name="R² Score"; sort_col="R² Score"
    else:
        ml_models={}; param_grids={}; metric_name="Silhouette"; sort_col="Silhouette"

    st.session_state.metric_name=metric_name; st.session_state.sort_col=sort_col
    total=len(ml_models)+4+3; progress=st.progress(0); status=st.empty(); done=0

    for name,(model,use_sc) in ml_models.items():
        status.markdown(f'<div style="font-size:.85rem;color:#6b7280;">🤖 {name}...</div>',unsafe_allow_html=True)
        Xtr=Xtr_sc if use_sc else X_train; Xte=Xte_sc if use_sc else X_test
        if enable_tuning and name in param_grids:
            status.markdown(f'<div style="font-size:.85rem;color:#fbbf24;">🔧 Tuning {name}...</div>',unsafe_allow_html=True)
            gs=GridSearchCV(model,param_grids[name],cv=3,scoring='accuracy' if problem_type=='classification' else 'r2',n_jobs=-1)
            gs.fit(Xtr,y_train); model=gs.best_estimator_
            tuning_results[name]={'best_params':gs.best_params_,'best_score':round(gs.best_score_,4)}
        model.fit(Xtr,y_train); yp=model.predict(Xte)
        is_tuned=name in tuning_results
        if problem_type=='classification':
            results.append({'Model':name,'Tuned':'✓' if is_tuned else '—',
                'Accuracy':round(accuracy_score(y_test,yp),4),
                'F1':round(f1_score(y_test,yp,average='weighted',zero_division=0),4),
                '_model':model,'_scaled':use_sc,'_tuned':is_tuned})
        else:
            results.append({'Model':name,'Tuned':'✓' if is_tuned else '—',
                'R² Score':round(r2_score(y_test,yp),4),
                'RMSE':round(np.sqrt(mean_squared_error(y_test,yp)),4),
                '_model':model,'_scaled':use_sc,'_tuned':is_tuned})
        done+=1; progress.progress(done/total)

    st.session_state.tuning_results=tuning_results

    status.markdown('<div style="font-size:.85rem;color:#fbbf24;">🔵 Clustering...</div>',unsafe_allow_html=True)
    n_cl=min(5,max(2,len(X_sc)//50))
    cluster_models={"K-Means":KMeans(n_clusters=n_cl,random_state=42,n_init=10),
                    "Agglomerative":AgglomerativeClustering(n_clusters=n_cl),
                    "Gaussian Mixture":GaussianMixture(n_components=n_cl,random_state=42),
                    "DBSCAN":DBSCAN(eps=0.5,min_samples=5)}
    cluster_results=[]
    for cname,cmodel in cluster_models.items():
        try:
            lbl=cmodel.fit_predict(X_sc); n_u=len(set(lbl)-{-1})
            sil=round(silhouette_score(X_sc,lbl),4) if n_u>=2 else -1
            db=round(davies_bouldin_score(X_sc,lbl),4) if n_u>=2 else 999
            cluster_results.append({'Model':cname,'Silhouette':sil,'Davies-Bouldin':db,'Clusters':n_u,'_labels':lbl})
        except: pass
        done+=1; progress.progress(done/total)
    st.session_state.cluster_results=cluster_results

    status.markdown('<div style="font-size:.85rem;color:#818cf8;">🔷 DR...</div>',unsafe_allow_html=True)
    dr_results=[]; n_comp=min(2,X_sc.shape[1],X_sc.shape[0])
    try:
        pca2=PCA(n_components=n_comp); X_pca=pca2.fit_transform(X_sc)
        pca_f=PCA(n_components=min(20,X_sc.shape[1],X_sc.shape[0])); pca_f.fit(X_sc)
        dr_results.append({'Method':'PCA','Var(2D)':round(sum(pca_f.explained_variance_ratio_[:2])*100,2),'_obj':pca2,'_X2d':X_pca,'_pca_full':pca_f})
    except: pass
    done+=1; progress.progress(done/total)
    try:
        svd=TruncatedSVD(n_components=n_comp,random_state=42); X_svd=svd.fit_transform(X_sc)
        dr_results.append({'Method':'Truncated SVD','Var(2D)':round(sum(svd.explained_variance_ratio_)*100,2),'_obj':svd,'_X2d':X_svd})
    except: pass
    done+=1; progress.progress(done/total)
    try:
        sn=min(1000,len(X_sc)); idx=np.random.choice(len(X_sc),sn,replace=False)
        tsne=TSNE(n_components=2,random_state=42,perplexity=min(30,sn-1),n_iter=300)
        X_tsne=tsne.fit_transform(X_sc[idx])
        dr_results.append({'Method':'t-SNE','Var(2D)':'N/A','_obj':tsne,'_X2d':X_tsne,'_idx':idx})
    except: pass
    done+=1; progress.progress(done/total)
    st.session_state.dr_results=dr_results

    progress.progress(1.0); status.empty()
    if problem_type!='clustering' and results:
        rdf=pd.DataFrame(results).sort_values(sort_col,ascending=False).reset_index(drop=True)
        st.session_state.results=rdf; best=rdf.iloc[0]
        st.session_state.best_model=best['_model']; st.session_state.best_name=best['Model']
        st.session_state.best_scaled=best['_scaled']

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(3)
    with cb:
        if st.button("View Results →"): nav(5)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — RESULTS + SHAP
# ══════════════════════════════════════════════════════════════════════════
elif step==5:
    results_df=st.session_state.results; cluster_res=st.session_state.cluster_results
    dr_res=st.session_state.dr_results; feat_cols=st.session_state.feat_cols
    sort_col=st.session_state.sort_col; metric_name=st.session_state.metric_name
    problem_type=st.session_state.problem_type; X_test=st.session_state.X_test
    X_test_sc=st.session_state.X_test_sc; y_test=st.session_state.y_test
    scaler=st.session_state.scaler; df_proc=st.session_state.df_proc
    target=st.session_state.target; tuning_results=st.session_state.tuning_results
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 6 — Results & Explainability</div>',unsafe_allow_html=True)

    best_score=results_df.iloc[0][sort_col] if results_df is not None and len(results_df) else 0
    best_name=st.session_state.best_name or '—'
    st.markdown(f"""<div class="metrics-row">
      <div class="metric-box"><div class="metric-val">{len(results_df) if results_df is not None else 0}</div><div class="metric-key">ML Models</div></div>
      <div class="metric-box"><div class="metric-val">{best_score:.2%}</div><div class="metric-key">Best {metric_name}</div></div>
      <div class="metric-box"><div class="metric-val">{len(tuning_results)}</div><div class="metric-key">Tuned</div></div>
      <div class="metric-box"><div class="metric-val">{len(cluster_res)}</div><div class="metric-key">Clusters</div></div>
    </div>""",unsafe_allow_html=True)

    if results_df is not None and len(results_df):
        st.markdown('<div class="section-head">🤖 ML Comparison</div>',unsafe_allow_html=True)
        # Model comparison bar chart with values
        fig_cmp,ax_cmp=dark_fig(9,3)
        model_names=[r['Model'] for _,r in results_df.iterrows()]
        model_scores=[r[sort_col] for _,r in results_df.iterrows()]
        colors_cmp=['#00c896' if i==0 else '#0078ff' if i==1 else '#a855f7' if i==2 else '#374151' for i in range(len(model_names))]
        bars_cmp=ax_cmp.barh(model_names[::-1],model_scores[::-1],color=colors_cmp[::-1],height=0.6)
        for bar,val,name in zip(bars_cmp,model_scores[::-1],model_names[::-1]):
            ax_cmp.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.4f}',
                        va='center',ha='left',color='#d4d8f0',fontsize=8,fontweight='bold')
        ax_cmp.set_xlabel(sort_col,color='#6b7280',fontsize=9)
        ax_cmp.set_xlim(0,min(1.15,max(model_scores)*1.2))
        style_ax(ax_cmp); plt.tight_layout(); st.pyplot(fig_cmp); plt.close()

        if tuning_results:
            with st.expander("🔧 Tuning Details"):
                for mn,tr in tuning_results.items():
                    st.markdown(f"**{mn}** — CV:{tr['best_score']} — `{tr['best_params']}`")

        best_model=st.session_state.best_model
        Xte=X_test_sc if st.session_state.best_scaled else X_test
        yp=best_model.predict(Xte)

        if problem_type=='classification':
            with st.expander("📋 Classification Report"):
                le_t=st.session_state.le_target; tn=le_t.classes_.astype(str) if le_t else None
                rpt=classification_report(y_test,yp,target_names=tn,output_dict=True)
                st.dataframe(pd.DataFrame(rpt).transpose().round(3),use_container_width=True)
            with st.expander("🔲 Confusion Matrix"):
                cm=confusion_matrix(y_test,yp); fig,ax=dark_fig(6,4)
                sns.heatmap(cm,annot=True,fmt='d',cmap='Greens',ax=ax,linewidths=.5,annot_kws={'size':10,'weight':'bold'})
                ax.set_xlabel('Predicted',color='#6b7280',fontsize=9); ax.set_ylabel('Actual',color='#6b7280',fontsize=9)
                ax.tick_params(colors='#6b7280',labelsize=8)
                # Add accuracy on title
                acc=accuracy_score(y_test,yp)
                ax.set_title(f'Accuracy: {acc:.2%}',color='#00c896',fontsize=10)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            with st.expander("📈 Actual vs Predicted"):
                fig,ax=dark_fig(7,4)
                ax.scatter(y_test,yp,alpha=.5,color='#00c896',s=15)
                mn2,mx2=min(y_test.min(),yp.min()),max(y_test.max(),yp.max())
                ax.plot([mn2,mx2],[mn2,mx2],'--',color='#0078ff',linewidth=1.5,label='Perfect fit')
                r2=r2_score(y_test,yp); rmse=np.sqrt(mean_squared_error(y_test,yp))
                ax.set_title(f'R²: {r2:.4f}  |  RMSE: {rmse:.4f}',color='#00c896',fontsize=9)
                ax.set_xlabel('Actual',color='#6b7280',fontsize=9); ax.set_ylabel('Predicted',color='#6b7280',fontsize=9)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        if hasattr(best_model,'feature_importances_'):
            with st.expander("🔍 Feature Importance (with values)"):
                fi=pd.Series(best_model.feature_importances_,index=feat_cols).sort_values(ascending=False)[:15]
                fig2,ax2=dark_fig(8,5)
                colors2=['#00c896' if i==0 else '#0078ff' if i<3 else '#374151' for i in range(len(fi))]
                bars2=ax2.barh(fi.index[::-1],fi.values[::-1],color=colors2[::-1])
                for bar,val in zip(bars2,fi.values[::-1]):
                    ax2.text(val+0.001,bar.get_y()+bar.get_height()/2,f'{val:.4f}',
                             va='center',ha='left',color='#9ca3af',fontsize=7)
                ax2.set_xlabel('Importance',color='#6b7280',fontsize=9)
                style_ax(ax2); plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown('<div class="section-sep"></div><div class="section-head">🔮 SHAP Explainability</div>',unsafe_allow_html=True)
        try:
            import shap
            with st.expander("🔮 SHAP Feature Impact"):
                with st.spinner("Computing SHAP values..."):
                    sample=min(100,len(Xte)); Xs=Xte[:sample]
                    if hasattr(best_model,'feature_importances_'):
                        explainer=shap.TreeExplainer(best_model); shap_vals=explainer.shap_values(Xs)
                        if isinstance(shap_vals,list): shap_vals=shap_vals[0]
                    else:
                        explainer=shap.KernelExplainer(best_model.predict,shap.sample(Xs,30))
                        shap_vals=explainer.shap_values(Xs,nsamples=50)
                    shap_mean=np.abs(shap_vals).mean(axis=0)
                    top_idx=np.argsort(shap_mean)[-15:]; top_feats=[feat_cols[i] for i in top_idx]; top_vals=shap_mean[top_idx]
                    fig3,ax3=dark_fig(8,5); ax3.set_facecolor('#0d1117')
                    bars3=ax3.barh(top_feats,top_vals,color=['#a855f7' if v>np.median(top_vals) else '#0078ff' for v in top_vals])
                    for bar,val in zip(bars3,top_vals):
                        ax3.text(val+0.001,bar.get_y()+bar.get_height()/2,f'{val:.4f}',
                                 va='center',ha='left',color='#9ca3af',fontsize=7)
                    ax3.set_xlabel('Mean |SHAP|',color='#6b7280',fontsize=9)
                    ax3.set_title('Feature Impact on Predictions',color='#9ca3af',fontsize=10)
                    style_ax(ax3); plt.tight_layout(); st.pyplot(fig3); plt.close()
        except ImportError:
            st.info("Add `shap` to requirements.txt to enable SHAP.")

    # ── Clustering ───────────────────────────────────────────────────────
    if cluster_res:
        st.markdown('<div class="section-sep"></div><div class="section-head">🔵 Clustering</div>',unsafe_allow_html=True)
        valid_cr=[r for r in cluster_res if r['Silhouette']>-1]
        if valid_cr:
            best_cr=max(valid_cr,key=lambda x:x['Silhouette'])
            # Cluster comparison bar chart with values
            fig_cl,ax_cl=dark_fig(8,2.5)
            cl_names=[r['Model'] for r in valid_cr]; cl_sils=[r['Silhouette'] for r in valid_cr]
            bars_cl=ax_cl.bar(cl_names,cl_sils,color=['#00c896' if r['Model']==best_cr['Model'] else '#374151' for r in valid_cr])
            for bar,val in zip(bars_cl,cl_sils):
                ax_cl.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{val:.4f}',
                           ha='center',va='bottom',color='#9ca3af',fontsize=8,fontweight='bold')
            ax_cl.set_ylabel('Silhouette Score',color='#6b7280',fontsize=9)
            style_ax(ax_cl); plt.tight_layout(); st.pyplot(fig_cl); plt.close()

            with st.expander(f"🗺 Cluster Viz — {best_cr['Model']}"):
                Xall=scaler.transform(df_proc[[c for c in df_proc.columns if c!=target]].values)
                n_vis=min(2,Xall.shape[1],Xall.shape[0]); pca_vis=PCA(n_components=n_vis); X2d=pca_vis.fit_transform(Xall)
                lbl=best_cr['_labels']; fig,ax=dark_fig(7,5)
                sc_cols=['#00c896','#0078ff','#a855f7','#f59e0b','#ef4444','#06b6d4','#84cc16']
                for ci in sorted(set(lbl)):
                    mask=lbl==ci; col=sc_cols[ci%len(sc_cols)] if ci>=0 else '#374151'
                    cnt=mask.sum()
                    ax.scatter(X2d[mask,0],X2d[mask,1],c=col,s=15,alpha=.7,
                               label=f'{"Noise" if ci==-1 else f"C{ci}"} (n={cnt})')
                ax.legend(fontsize=7,labelcolor='#9ca3af',facecolor='#0d1117',edgecolor='#1f2937')
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

            with st.expander("📐 Elbow Chart"):
                Xall2=scaler.transform(df_proc[[c for c in df_proc.columns if c!=target]].values)
                inertias=[]; ks=range(2,min(11,len(Xall2)//5+2))
                for k in ks: km=KMeans(n_clusters=k,random_state=42,n_init=10); km.fit(Xall2); inertias.append(km.inertia_)
                fig,ax=dark_fig(7,3)
                ax.plot(list(ks),inertias,color='#00c896',marker='o',markersize=6,linewidth=2)
                for k,v in zip(ks,inertias):
                    ax.text(k,v+max(inertias)*0.01,f'{v:.0f}',ha='center',va='bottom',color='#6b7280',fontsize=7)
                ax.set_xlabel('K',color='#6b7280',fontsize=9); ax.set_ylabel('Inertia',color='#6b7280',fontsize=9)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── DR ───────────────────────────────────────────────────────────────
    if dr_res:
        st.markdown('<div class="section-sep"></div><div class="section-head">🔷 Dimensionality Reduction</div>',unsafe_allow_html=True)
        pca_dr=[r for r in dr_res if r['Method']=='PCA' and '_pca_full' in r]
        if pca_dr:
            with st.expander("📊 PCA Variance per Component (with values)"):
                pf=pca_dr[0]['_pca_full']; evr=pf.explained_variance_ratio_*100
                fig,ax=dark_fig(8,3)
                bars_pca=ax.bar(range(1,len(evr)+1),evr,color='#818cf8',alpha=.85)
                for bar,val in zip(bars_pca,evr):
                    if val>1: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,f'{val:.1f}%',
                                      ha='center',va='bottom',color='#9ca3af',fontsize=6)
                ax.plot(range(1,len(evr)+1),np.cumsum(evr),color='#00c896',marker='o',markersize=4,linewidth=2,label='Cumulative')
                ax.legend(fontsize=8,labelcolor='#9ca3af',facecolor='#0d1117',edgecolor='#1f2937')
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        lbl_col=None
        if cluster_res:
            valid2=[r for r in cluster_res if r.get('_labels') is not None]
            if valid2: lbl_col=max(valid2,key=lambda x:x['Silhouette'])['_labels']
        for r in dr_res:
            with st.expander(f"🗺 {r['Method']} 2D Projection"):
                X2d=r['_X2d']; idx=r.get('_idx',None)
                c_dr=lbl_col[idx] if lbl_col is not None and idx is not None else (
                          lbl_col[:len(X2d)] if lbl_col is not None else np.zeros(len(X2d)))
                fig,ax=dark_fig(7,5)
                sc=ax.scatter(X2d[:,0],X2d[:,1],c=c_dr,cmap='plasma',s=12,alpha=.7)
                ax.set_xlabel('C1',color='#6b7280',fontsize=9); ax.set_ylabel('C2',color='#6b7280',fontsize=9)
                ax.set_title(f'{r["Method"]} | Var: {r["Var(2D)"]}{"%" if isinstance(r["Var(2D)"],float) else ""}',
                             color='#9ca3af',fontsize=9)
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(4)
    with cb:
        if st.button("Business Insights →"): nav(6)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif step==6:
    problem_type=st.session_state.problem_type
    best_name=st.session_state.best_name
    sort_col=st.session_state.sort_col
    results_df=st.session_state.results
    best_score=results_df.iloc[0][sort_col] if results_df is not None and len(results_df) else 0
    target=st.session_state.target
    df=st.session_state.df

    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 7 — Business Use Cases & Insights</div>',unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
      <div class="insight-title">✦ Your Model Summary</div>
      <div style="display:flex;gap:2rem;flex-wrap:wrap">
        <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.08em">Problem Type</div>
          <div style="font-size:1.1rem;font-weight:700;color:#00c896;margin-top:.2rem">{problem_type.title()}</div></div>
        <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.08em">Best Model</div>
          <div style="font-size:1.1rem;font-weight:700;color:#d4d8f0;margin-top:.2rem">{best_name}</div></div>
        <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.08em">{sort_col}</div>
          <div style="font-size:1.1rem;font-weight:700;color:#00c896;margin-top:.2rem">{best_score:.2%}</div></div>
        <div><div style="font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:.08em">Target Column</div>
          <div style="font-size:1.1rem;font-weight:700;color:#d4d8f0;margin-top:.2rem">{target if target else 'None (Clustering)'}</div></div>
      </div>
      <div style="margin-top:1rem;font-size:.85rem;color:#6b7280;line-height:1.7">
        Based on your <strong style="color:#d4d8f0">{problem_type}</strong> model with
        <strong style="color:#00c896">{best_score:.2%}</strong> accuracy using <strong style="color:#d4d8f0">{best_name}</strong>,
        here are the most relevant real-world business applications where this model can generate value:
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    insights=get_business_insights(problem_type, best_name, best_score, st.session_state.feat_cols, target, df)

    # Display in grid
    html_grid='<div class="biz-grid">'
    for ins in insights:
        tags=''.join([f'<span class="biz-tag">{t}</span>' for t in ins['tags']])
        html_grid+=f"""
        <div class="biz-card">
          <div class="biz-icon">{ins['icon']}</div>
          <div class="biz-title">{ins['title']}</div>
          <div class="biz-desc">{ins['desc']}</div>
          <div>{tags}</div>
        </div>"""
    html_grid+='</div>'
    st.markdown(html_grid, unsafe_allow_html=True)

    # Deployment advice
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
      <div class="insight-title">🚀 How to Deploy This Model in Production</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-top:.8rem">
        <div style="background:rgba(0,200,150,.05);border:1px solid rgba(0,200,150,.15);border-radius:12px;padding:1rem">
          <div style="font-size:.75rem;font-weight:700;color:#00c896;margin-bottom:.4rem">① Download Model</div>
          <div style="font-size:.8rem;color:#6b7280">Export the trained .pkl file from the Report step and host it on your server.</div>
        </div>
        <div style="background:rgba(0,120,255,.05);border:1px solid rgba(0,120,255,.15);border-radius:12px;padding:1rem">
          <div style="font-size:.75rem;font-weight:700;color:#60a5fa;margin-bottom:.4rem">② Build an API</div>
          <div style="font-size:.8rem;color:#6b7280">Wrap it in a FastAPI or Flask endpoint — accept new data, return predictions in real time.</div>
        </div>
        <div style="background:rgba(168,85,247,.05);border:1px solid rgba(168,85,247,.15);border-radius:12px;padding:1rem">
          <div style="font-size:.75rem;font-weight:700;color:#c084fc;margin-bottom:.4rem">③ Integrate</div>
          <div style="font-size:.8rem;color:#6b7280">Connect your API to your CRM, app or dashboard via webhooks or REST calls.</div>
        </div>
        <div style="background:rgba(251,191,36,.05);border:1px solid rgba(251,191,36,.15);border-radius:12px;padding:1rem">
          <div style="font-size:.75rem;font-weight:700;color:#fbbf24;margin-bottom:.4rem">④ Monitor</div>
          <div style="font-size:.8rem;color:#6b7280">Track model drift over time. Retrain monthly with new data to keep accuracy high.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back to Results"): nav(5)
    with cb:
        if st.button("Export PDF Report →"): nav(7)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — PDF + DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════
elif step==7:
    results_df=st.session_state.results; cluster_res=st.session_state.cluster_results
    dr_res=st.session_state.dr_results; best_model=st.session_state.best_model
    best_name=st.session_state.best_name; scaler=st.session_state.scaler
    sort_col=st.session_state.sort_col; problem_type=st.session_state.problem_type
    target=st.session_state.target; tuning_results=st.session_state.tuning_results
    outlier_report=st.session_state.outlier_report
    best_score=results_df.iloc[0][sort_col] if results_df is not None and len(results_df) else 0
    st.markdown('<div class="card fadein"><div class="card-title">⬡ Step 8 — Export & Download</div>',unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;padding:1.5rem 0">
      <div style="font-size:2.5rem;margin-bottom:.8rem">🎉</div>
      <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#00c896;margin-bottom:.4rem">AutoML Complete!</div>
      <div style="font-size:.9rem;color:#6b7280">Best: <strong style="color:#d4d8f0">{best_name}</strong>
      | {sort_col}: <strong style="color:#00c896">{best_score:.4f}</strong></div>
    </div>""",unsafe_allow_html=True)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER

        buf_pdf=io.BytesIO()
        doc=SimpleDocTemplate(buf_pdf,pagesize=A4,topMargin=2*cm,bottomMargin=2*cm,leftMargin=2*cm,rightMargin=2*cm)
        styles=getSampleStyleSheet()
        ts=ParagraphStyle('t',parent=styles['Title'],fontSize=20,textColor=colors.HexColor('#00c896'),spaceAfter=6,alignment=TA_CENTER)
        h1=ParagraphStyle('h1',parent=styles['Heading1'],fontSize=13,textColor=colors.HexColor('#0078ff'),spaceBefore=10,spaceAfter=5)
        h2=ParagraphStyle('h2',parent=styles['Heading2'],fontSize=10,textColor=colors.HexColor('#a855f7'),spaceBefore=7,spaceAfter=3)
        bs=ParagraphStyle('b',parent=styles['Normal'],fontSize=9,textColor=colors.HexColor('#374151'),spaceAfter=3)
        story=[]
        story.append(Paragraph("AutoML Pro — Analysis Report",ts))
        story.append(Paragraph(f"Problem: {problem_type.title()} | Best: {best_name} | Score: {best_score:.4f}",bs))
        story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor('#00c896')))
        story.append(Spacer(1,.3*cm))

        story.append(Paragraph("1. Dataset Summary",h1))
        df_orig=st.session_state.df
        ds=[['Metric','Value'],['Rows',str(df_orig.shape[0])],['Columns',str(df_orig.shape[1])],
            ['Missing',str(df_orig.isnull().sum().sum())],['Target',str(target) if target else 'None'],
            ['Problem',problem_type.title()],['Best Model',best_name],['Best Score',f'{best_score:.4f}']]
        t=Table(ds,colWidths=[7*cm,8*cm])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#00c896')),('FONTSIZE',(0,0),(-1,-1),9),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f9fafb')]),
            ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),5)]))
        story.append(t); story.append(Spacer(1,.3*cm))

        if outlier_report:
            story.append(Paragraph("2. Outlier Detection",h1))
            if 'zscore' in outlier_report:
                od=[['Column','Count']]+[[k,str(v)] for k,v in outlier_report['zscore'].items()]
                t2=Table(od,colWidths=[9*cm,6*cm])
                t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#f87171')),
                    ('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),
                    ('PADDING',(0,0),(-1,-1),5)]))
                story.append(t2)
            if 'isolation_forest' in outlier_report:
                story.append(Paragraph(f"Isolation Forest: {outlier_report['isolation_forest']} anomalies detected",bs))
            story.append(Spacer(1,.3*cm))

        if results_df is not None and len(results_df):
            story.append(Paragraph("3. ML Model Comparison",h1))
            disp=[c for c in results_df.columns if not c.startswith('_')]
            ml_data=[disp]+[[str(x) for x in row] for row in results_df[disp].values.tolist()]
            t3=Table(ml_data,colWidths=[17*cm/len(disp)]*len(disp))
            t3.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#00c896')),('FONTSIZE',(0,0),(-1,-1),8),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0fdf4')]),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),4)]))
            story.append(t3); story.append(Spacer(1,.3*cm))

        if tuning_results:
            story.append(Paragraph("4. Hyperparameter Tuning",h1))
            for mn,tr in tuning_results.items():
                story.append(Paragraph(f"<b>{mn}</b>: CV Score={tr['best_score']} | Params={tr['best_params']}",bs))
            story.append(Spacer(1,.3*cm))

        if cluster_res:
            story.append(Paragraph("5. Clustering Results",h1))
            cd=[['Model','Silhouette','Davies-Bouldin','Clusters']]+[
                [r['Model'],str(r['Silhouette']),str(r['Davies-Bouldin']),str(r['Clusters'])] for r in cluster_res]
            t4=Table(cd,colWidths=[6*cm,4*cm,4*cm,3*cm])
            t4.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#fbbf24')),('FONTSIZE',(0,0),(-1,-1),9),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#fffbeb')]),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),5)]))
            story.append(t4); story.append(Spacer(1,.3*cm))

        # Business insights in PDF
        story.append(Paragraph("6. Business Use Cases",h1))
        ins_list=get_business_insights(problem_type,best_name,best_score,st.session_state.feat_cols,target,df_orig)
        for ins in ins_list:
            story.append(Paragraph(f"{ins['icon']} <b>{ins['title']}</b>",h2))
            story.append(Paragraph(ins['desc'],bs))
        story.append(Spacer(1,.3*cm))

        story.append(Paragraph("7. Deployment Recommendations",h1))
        deploy_steps=[['Step','Action','Details'],
            ['①','Download Model','Export .pkl file and host on your server or cloud (AWS/GCP/Azure)'],
            ['②','Build API','Wrap in FastAPI/Flask — accept new data, return predictions in real time'],
            ['③','Integrate','Connect to your CRM, app or dashboard via REST API or webhooks'],
            ['④','Monitor','Track accuracy drift monthly, retrain with new data to maintain performance']]
        t5=Table(deploy_steps,colWidths=[1.5*cm,4*cm,11.5*cm])
        t5.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#818cf8')),('FONTSIZE',(0,0),(-1,-1),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#eef2ff')]),
            ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),5),
            ('VALIGN',(0,0),(-1,-1),'TOP')]))
        story.append(t5)

        doc.build(story); buf_pdf.seek(0)
        st.session_state.pdf_buf=buf_pdf.read()
        st.success("✅ PDF Report ready!")
    except ImportError:
        st.warning("Add `reportlab` to requirements.txt to enable PDF export.")
        st.session_state.pdf_buf=None

    c1,c2,c3,c4=st.columns(4)
    with c1:
        if best_model:
            buf=io.BytesIO(); joblib.dump(best_model,buf); buf.seek(0)
            st.download_button("⬇ Best Model (.pkl)",buf,f"{best_name.replace(' ','_')}.pkl","application/octet-stream",use_container_width=True)
    with c2:
        buf2=io.BytesIO(); joblib.dump(scaler,buf2); buf2.seek(0)
        st.download_button("⬇ Scaler (.pkl)",buf2,"scaler.pkl","application/octet-stream",use_container_width=True)
    with c3:
        rows=[]
        if results_df is not None:
            for _,r in results_df.iterrows(): rows.append({k:v for k,v in r.items() if not k.startswith('_')})
        if cluster_res:
            for r in cluster_res: rows.append({k:v for k,v in r.items() if not k.startswith('_')})
        if rows: st.download_button("⬇ Results (.csv)",pd.DataFrame(rows).to_csv(index=False),"results.csv","text/csv",use_container_width=True)
    with c4:
        if st.session_state.get('pdf_buf'):
            st.download_button("⬇ Full PDF Report",st.session_state.pdf_buf,"automl_pro_report.pdf","application/pdf",use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🔄 Start Over"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1.5rem;
  border-top:1px solid rgba(255,255,255,.04);
  font-family:'Space Mono',monospace;font-size:.65rem;color:#1f2937;letter-spacing:.08em">
  AUTOML PRO · EDA · WORD CLOUD · ML · CLUSTERING · DR · TUNING · SHAP · BUSINESS INSIGHTS · PDF
</div>
""",unsafe_allow_html=True)
