import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import joblib, io, warnings, re
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
    mean_squared_error, classification_report, confusion_matrix,
    silhouette_score, davies_bouldin_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy import stats

# ── Theme constants ────────────────────────────────────────────────────────
BG='#04060d'; PANEL='#090d18'; GRID='#151c2c'
C1,C2,C3,C4,C5='#00e5a0','#3b82f6','#a855f7','#f59e0b','#ef4444'
PAL=[C1,C2,C3,C4,C5,'#06b6d4','#84cc16','#f43f5e','#fb923c','#818cf8']

st.set_page_config(page_title="AutoML Pro", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background:#04060d!important;color:#d4d8f0;font-family:'Outfit',sans-serif}
[data-testid="stAppViewContainer"]{
  background:#04060d!important;
  background-image:
    radial-gradient(ellipse 70% 50% at 10% -10%,rgba(0,229,160,.07) 0%,transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 110%,rgba(59,130,246,.07) 0%,transparent 55%),
    radial-gradient(ellipse 40% 40% at 50% 50%,rgba(168,85,247,.03) 0%,transparent 70%)}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important}
.block-container{padding:1.8rem 2.5rem 4rem;max-width:1240px;margin:auto}
/* Hero */
.hero{padding:2.5rem 0 1.5rem;border-bottom:1px solid rgba(255,255,255,.05);margin-bottom:2rem}
.hero-top{display:flex;align-items:center;gap:1.5rem;margin-bottom:.9rem}
.hero-icon{font-size:2.8rem;background:linear-gradient(135deg,#00e5a0,#3b82f6,#a855f7);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero h1{font-size:2.5rem;font-weight:800;
  background:linear-gradient(135deg,#fff 0%,#00e5a0 45%,#3b82f6 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1}
.hero-sub{font-size:.88rem;color:#374151;margin-top:.3rem}
.feature-pills{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.9rem}
.fp{display:inline-block;padding:.25rem .75rem;border-radius:50px;font-size:.7rem;font-weight:600;letter-spacing:.02em}
.fp.g{background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.25);color:#00e5a0}
.fp.b{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);color:#60a5fa}
.fp.p{background:rgba(168,85,247,.1);border:1px solid rgba(168,85,247,.25);color:#c084fc}
.fp.y{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25);color:#fbbf24}
.fp.r{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);color:#f87171}
.fp.t{background:rgba(6,182,212,.1);border:1px solid rgba(6,182,212,.25);color:#22d3ee}
.fp.o{background:rgba(251,146,60,.1);border:1px solid rgba(251,146,60,.25);color:#fb923c}
.fp.pk{background:rgba(244,63,94,.1);border:1px solid rgba(244,63,94,.25);color:#fb7185}
/* Pipeline */
.pipeline{display:flex;margin-bottom:2rem;background:rgba(255,255,255,.015);
  border:1px solid rgba(255,255,255,.05);border-radius:14px;overflow:hidden}
.ps{flex:1;padding:.65rem .25rem;text-align:center;border-right:1px solid rgba(255,255,255,.05);cursor:default}
.ps:last-child{border-right:none}
.ps.active{background:rgba(0,229,160,.07)}
.ps.done{background:rgba(0,229,160,.03)}
.ps-num{font-family:'Space Mono',monospace;font-size:.52rem;color:#1f2937;margin-bottom:.12rem}
.ps.active .ps-num,.ps.done .ps-num{color:#00e5a0}
.ps-label{font-size:.6rem;font-weight:600;color:#374151}
.ps.active .ps-label{color:#d4d8f0}
.ps.done .ps-label{color:#4b5563}
/* Cards */
.card{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);
  border-radius:18px;padding:1.5rem;margin-bottom:1.5rem}
.ctitle{font-family:'Space Mono',monospace;font-size:.65rem;color:#00e5a0;
  letter-spacing:.14em;text-transform:uppercase;margin-bottom:1rem}
.sep{height:1px;background:linear-gradient(90deg,transparent,rgba(0,229,160,.2),transparent);margin:1.4rem 0}
.sh{font-family:'Space Mono',monospace;font-size:.58rem;color:#374151;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.7rem}
/* Metric cards */
.mrow{display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1.2rem}
.mbox{flex:1;min-width:90px;background:linear-gradient(135deg,rgba(0,229,160,.06),rgba(59,130,246,.04));
  border:1px solid rgba(0,229,160,.15);border-radius:14px;padding:.9rem;text-align:center;
  box-shadow:0 2px 20px rgba(0,229,160,.05)}
.mval{font-family:'Space Mono',monospace;font-size:1.35rem;font-weight:700;color:#00e5a0}
.mkey{font-size:.63rem;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;margin-top:.2rem}
/* Model rows */
.mrow2{display:flex;align-items:center;gap:.8rem;padding:.6rem .5rem;
  border-bottom:1px solid rgba(255,255,255,.03);border-radius:8px;margin-bottom:2px}
.mrow2.best{background:linear-gradient(90deg,rgba(0,229,160,.07),transparent);
  border:1px solid rgba(0,229,160,.18);border-radius:11px}
.mname{flex:2;font-size:.82rem;font-weight:500;color:#d4d8f0}
.mscore{flex:1;font-family:'Space Mono',monospace;font-size:.78rem;color:#00e5a0;text-align:right}
.mbar-w{flex:3;height:5px;background:rgba(255,255,255,.05);border-radius:10px;overflow:hidden}
.mbar{height:100%;background:linear-gradient(90deg,#00e5a0,#3b82f6);border-radius:10px}
.badge{font-size:.55rem;font-weight:700;padding:.1rem .42rem;border-radius:50px}
.badge.best{background:linear-gradient(135deg,#00e5a0,#3b82f6);color:#000}
.badge.tune{background:rgba(245,158,11,.2);border:1px solid rgba(245,158,11,.4);color:#fbbf24}
.badge.boost{background:rgba(239,68,68,.2);border:1px solid rgba(239,68,68,.4);color:#f87171}
/* Chips */
.chip{display:inline-block;border-radius:50px;padding:.2rem .65rem;font-size:.73rem;margin:.18rem}
.chip.g{background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.2);color:#00e5a0}
.chip.r{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);color:#f87171}
.chip.p{background:rgba(168,85,247,.08);border:1px solid rgba(168,85,247,.2);color:#c084fc}
.chip.b{background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);color:#60a5fa}
/* Insight/biz boxes */
.ibox{background:linear-gradient(135deg,rgba(0,229,160,.04),rgba(59,130,246,.04));
  border:1px solid rgba(0,229,160,.15);border-radius:16px;padding:1.4rem;margin:1rem 0}
.ititle{font-family:'Space Mono',monospace;font-size:.65rem;color:#00e5a0;
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:.8rem}
.biz-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem;margin-top:1rem}
.bcard{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06);
  border-radius:14px;padding:1.2rem;transition:all .25s}
.bcard:hover{border-color:rgba(0,229,160,.25);background:rgba(0,229,160,.03)}
.bicon{font-size:1.8rem;margin-bottom:.5rem}
.btitle{font-family:'Space Mono',monospace;font-size:.66rem;color:#00e5a0;
  letter-spacing:.08em;text-transform:uppercase;margin-bottom:.4rem}
.bdesc{font-size:.8rem;color:#6b7280;line-height:1.6}
.btag{display:inline-block;background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.18);
  border-radius:50px;padding:.1rem .5rem;font-size:.64rem;color:#00e5a0;margin:.3rem .2rem 0 0}
.nlpbox{background:rgba(168,85,247,.05);border:1px solid rgba(168,85,247,.2);
  border-radius:14px;padding:1.1rem 1.4rem;margin:1rem 0}
.aibox{background:rgba(168,85,247,.03);border:1px solid rgba(168,85,247,.15);
  border-radius:14px;padding:1.5rem;margin:1rem 0;line-height:1.85;font-size:.9rem}
.aibox h3{color:#c084fc;font-size:.95rem;margin:.9rem 0 .4rem;font-weight:700}
/* Buttons */
div[data-testid="stFileUploader"]{background:rgba(0,229,160,.02);
  border:2px dashed rgba(0,229,160,.2);border-radius:14px;padding:1rem}
.stButton>button{background:linear-gradient(135deg,#00e5a0,#3b82f6)!important;color:#000!important;
  font-family:'Outfit',sans-serif!important;font-weight:700!important;border:none!important;
  border-radius:12px!important;padding:.6rem 1.8rem!important;width:100%!important;
  box-shadow:0 4px 24px rgba(0,229,160,.2)!important;transition:all .25s!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(0,229,160,.35)!important}
.stProgress>div>div{background:linear-gradient(90deg,#00e5a0,#3b82f6)!important}
[data-testid="stExpander"]{background:rgba(255,255,255,.015)!important;
  border:1px solid rgba(255,255,255,.06)!important;border-radius:12px!important}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.fadein{animation:fadeUp .45s ease both}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-top"><div class="hero-icon">⚡</div>
    <div><h1>AutoML Pro</h1>
    <div class="hero-sub">Enterprise automated ML · Best-in-class visualisations · Business-ready insights</div></div>
  </div>
  <div class="feature-pills">
    <span class="fp g">✓ EDA + Word Cloud</span>
    <span class="fp r">✓ Outlier Detection</span>
    <span class="fp b">✓ XGBoost · LightGBM · CatBoost</span>
    <span class="fp y">✓ Hyperparameter Tuning</span>
    <span class="fp t">✓ Time Series ARIMA</span>
    <span class="fp p">✓ SHAP Explainability</span>
    <span class="fp pk">✓ Neural Net Visualizer</span>
    <span class="fp o">✓ AI Report · Business Insights</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
DEFS={'step':0,'df':None,'df_clean':None,'target':None,'results':None,'best_model':None,
      'problem_type':None,'nlp_mode':False,'text_col':None,'text_cols_detected':[],
      'df_proc':None,'scaler':None,'feat_cols':[],'X_test':None,'X_test_sc':None,
      'y_test':None,'le_target':None,'tfidf':None,'best_name':'','best_scaled':False,
      'metric_name':'Accuracy','sort_col':'Accuracy','cluster_results':[],'dr_results':[],
      'outlier_report':{},'tuning_results':{},'pdf_buf':None,'ai_report':'',
      'ts_results':{},'is_time_series':False,'date_col':None}
for k,v in DEFS.items():
    if k not in st.session_state: st.session_state[k]=v

step=st.session_state.step
STEPS=["Upload","Clean","EDA","Preprocess","Train","Results","NN Viz","AI Report","Insights","Export"]
ph='<div class="pipeline">'
for i,lbl in enumerate(STEPS):
    cls="active" if i==step else ("done" if i<step else "")
    icon="✓" if i<step else str(i+1)
    ph+=f'<div class="ps {cls}"><div class="ps-num">{icon}</div><div class="ps-label">{lbl}</div></div>'
ph+='</div>'
st.markdown(ph,unsafe_allow_html=True)

def nav(s): st.session_state.step=s; st.rerun()

def safe_chart(fn, *args, **kwargs):
    """Run a chart function, silently skip if it errors."""
    try:
        fn(*args, **kwargs)
    except Exception as e:
        st.warning(f"⚠️ Chart skipped: {e}")
        plt.close('all')

import contextlib
@contextlib.contextmanager
def safe_section(label="section"):
    """Context manager — shows warning instead of crashing on error."""
    try:
        yield
    except Exception as e:
        st.warning(f"⚠️ {label} could not render: {e}")
        plt.close('all')

# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════
def make_fig(w=9,h=4):
    fig,ax=plt.subplots(figsize=(w,h),facecolor=BG); ax.set_facecolor(PANEL); return fig,ax

def make_figs(rows,cols,w=12,h=4,**kw):
    fig,axes=plt.subplots(rows,cols,figsize=(w,h),facecolor=BG,**kw)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(PANEL)
    return fig,axes

def polish(ax,xlabel=None,ylabel=None,title=None,grid='y',legend=False):
    ax.tick_params(colors='#4b5563',labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    if grid=='y': ax.yaxis.grid(True,color=GRID,lw=.6,alpha=.8,ls='--'); ax.set_axisbelow(True)
    if grid=='x': ax.xaxis.grid(True,color=GRID,lw=.6,alpha=.8,ls='--'); ax.set_axisbelow(True)
    if grid=='both': ax.grid(True,color=GRID,lw=.5,alpha=.7,ls='--'); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel,color='#4b5563',fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel,color='#4b5563',fontsize=8.5)
    if title:  ax.set_title(title,color='#9ca3af',fontsize=10,pad=8)
    if legend:
        leg=ax.legend(fontsize=7.5,labelcolor='#9ca3af',facecolor='#0b0f1a',
                      edgecolor=GRID,framealpha=.9,loc='best')

def vlabel(ax,fmt='{:.0f}',color='#6b7280',fs=7.5,pad_frac=0.025):
    """Value labels on vertical bars"""
    for p in ax.patches:
        h=p.get_height()
        if h>0:
            ax.text(p.get_x()+p.get_width()/2, h+max(.002,h*pad_frac),
                    fmt.format(h), ha='center', va='bottom', color=color, fontsize=fs, fontweight='bold')

def hlabel(ax,fmt='{:.4f}',color='#9ca3af',fs=7.5,pad_frac=0.015):
    """Value labels on horizontal bars"""
    for p in ax.patches:
        w=p.get_width()
        if abs(w)>0:
            ax.text(w+max(.002,abs(w)*pad_frac), p.get_y()+p.get_height()/2,
                    fmt.format(w), va='center', ha='left', color=color, fontsize=fs, fontweight='bold')

def grad_fill(ax, x, y, color1, alpha=0.22):
    """Multi-layer gradient fill under a line for depth"""
    for a in [alpha*0.3, alpha*0.6, alpha]:
        ax.fill_between(x, y, alpha=a, color=color1, zorder=1)

def glow_line(ax, x, y, color=C1, lw=2.2, label='', zorder=3):
    """Line with layered glow"""
    for w,a in [(lw*4,.04),(lw*2.5,.09),(lw*1.5,.2),(lw,1.0)]:
        kw={'label':label} if (a==1.0 and label) else {}
        ax.plot(x, y, color=color, lw=w, alpha=a, zorder=zorder,
                solid_capstyle='round', **kw)

def glow_scatter(ax, x, y, color=C1, s=18, label='', zorder=3):
    """Scatter with concentric glow rings"""
    for si,a in [(s*10,.03),(s*5,.07),(s*2,.15),(s,.9)]:
        kw={'label':label} if (a==.9 and label) else {}
        ax.scatter(x, y, color=color, s=si, alpha=a, zorder=zorder,
                   edgecolors='none', **kw)

def hbar_gradient(ax, labels, values, base_color=C1, title=''):
    """Horizontal bars with gradient + embedded name + value labels"""
    n=len(labels)
    cmap=mcolors.LinearSegmentedColormap.from_list('hb',['#1a2a40', base_color])
    colors=[cmap(i/max(n-1,1)) for i in range(n)]
    colors[-1]=base_color  # brightest = top (best)
    bars=ax.barh(range(n), values, color=colors[::-1], height=0.62, zorder=3)
    for bar,val,lbl in zip(bars,values[::-1],labels[::-1]):
        ax.text(val+max(.002,abs(val)*.015), bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', color='#e2e8f0', fontsize=8, fontweight='bold')
        ax.text(max(.002,val*.01), bar.get_y()+bar.get_height()/2,
                lbl, va='center', ha='left', color='white', fontsize=7.5,
                fontweight='600', zorder=4, clip_on=True)
    ax.set_yticks([])
    return bars

def gauge(fig_ax, score, title='', color=C1, max_val=1.0):
    """Beautiful half-gauge chart"""
    ax=fig_ax; ax.set_facecolor(BG); ax.axis('off')
    θ=np.linspace(np.pi,0,300)
    # Track rings
    for r,alpha in [(1.0,.08),(0.85,.12),(0.7,.18)]:
        ax.plot(r*np.cos(θ),r*np.sin(θ),color=color,lw=1,alpha=alpha)
    # BG arc
    ax.plot(np.cos(θ),np.sin(θ),color=GRID,lw=18,solid_capstyle='round',zorder=2)
    # Colored arc
    frac=min(score/max_val,1.0)
    θend=np.pi-frac*np.pi
    θarc=np.linspace(np.pi,θend,300)
    # Glow effect layers
    for lw,alpha in [(24,.08),(20,.12),(16,.2),(12,1.0)]:
        ax.plot(np.cos(θarc),np.sin(θarc),color=color,lw=lw,
                solid_capstyle='round',zorder=3,alpha=alpha)
    # Needle
    na=np.pi-frac*np.pi
    ax.annotate('',xy=(0.78*np.cos(na),0.78*np.sin(na)),xytext=(0,0),
                arrowprops=dict(arrowstyle='->,head_width=0.08,head_length=0.06',
                                color='white',lw=2.5),zorder=5)
    ax.add_patch(plt.Circle((0,0),.07,color=color,zorder=6,alpha=.9))
    ax.add_patch(plt.Circle((0,0),.04,color='white',zorder=7))
    # Labels
    ax.text(0,-0.28,f'{score:.1%}' if max_val==1.0 else f'{score:.4f}',
            ha='center',va='center',fontsize=16,fontweight='bold',color=color,fontfamily='monospace',zorder=8)
    ax.text(0,-0.5,title,ha='center',va='center',fontsize=8.5,color='#6b7280',zorder=8)
    # Tick marks
    for pct in [0,0.25,0.5,0.75,1.0]:
        a=np.pi-pct*np.pi
        ax.text(0.93*np.cos(a),0.93*np.sin(a),f'{int(pct*100)}',
                ha='center',va='center',fontsize=6,color='#374151')
    ax.set_xlim(-1.15,1.15); ax.set_ylim(-0.6,1.1); ax.set_aspect('equal')

def radar(ax, labels, vals, color=C1, title=''):
    N=len(labels)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()
    angles_c=angles+[angles[0]]; vals_c=list(vals)+[vals[0]]
    ax.set_facecolor(PANEL)
    # Background rings
    for r in [0.25,0.5,0.75,1.0]:
        ax.plot([np.cos(a)*r for a in angles_c],[np.sin(a)*r for a in angles_c],
                color=GRID,lw=.6,alpha=.6)
    # Spokes
    for a in angles:
        ax.plot([0,np.cos(a)],[0,np.sin(a)],color=GRID,lw=.6,alpha=.5)
    # Data
    ax.fill([np.cos(a)*v for a,v in zip(angles_c,vals_c)],
            [np.sin(a)*v for a,v in zip(angles_c,vals_c)],
            alpha=.18,color=color)
    ax.plot([np.cos(a)*v for a,v in zip(angles_c,vals_c)],
            [np.sin(a)*v for a,v in zip(angles_c,vals_c)],
            color=color,lw=2,zorder=3)
    ax.scatter([np.cos(a)*v for a,v in zip(angles,vals)],
               [np.sin(a)*v for a,v in zip(angles,vals)],
               color=color,s=35,zorder=4,edgecolors='white',linewidths=.7)
    # Labels
    for a,lbl,v in zip(angles,labels,vals):
        r=1.22
        ax.text(r*np.cos(a),r*np.sin(a),f'{lbl}\n{v:.2f}',
                ha='center',va='center',fontsize=6.5,color='#9ca3af',fontweight='bold')
    ax.set_xlim(-1.45,1.45); ax.set_ylim(-1.45,1.45)
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title,color='#9ca3af',fontsize=9,pad=4)

def scatter_3d_feel(ax, x, y, c=None, cmap='plasma', size=18, label='', color=C1):
    """Scatter with glow effect"""
    if c is not None:
        for alpha,s in [(0.05,size*8),(0.1,size*4),(0.6,size)]:
            ax.scatter(x,y,c=c,cmap=cmap,s=s*1.5,alpha=alpha,zorder=2)
        ax.scatter(x,y,c=c,cmap=cmap,s=size,alpha=.9,zorder=3)
    else:
        for alpha,s in [(0.05,size*8),(0.1,size*4)]:
            ax.scatter(x,y,color=color,s=s,alpha=alpha,zorder=2)
        ax.scatter(x,y,color=color,s=size,alpha=.9,zorder=3,label=label)

def draw_confusion(ax, cm, class_names=None):
    """Styled confusion matrix"""
    n=cm.shape[0]
    cmap=mcolors.LinearSegmentedColormap.from_list('cm',['#0b0f1a','#00e5a0'])
    im=ax.imshow(cm,cmap=cmap,aspect='auto',alpha=.85)
    for i in range(n):
        for j in range(n):
            clr='white' if cm[i,j]<cm.max()*0.55 else '#04060d'
            ax.text(j,i,f'{cm[i,j]}',ha='center',va='center',
                    fontsize=max(7,min(14,80//n)),fontweight='bold',color=clr)
    if class_names is not None and len(class_names) > 0:
        ax.set_xticks(range(n)); ax.set_xticklabels(class_names,rotation=35,ha='right',fontsize=7,color='#6b7280')
        ax.set_yticks(range(n)); ax.set_yticklabels(class_names,fontsize=7,color='#6b7280')
    for sp in ax.spines.values(): sp.set_color(GRID)

# ══════════════════════════════════════════════════════════════════════════
# BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
def get_biz(problem_type,best_name,best_score,target):
    sc=f"{best_score*100:.1f}%"
    if problem_type=='classification':
        return [
            {"icon":"🎯","title":"Customer Churn Prevention","desc":f"Predict which customers will leave with {sc} accuracy. Trigger automated win-back at 30-day risk window — saves 5–10× acquisition cost.","tags":["CRM","Retention","SaaS"],"roi":"💰 Very High ROI"},
            {"icon":"🔍","title":"Real-Time Fraud Detection","desc":f"Flag suspicious transactions instantly with {best_name}. Reduces false positives vs rule-based systems, cutting fraud losses by up to 40%.","tags":["FinTech","Security","Banking"],"roi":"💰 Very High ROI"},
            {"icon":"🏥","title":"Clinical Decision Support","desc":f"{sc} diagnostic accuracy to triage high-risk patients. Reduces missed diagnoses and ER overcrowding.","tags":["Healthcare","Diagnostics","AI"],"roi":"⚕️ Critical Impact"},
            {"icon":"📧","title":"Smart Lead Scoring","desc":f"Classify inbound leads Hot/Warm/Cold in real time. Sales teams focus only on the top {sc} probability prospects.","tags":["B2B Sales","CRM","Revenue"],"roi":"💰 High ROI"},
            {"icon":"🛒","title":"Product Recommendation Engine","desc":f"Classify customer intent and serve personalised categories, boosting CVR and average order value automatically.","tags":["E-commerce","UX","Personalisation"],"roi":"💰 Medium ROI"},
            {"icon":"⚠️","title":"Predictive Maintenance","desc":f"Classify machine health before failure. Reduce unplanned downtime by scheduling maintenance proactively with {sc} accuracy.","tags":["Manufacturing","IoT","Ops"],"roi":"💰 Very High ROI"},
        ]
    elif problem_type=='regression':
        return [
            {"icon":"🏠","title":"Automated Property Valuation","desc":f"Instant valuations for banks, agents and buyers powered by {best_name} with {sc} R² accuracy.","tags":["Real Estate","FinTech","PropTech"],"roi":"💰 High ROI"},
            {"icon":"📈","title":"Revenue & Sales Forecasting","desc":f"Data-backed revenue forecasts driving better inventory, hiring and budget decisions every quarter.","tags":["Finance","Planning","Retail"],"roi":"💰 Very High ROI"},
            {"icon":"⚡","title":"Energy Demand Optimisation","desc":f"Predict consumption by hour/day. Reduce grid waste and procurement costs for utilities and smart buildings.","tags":["Energy","IoT","Green"],"roi":"🌱 Sustainability"},
            {"icon":"💰","title":"Dynamic Pricing Engine","desc":f"Predict optimal price points based on demand and seasonality. Maximise revenue without manual rules.","tags":["E-commerce","Pricing","Strategy"],"roi":"💰 Very High ROI"},
            {"icon":"👥","title":"Employee Lifetime Value","desc":f"Predict tenure and productivity. Focus HR retention on high-value staff before they leave.","tags":["HR Tech","People Ops","Culture"],"roi":"💰 Medium ROI"},
            {"icon":"🚚","title":"Logistics Cost Prediction","desc":f"Forecast delivery costs and route times. Reduce logistics spend and improve SLA compliance at scale.","tags":["Logistics","Supply Chain","Ops"],"roi":"💰 High ROI"},
        ]
    else:
        return [
            {"icon":"👥","title":"Customer Micro-Segmentation","desc":f"Group customers by spend and behaviour. Run hyper-targeted campaigns per segment, increasing ROAS 2–5×.","tags":["Marketing","CRM","Growth"],"roi":"💰 Very High ROI"},
            {"icon":"🗺","title":"Geo-Market Clustering","desc":f"Identify geographic markets for expansion. Prioritise regions with cluster profiles matching your best markets.","tags":["Strategy","Expansion","GTM"],"roi":"💰 Strategic"},
            {"icon":"🔬","title":"Anomaly & Threat Detection","desc":f"Isolate points outside all clusters — potential fraud, network intrusions or equipment faults.","tags":["Security","FinTech","QA"],"roi":"💰 Very High ROI"},
            {"icon":"🏭","title":"Predictive Maintenance Groups","desc":f"Cluster machines by failure signature. Build maintenance schedules per cluster, cutting breakdown events dramatically.","tags":["Manufacturing","IoT","Reliability"],"roi":"💰 High ROI"},
            {"icon":"📰","title":"Content Topic Clustering","desc":f"Group articles and social posts by theme. Power content recommendations and editorial planning without labels.","tags":["Media","NLP","Publishing"],"roi":"💰 Medium ROI"},
            {"icon":"🧬","title":"Patient Cohort Discovery","desc":f"Uncover hidden patient subgroups with similar profiles. Accelerate clinical trial design and personalised treatment.","tags":["BioTech","Research","Pharma"],"roi":"⚕️ Critical Impact"},
        ]

# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════
if step==0:
    st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 1 — Upload Dataset</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload CSV (max 20k rows on Community Cloud)",type=["csv"],label_visibility="collapsed")
    if uploaded:
        df=pd.read_csv(uploaded)
        if len(df)>20000:
            st.warning(f"⚠️ Large dataset ({len(df):,} rows) — sampling 20,000 rows to stay within Streamlit Community Cloud memory limits.")
            df=df.sample(20000,random_state=42).reset_index(drop=True)
        st.session_state.df=df; st.session_state.df_clean=None  # will set after clean step

        # Animated metric cards
        st.markdown(f"""<div class="mrow">
          <div class="mbox"><div class="mval">{df.shape[0]:,}</div><div class="mkey">Rows</div></div>
          <div class="mbox"><div class="mval">{df.shape[1]}</div><div class="mkey">Columns</div></div>
          <div class="mbox"><div class="mval">{df.isnull().sum().sum()}</div><div class="mkey">Missing</div></div>
          <div class="mbox"><div class="mval">{df.dtypes[df.dtypes=='object'].count()}</div><div class="mkey">Categorical</div></div>
          <div class="mbox"><div class="mval">{df.select_dtypes(include=np.number).shape[1]}</div><div class="mkey">Numeric</div></div>
        </div>""",unsafe_allow_html=True)

        # Data type pie chart + missing heatmap side by side
        c1,c2=st.columns(2)
        with c1:
            dtype_counts={'Numeric':df.select_dtypes(include=np.number).shape[1],
                          'Categorical':df.select_dtypes(include='object').shape[1],
                          'Other':max(0,df.shape[1]-df.select_dtypes(include=np.number).shape[1]-df.select_dtypes(include='object').shape[1])}
            dtype_counts={k:v for k,v in dtype_counts.items() if v>0}
            fig,ax=plt.subplots(figsize=(4,3.5),facecolor=BG); ax.set_facecolor(BG)
            wedges,texts,autotexts=ax.pie(dtype_counts.values(),labels=dtype_counts.keys(),
                autopct='%1.0f%%',colors=[C1,C2,C3],startangle=140,
                wedgeprops=dict(width=0.6,edgecolor=BG,linewidth=3),
                textprops=dict(color='#9ca3af',fontsize=8))
            for at in autotexts: at.set_color('#04060d'); at.set_fontweight('bold'); at.set_fontsize(8)
            ax.set_title('Column Types',color='#9ca3af',fontsize=9,pad=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            miss=df.isnull().sum(); miss=miss[miss>0]
            if len(miss)>0:
                fig2,ax2=make_fig(4,3.5)
                bars=ax2.barh(miss.index[::-1],miss.values[::-1]/len(df)*100,
                              color=[C5 if v>50 else C4 if v>20 else C1 for v in miss.values[::-1]],height=0.6)
                hlabel(ax2,fmt='{:.1f}%',fs=7)
                ax2.set_xlabel('Missing %',color='#4b5563',fontsize=8)
                ax2.axvline(50,color=C5,lw=1,ls='--',alpha=.5)
                polish(ax2,title='Missing Values %')
                plt.tight_layout(); st.pyplot(fig2); plt.close()
            else:
                st.success("✅ No missing values in this dataset!")

        st.dataframe(df.head(6),use_container_width=True)
        date_candidates=[c for c in df.columns if any(x in c.lower() for x in ['date','time','year','month','day','period'])]
        col1,col2=st.columns(2)
        with col1:
            target=st.selectbox("🎯 Target Column",["-- No target (Clustering) --"]+df.columns.tolist())
            st.session_state.target=None if target.startswith("--") else target
        with col2:
            is_ts=st.toggle("📈 Time Series Mode",value=False); st.session_state.is_time_series=is_ts
            if is_ts and date_candidates:
                st.session_state.date_col=st.selectbox("📅 Date/Time Column",date_candidates)
        if st.button("Continue →"): nav(1)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — CLEAN + OUTLIERS
# ══════════════════════════════════════════════════════════════════════════
elif step==1:
    plt.close('all')
    try:
        df=st.session_state.df; target=st.session_state.target
        num_cols=df.select_dtypes(include=np.number).columns.tolist()
        feat_num=[c for c in num_cols if c!=target]
        st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 2 — Cleaning & Outlier Detection</div>',unsafe_allow_html=True)

        outlier_report={}
        if feat_num:
            st.markdown('<div class="sh">🔴 Outlier Analysis</div>',unsafe_allow_html=True)
            z_out={col:int((np.abs(stats.zscore(df[col].dropna()))>3).sum()) for col in feat_num[:10]}
            z_out={k:v for k,v in z_out.items() if v>0}
            iqr_out={}
            for col in feat_num[:10]:
                Q1,Q3=df[col].quantile(.25),df[col].quantile(.75); IQR=Q3-Q1
                n=int(((df[col]<Q1-1.5*IQR)|(df[col]>Q3+1.5*IQR)).sum())
                if n>0: iqr_out[col]=n

            # Combined outlier bar chart
            if z_out or iqr_out:
                all_cols=list(set(list(z_out.keys())+list(iqr_out.keys())))[:10]
                z_vals=[z_out.get(c,0) for c in all_cols]
                i_vals=[iqr_out.get(c,0) for c in all_cols]
                x=np.arange(len(all_cols)); w=0.38
                fig,ax=make_fig(10,3.5)
                b1=ax.bar(x-w/2,z_vals,w,color=C5,alpha=.85,label='Z-Score',zorder=3)
                b2=ax.bar(x+w/2,i_vals,w,color=C4,alpha=.85,label='IQR',zorder=3)
                for bars in [b1,b2]:
                    for bar in bars:
                        if bar.get_height()>0:
                            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.3,
                                    f'{int(bar.get_height())}',ha='center',va='bottom',color='#9ca3af',fontsize=7,fontweight='bold')
                ax.set_xticks(x); ax.set_xticklabels(all_cols,rotation=25,ha='right',fontsize=7,color='#6b7280')
                polish(ax,ylabel='Count',title='Outliers per Column — Z-Score vs IQR',legend=True)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                outlier_report['zscore']=z_out; outlier_report['iqr']=iqr_out

            # Boxplots for top outlier columns — styled
            top_cols=(list(z_out.keys()) if z_out else feat_num)[:5]
            fig2,axes2=make_figs(1,len(top_cols),w=min(14,len(top_cols)*2.8),h=3.5)
            axes2=np.array(axes2).flatten()
            for ax,col in zip(axes2,top_cols):
                data=df[col].dropna()
                parts=ax.violinplot(data,positions=[0],showmeans=False,showmedians=False)
                for pc in parts['bodies']: pc.set_facecolor(C2); pc.set_alpha(.35); pc.set_edgecolor(C2)
                parts['cbars'].set_color(GRID); parts['cmaxes'].set_color(GRID); parts['cmins'].set_color(GRID)
                bp=ax.boxplot(data,positions=[0],patch_artist=True,
                    boxprops=dict(facecolor=(0,229/255,160/255,.12),color=C1),
                    medianprops=dict(color=C1,linewidth=2.5),
                    whiskerprops=dict(color='#374151',lw=1.2),capprops=dict(color='#374151',lw=1.2),
                    flierprops=dict(marker='o',markerfacecolor=C5,markersize=3.5,alpha=.7,markeredgewidth=0))
                Q1,Q3=data.quantile(.25),data.quantile(.75); IQR=Q3-Q1
                ax.axhline(Q3+1.5*IQR,color=C5,lw=.8,ls='--',alpha=.6); ax.axhline(Q1-1.5*IQR,color=C5,lw=.8,ls='--',alpha=.6)
                ax.text(0.5,0.96,f'σ={data.std():.2f}',transform=ax.transAxes,ha='center',va='top',fontsize=6.5,color='#6b7280')
                ax.text(0.5,0.88,f'μ={data.mean():.2f}',transform=ax.transAxes,ha='center',va='top',fontsize=6.5,color=C1)
                polish(ax,title=col[:14]); ax.set_xticks([])
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            # Isolation forest
            if len(feat_num)>=2:
                try:
                    iso=IsolationForest(contamination=0.05,random_state=42)
                    preds=iso.fit_predict(df[feat_num[:8]].fillna(df[feat_num[:8]].median()))
                    n_anom=int((preds==-1).sum())
                    st.markdown(f'<span class="chip r">🌲 Isolation Forest: {n_anom} anomalies ({n_anom/len(df)*100:.1f}%)</span>',unsafe_allow_html=True)
                    outlier_report['isolation_forest']=n_anom
                    # Scatter of anomalies on first 2 dims
                    if len(feat_num)>=2:
                        fig3,ax3=make_fig(7,3.5)
                        normal=preds==1; anom=preds==-1
                        glow_scatter(ax3,df[feat_num[0]][normal],df[feat_num[1]][normal],color=C1,s=14,label=f'Normal ({normal.sum():,})')
                        glow_scatter(ax3,df[feat_num[0]][anom],df[feat_num[1]][anom],color=C5,s=22,label=f'Anomaly ({n_anom})')
                        polish(ax3,xlabel=feat_num[0],ylabel=feat_num[1],
                               title='Isolation Forest — Anomaly Map',legend=True)
                        plt.tight_layout(); st.pyplot(fig3); plt.close()
                except: pass

            remove_out=st.toggle("Remove Z-score outliers (|z|>3) before training",value=False)
            if remove_out:
                df_cl=df.copy()
                for col in feat_num:
                    z=np.abs(stats.zscore(df_cl[col].fillna(df_cl[col].median())))
                    df_cl=df_cl[z<=3]
                st.session_state.df_clean=df_cl
                st.markdown(f'<span class="chip g">✓ {len(df)-len(df_cl)} rows removed → {len(df_cl):,} remaining</span>',unsafe_allow_html=True)
            else: st.session_state.df_clean=df.copy()

        st.session_state.outlier_report=outlier_report
        ca,cb=st.columns(2)
        with ca:
            if st.button("← Back"): nav(0)
        with cb:
            if st.button("Continue to EDA →"): nav(2)
        st.markdown('</div>',unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════════

    except Exception as _e:
        import traceback, gc; gc.collect(); plt.close('all')
        st.error(f"⚠️ Step error — {_e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())

# STEP 2 — EDA + WORD CLOUD + TIME SERIES
# ══════════════════════════════════════════════════════════════════════════
elif step==2:
    plt.close('all')
    try:
        df=st.session_state.df_clean; target=st.session_state.target
        is_ts=st.session_state.is_time_series; date_col=st.session_state.date_col
        num_cols=df.select_dtypes(include=np.number).columns.tolist()
        cat_cols=df.select_dtypes(include='object').columns.tolist()
        txt_det=[c for c in cat_cols if c!=target and df[c].dropna().apply(lambda x:len(str(x).split())).mean()>5]
        st.session_state.text_cols_detected=txt_det
        st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 3 — EDA, Distributions & Word Cloud</div>',unsafe_allow_html=True)

        # ── Summary stats ─────────────────────────────────────────────────────
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**📊 Numeric Summary**")
            if num_cols: st.dataframe(df[num_cols].describe().round(2),use_container_width=True)
        with c2:
            st.markdown("**🔎 Data Profile**")
            profile=pd.DataFrame({'Dtype':df.dtypes.astype(str),'Non-Null':df.count(),
                'Null':df.isnull().sum(),'Unique':df.nunique(),'Null%':(df.isnull().mean()*100).round(1)})
            st.dataframe(profile,use_container_width=True)

        # ── Target distribution with beautiful bar + donut ─────────────────
        if target:
            st.markdown('<div class="sep"></div>',unsafe_allow_html=True)
            st.markdown("**🎯 Target Distribution**")
            if df[target].dtype=='object' or df[target].nunique()<=20:
                counts=df[target].value_counts()
                fig,axes=make_figs(1,2,w=12,h=4)
                # Gradient bar chart
                ax=axes[0]
                colors_bar=[PAL[i%len(PAL)] for i in range(len(counts))]
                for i,(label,val) in enumerate(counts.items()):
                    bar=ax.bar(str(label),val,color=colors_bar[i],alpha=.85,zorder=3,width=.65)
                    ax.text(i,val+max(1,val*.02),f'{val:,}',ha='center',va='bottom',
                            color='#9ca3af',fontsize=8,fontweight='bold')
                    ax.text(i,val/2,f'{val/counts.sum()*100:.1f}%',ha='center',va='center',
                            color='white',fontsize=8,fontweight='bold',alpha=.9)
                polish(ax,ylabel='Count',title='Class Counts',grid='y')
                ax.tick_params(axis='x',colors='#6b7280',labelsize=8,rotation=15)
                # Donut
                ax2=axes[1]; ax2.set_facecolor(BG)
                wedges,_,autotexts=ax2.pie(counts.values,labels=None,autopct='%1.1f%%',
                    colors=colors_bar,startangle=90,
                    wedgeprops=dict(width=0.55,edgecolor=BG,linewidth=3),
                    pctdistance=0.75,textprops=dict(fontsize=8))
                for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
                ax2.legend(counts.index.astype(str),loc='center left',bbox_to_anchor=(1,.5),
                           fontsize=8,labelcolor='#9ca3af',facecolor=PANEL,edgecolor=GRID,framealpha=.9)
                ax2.set_title('Class Distribution',color='#9ca3af',fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                fig,ax=make_fig(9,3.5)
                n,bins,patches=ax.hist(df[target].dropna(),bins=40,color=C1,edgecolor=BG,linewidth=.4,alpha=.85,zorder=3)
                for i,patch in enumerate(patches):
                    patch.set_facecolor(PAL[int(i/len(patches)*len(PAL))%len(PAL)])
                    patch.set_alpha(.8)
                mv=df[target].mean(); md=df[target].median()
                ax.axvline(mv,color='#fbbf24',lw=2,ls='--',zorder=4,label=f'Mean {mv:.2f}')
                ax.axvline(md,color=C3,lw=2,ls=':',zorder=4,label=f'Median {md:.2f}')
                grad_fill(ax,bins[:-1],n,C1)
                polish(ax,xlabel=target,ylabel='Count',title=f'{target} Distribution',legend=True)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Correlation heatmap (styled) ──────────────────────────────────────
        if len(num_cols)>1:
            st.markdown('<div class="sep"></div>',unsafe_allow_html=True)
            st.markdown("**🔥 Correlation Heatmap**")
            fig2,ax2=make_fig(11,5)
            corr=df[num_cols].corr()
            mask=np.triu(np.ones_like(corr,dtype=bool))
            corr_masked=corr.where(~mask)
            cmap_custom=mcolors.LinearSegmentedColormap.from_list('cc',['#ef4444','#0b0f1a','#00e5a0'])
            im=ax2.imshow(corr_masked,cmap=cmap_custom,vmin=-1,vmax=1,aspect='auto')
            plt.colorbar(im,ax=ax2,shrink=.7)
            if len(num_cols)<=14:
                for i in range(len(num_cols)):
                    for j in range(len(num_cols)):
                        if not mask[i,j]:
                            ax2.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',
                                     fontsize=6.5,color='#d4d8f0')
            ax2.set_xticks(range(len(num_cols))); ax2.set_xticklabels(num_cols,rotation=35,ha='right',fontsize=7,color='#6b7280')
            ax2.set_yticks(range(len(num_cols))); ax2.set_yticklabels(num_cols,fontsize=7,color='#6b7280')
            for sp in ax2.spines.values(): sp.set_color(GRID)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── Feature distributions — violin + kde ─────────────────────────────
        plot_cols=[c for c in num_cols if c!=target][:8]
        if plot_cols:
            st.markdown('<div class="sep"></div>',unsafe_allow_html=True)
            st.markdown("**📈 Feature Distributions — Violin + KDE**")
            ncols=min(4,len(plot_cols)); nrows=(len(plot_cols)+ncols-1)//ncols
            fig3,axes3=make_figs(nrows,ncols,w=min(16,ncols*4),h=nrows*3.2)
            axes3=np.array(axes3).flatten()
            for i,col in enumerate(plot_cols):
                ax3=axes3[i]; data=df[col].dropna(); color=PAL[i%len(PAL)]
                # Skip if not enough data for violin
                if len(data) < 3 or data.nunique() < 2:
                    ax3.text(0.5,0.5,f'{col}\n(insufficient data)',ha='center',va='center',
                             color='#4b5563',fontsize=8,transform=ax3.transAxes)
                    ax3.set_facecolor(PANEL); polish(ax3,title=col)
                    continue
                try:
                    # Violin + box
                    parts=ax3.violinplot(data,positions=[0],showmeans=False,showmedians=False)
                    for pc in parts['bodies']:
                        pc.set_facecolor(color); pc.set_alpha(.25); pc.set_edgecolor(color); pc.set_linewidth(1.2)
                    for k in ['cbars','cmaxes','cmins']: parts[k].set_color(GRID)
                except Exception:
                    pass
                try:
                    ax3.boxplot(data,positions=[0],patch_artist=True,widths=0.13,
                        boxprops=dict(facecolor=(1,1,1,.08),color=color,linewidth=1.5),
                        medianprops=dict(color='white',lw=2.5),whiskerprops=dict(color='#374151',lw=1),
                        capprops=dict(color='#374151',lw=1),showfliers=False)
                except Exception:
                    pass
                # Mean dot with glow
                mn=data.mean()
                for si,a in [(120,.04),(60,.1),(20,.5)]:
                    ax3.scatter([0],[mn],color=color,s=si,alpha=a,zorder=5,edgecolors='none')
                ax3.scatter([0],[mn],color='white',s=12,zorder=6,edgecolors=color,linewidths=1.2)
                # Stat annotations
                ax3.text(0.5,0.97,f'μ = {data.mean():.2f}',transform=ax3.transAxes,
                         ha='center',va='top',fontsize=6.5,color=color,fontweight='bold')
                ax3.text(0.5,0.90,f'σ = {data.std():.2f}',transform=ax3.transAxes,
                         ha='center',va='top',fontsize=6.5,color='#6b7280')
                ax3.text(0.5,0.83,f'n = {len(data):,}',transform=ax3.transAxes,
                         ha='center',va='top',fontsize=6.5,color='#374151')
            # Hide unused axes
            for j in range(len(plot_cols),len(axes3)): axes3[j].set_visible(False)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

        # ── Categorical columns ───────────────────────────────────────────────
        short_cats=[c for c in cat_cols if c!=target and c not in txt_det and df[c].nunique()<=15][:4]
        if short_cats:
            st.markdown('<div class="sep"></div>',unsafe_allow_html=True)
            st.markdown("**📦 Categorical Columns**")
            ncols=min(2,len(short_cats)); nrows=(len(short_cats)+ncols-1)//ncols
            fig4,axes4=make_figs(nrows,ncols,w=min(13,ncols*6.5),h=nrows*3.5)
            axes4=np.array(axes4).flatten()
            for i,(ax4,col) in enumerate(zip(axes4,short_cats)):
                counts=df[col].value_counts()[:12]
                color=PAL[i%len(PAL)]
                cmap_bar=mcolors.LinearSegmentedColormap.from_list('cb',['#151c2c',color])
                bar_colors=[cmap_bar(j/max(1,len(counts)-1)) for j in range(len(counts))]
                bars=ax4.barh(counts.index.astype(str)[::-1],counts.values[::-1],
                              color=bar_colors[::-1],height=0.65,zorder=3)
                for bar,val in zip(bars,counts.values[::-1]):
                    ax4.text(val+max(.1,val*.01),bar.get_y()+bar.get_height()/2,
                             f'{val:,}  ({val/len(df)*100:.1f}%)',va='center',ha='left',color='#9ca3af',fontsize=7)
                polish(ax4,xlabel='Count',title=col[:20],grid='x')
            for j in range(len(short_cats),len(axes4)): axes4[j].set_visible(False)
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        # ── Time Series EDA ───────────────────────────────────────────────────
        if is_ts and date_col and date_col in df.columns:
            st.markdown('<div class="sep"></div>',unsafe_allow_html=True)
            st.markdown("**📈 Time Series Analysis**")
            try:
                df_ts=df.copy(); df_ts[date_col]=pd.to_datetime(df_ts[date_col],errors='coerce')
                df_ts=df_ts.dropna(subset=[date_col]).sort_values(date_col)
                ts_col=target if (target and target in df_ts.columns) else ([c for c in num_cols if c!=target]+[None])[0]
                if ts_col:
                    roll=max(2,min(30,len(df_ts)//8))
                    df_ts['ma']=df_ts[ts_col].rolling(roll).mean()
                    df_ts['std']=df_ts[ts_col].rolling(roll).std()
                    fig_ts,ax_ts=make_fig(11,4.2)
                    glow_line(ax_ts,df_ts[date_col],df_ts[ts_col],color='#1f2937',lw=.9,label='Actual')
                    glow_line(ax_ts,df_ts[date_col],df_ts['ma'],color=C1,lw=2.2,label=f'{roll}-period MA')
                    ax_ts.fill_between(df_ts[date_col],df_ts['ma']-df_ts['std'],df_ts['ma']+df_ts['std'],
                                       alpha=.12,color=C1,label='±1 σ')
                    grad_fill(ax_ts,df_ts[date_col],df_ts['ma'],C1,alpha=.15)
                    # Annotate min/max
                    idx_max=df_ts[ts_col].idxmax(); idx_min=df_ts[ts_col].idxmin()
                    ax_ts.annotate(f'Peak {df_ts[ts_col].max():.1f}',
                        xy=(df_ts.loc[idx_max,date_col],df_ts.loc[idx_max,ts_col]),
                        xytext=(0,18),textcoords='offset points',
                        fontsize=7,color=C1,ha='center',
                        arrowprops=dict(arrowstyle='->',color=C1,lw=.8))
                    ax_ts.annotate(f'Low {df_ts[ts_col].min():.1f}',
                        xy=(df_ts.loc[idx_min,date_col],df_ts.loc[idx_min,ts_col]),
                        xytext=(0,-18),textcoords='offset points',
                        fontsize=7,color=C5,ha='center',
                        arrowprops=dict(arrowstyle='->',color=C5,lw=.8))
                    polish(ax_ts,ylabel=ts_col,title=f'{ts_col} — Rolling Mean & Confidence Band',legend=True)
                    plt.tight_layout(); st.pyplot(fig_ts); plt.close()
            except Exception as e: st.warning(f"Time series: {e}")

        # ── Word Cloud ────────────────────────────────────────────────────────
        if txt_det:
            st.markdown('<div class="sep"></div>',unsafe_allow_html=True)
            st.markdown("**☁️ Word Cloud**")
            wc_col=st.selectbox("Text column",txt_det,key='wc_col')
            try:
                from wordcloud import WordCloud; from collections import Counter; import re as re2
                text=' '.join(df[wc_col].dropna().astype(str).tolist())
                wc=WordCloud(width=1100,height=420,background_color=PANEL,
                             colormap='cool',max_words=250,collocations=False,
                             contour_width=0,prefer_horizontal=.85).generate(text)
                fig_wc,ax_wc=plt.subplots(figsize=(11,4.2),facecolor=BG)
                ax_wc.set_facecolor(PANEL); ax_wc.imshow(wc,interpolation='bilinear'); ax_wc.axis('off')
                plt.tight_layout(pad=0); st.pyplot(fig_wc); plt.close()
                # Top words bar
                words=re2.findall(r'\b[a-zA-Z]{3,}\b',text.lower())
                stops={'the','and','for','are','but','not','you','all','can','had','was','one','our','out',
                       'get','has','him','his','how','its','new','now','see','too','use','with','that',
                       'this','from','they','been','have','more','will','than','then','when','your','also'}
                top_w=Counter([w for w in words if w not in stops]).most_common(20)
                if top_w:
                    wl,wc2=zip(*top_w)
                    fig_tw,ax_tw=make_fig(10,3.5)
                    bar_colors=[mcolors.LinearSegmentedColormap.from_list('wg',[C2,C3])(i/len(wl)) for i in range(len(wl))]
                    bars_tw=ax_tw.bar(wl,wc2,color=bar_colors,zorder=3,width=.7)
                    for bar,val in zip(bars_tw,wc2):
                        ax_tw.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(.5,bar.get_height()*.02),
                                   f'{val:,}',ha='center',va='bottom',color='#9ca3af',fontsize=7,fontweight='bold')
                    polish(ax_tw,ylabel='Frequency',title='Top 20 Word Frequencies')
                    ax_tw.tick_params(axis='x',rotation=35,colors='#6b7280',labelsize=7.5)
                    plt.tight_layout(); st.pyplot(fig_tw); plt.close()
            except ImportError: st.info("Add `wordcloud` to requirements.txt")

        ca,cb=st.columns(2)
        with ca:
            if st.button("← Back"): nav(1)
        with cb:
            if st.button("Continue to Preprocessing →"): nav(3)
        st.markdown('</div>',unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════════

    except Exception as _e:
        import traceback, gc; gc.collect(); plt.close('all')
        st.error(f"⚠️ Step error — {_e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())

# STEP 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════
elif step==3:
    plt.close('all')
    try:
        df=st.session_state.df_clean; target=st.session_state.target
        txt_det=st.session_state.text_cols_detected
        st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 4 — Preprocessing</div>',unsafe_allow_html=True)

        nlp_mode=False; text_col=None
        if txt_det:
            st.markdown('<div class="nlpbox"><div style="font-family:Space Mono,monospace;font-size:.62rem;color:#c084fc;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem">🧠 NLP Mode Available</div>'
                        '<div style="font-size:.83rem;color:#6b7280">TF-IDF vectorisation detected for text column.</div></div>',unsafe_allow_html=True)
            nlp_mode=st.toggle("Enable NLP Mode (TF-IDF)",value=True)
            if nlp_mode: text_col=st.selectbox("Text column",txt_det)
        st.session_state.nlp_mode=nlp_mode; st.session_state.text_col=text_col

        # ── Auto Feature Engineering ──────────────────────────────────────
        st.markdown('<div class="sep"></div>',unsafe_allow_html=True)
        st.markdown('<div class="sh">⚙️ Auto Feature Engineering</div>',unsafe_allow_html=True)
        st.markdown('<div style="font-size:.8rem;color:#6b7280;margin-bottom:.6rem">Automatically create new features from existing numeric columns to improve model performance.</div>',unsafe_allow_html=True)

        afe_on=st.toggle("Enable Auto Feature Engineering",value=False)
        afe_new_cols=[]
        if afe_on:
            col1,col2,col3,col4=st.columns(4)
            with col1: do_poly=st.checkbox("Polynomial (degree 2)",value=True)
            with col2: do_interact=st.checkbox("Interaction Terms",value=True)
            with col3: do_log=st.checkbox("Log Transform",value=True)
            with col4: do_bin=st.checkbox("Binning (quartiles)",value=False)

        with st.spinner("Preprocessing..."):
            df_proc=df.copy()
            date_col=st.session_state.date_col
            if date_col and date_col in df_proc.columns: df_proc.drop(columns=[date_col],inplace=True)
            high_miss=[c for c in df_proc.columns if df_proc[c].isnull().mean()>0.5 and c!=target]
            if high_miss: df_proc.drop(columns=high_miss,inplace=True)
            if target:
                if df_proc[target].dtype=='object' or df_proc[target].nunique()<=20:
                    problem_type='classification'; le_t=LabelEncoder()
                    df_proc[target]=le_t.fit_transform(df_proc[target].astype(str))
                else: problem_type='regression'; le_t=None
            else: problem_type='clustering'; le_t=None
            st.session_state.problem_type=problem_type; st.session_state.le_target=le_t
            cat_cols=df_proc.select_dtypes(include='object').columns.tolist()
            non_txt=[c for c in cat_cols if c!=target and c!=text_col]
            for col in non_txt:
                le=LabelEncoder(); df_proc[col]=le.fit_transform(df_proc[col].astype(str))
            num_cols2=[c for c in df_proc.select_dtypes(include=np.number).columns if c!=target]
            if df_proc[num_cols2].isnull().sum().sum()>0:
                imp=SimpleImputer(strategy='median'); df_proc[num_cols2]=imp.fit_transform(df_proc[num_cols2])
            tfidf=None; tfidf_cols=[]
            if nlp_mode and text_col and text_col in df_proc.columns:
                tfidf=TfidfVectorizer(max_features=300,stop_words='english')
                tm=tfidf.fit_transform(df_proc[text_col].fillna('').astype(str))
                tdf=pd.DataFrame(tm.toarray(),columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()],index=df_proc.index)
                df_proc=pd.concat([df_proc.drop(columns=[text_col]),tdf],axis=1); tfidf_cols=list(tdf.columns)

            # ── Apply Auto Feature Engineering ───────────────────────────
            if afe_on and num_cols2:
                top_num=num_cols2[:8]  # cap at 8 cols to avoid explosion
                before=df_proc.shape[1]
                # 1. Polynomial features (x², x³ omitted — just x²)
                if do_poly:
                    for c in top_num:
                        try:
                            df_proc[f'{c}²']=df_proc[c]**2
                            afe_new_cols.append(f'{c}²')
                        except: pass
                # 2. Interaction terms (top pairs)
                if do_interact:
                    pairs=[(top_num[i],top_num[j]) for i in range(len(top_num)) for j in range(i+1,min(i+3,len(top_num)))]
                    for a,b in pairs[:10]:
                        try:
                            df_proc[f'{a}×{b}']=df_proc[a]*df_proc[b]
                            afe_new_cols.append(f'{a}×{b}')
                        except: pass
                # 3. Log transform (positive cols only)
                if do_log:
                    for c in top_num:
                        try:
                            if (df_proc[c]>0).all():
                                df_proc[f'log_{c}']=np.log1p(df_proc[c])
                                afe_new_cols.append(f'log_{c}')
                        except: pass
                # 4. Binning into quartiles
                if do_bin:
                    for c in top_num[:4]:
                        try:
                            df_proc[f'{c}_bin']=pd.qcut(df_proc[c],q=4,labels=False,duplicates='drop')
                            afe_new_cols.append(f'{c}_bin')
                        except: pass
                after=df_proc.shape[1]
                st.success(f"✅ Auto FE: {after-before} new features created ({', '.join(afe_new_cols[:6])}{'...' if len(afe_new_cols)>6 else ''})")

            # ── Auto Feature Selection — drop low-importance features ─────
            dropped_cols=[]
            if afe_on and afe_new_cols and target and problem_type in ('classification','regression'):
                try:
                    from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
                    feat_sel_cols=[c for c in df_proc.columns if c!=target]
                    Xfs=df_proc[feat_sel_cols].fillna(0).values
                    yfs=df_proc[target].values
                    rf_sel=RFC(n_estimators=30,random_state=42,max_depth=6) if problem_type=='classification' \
                           else RFR(n_estimators=30,random_state=42,max_depth=6)
                    rf_sel.fit(Xfs,yfs)
                    importances=rf_sel.feature_importances_
                    imp_df=pd.DataFrame({'feature':feat_sel_cols,'importance':importances}).sort_values('importance',ascending=False)
                    # Drop features below threshold (mean × 0.1) but keep ALL original cols — only drop AFE-generated ones
                    threshold=imp_df['importance'].mean()*0.1
                    low_imp_afe=[r['feature'] for _,r in imp_df.iterrows()
                                 if r['importance']<threshold and r['feature'] in afe_new_cols]
                    if low_imp_afe:
                        df_proc.drop(columns=low_imp_afe,inplace=True)
                        dropped_cols=low_imp_afe
                        kept=[c for c in afe_new_cols if c not in dropped_cols]
                        st.info(f"🗑️ Feature Selection: dropped {len(dropped_cols)} low-importance AFE features "
                                f"({', '.join(dropped_cols[:5])}{'...' if len(dropped_cols)>5 else ''}). "
                                f"Kept {len(kept)} useful ones.")

                        # Show top features chart
                        top_n=min(15,len(imp_df))
                        top_imp=imp_df[~imp_df['feature'].isin(dropped_cols)].head(top_n)
                        fig_fs,ax_fs=make_fig(9,3.5)
                        cmap_fs=mcolors.LinearSegmentedColormap.from_list('fi',[C2,C1])
                        bar_cols_fs=[cmap_fs(i/max(1,top_n-1)) for i in range(len(top_imp))]
                        bars=ax_fs.barh(range(len(top_imp)),top_imp['importance'].values,color=bar_cols_fs,zorder=3)
                        ax_fs.set_yticks(range(len(top_imp)))
                        ax_fs.set_yticklabels(top_imp['feature'].values,fontsize=7.5,color='#9ca3af')
                        ax_fs.invert_yaxis()
                        for bar,val in zip(bars,top_imp['importance'].values):
                            ax_fs.text(val+0.001,bar.get_y()+bar.get_height()/2,
                                      f'{val:.4f}',va='center',fontsize=7,color='#6b7280')
                        polish(ax_fs,title=f'Top {top_n} Features After Selection',xlabel='Importance')
                        plt.tight_layout(); st.pyplot(fig_fs); plt.close()
                    else:
                        st.success("✅ Feature Selection: all AFE features passed importance threshold — none dropped.")
                except Exception as fs_err:
                    st.warning(f"Feature selection skipped: {fs_err}")

            st.session_state.tfidf=tfidf; st.session_state.df_proc=df_proc

        # Preprocessing pipeline visual
        steps_done=[f'Problem: {problem_type.title()}',f'{len(non_txt)} cats encoded',
                    'Missing imputed',f'{df_proc.shape[1]-(1 if target else 0)} features']
        if high_miss: steps_done.append(f'{len(high_miss)} cols dropped')
        if nlp_mode and text_col: steps_done.append(f'TF-IDF {len(tfidf_cols)} feats')
        if afe_on and afe_new_cols: steps_done.append(f'AFE +{len(afe_new_cols)} feats')
        if dropped_cols: steps_done.append(f'Selection −{len(dropped_cols)} dropped')

        fig_pp,ax_pp=plt.subplots(figsize=(10,1.8),facecolor=BG); ax_pp.set_facecolor(BG); ax_pp.axis('off')
        n=len(steps_done); xs=np.linspace(.05,.95,n)
        for i,(x,s) in enumerate(zip(xs,steps_done)):
            col=C1 if i<4 else C5 if 'dropped' in s else C3
            ax_pp.add_patch(FancyBboxPatch((x-.09,.2),.18,.6,boxstyle="round,pad=0.02",
                                           facecolor=(0,229/255,160/255,.08),edgecolor=col,lw=1.2))
            ax_pp.text(x,.5,s,ha='center',va='center',fontsize=7,color='#d4d8f0',fontweight='500',wrap=True)
            if i<n-1: ax_pp.annotate('',xy=(xs[i+1]-.09,.5),xytext=(x+.09,.5),
                arrowprops=dict(arrowstyle='->',color='#374151',lw=1.2))
        plt.tight_layout(); st.pyplot(fig_pp); plt.close()

        st.dataframe(df_proc.head(),use_container_width=True)
        ca,cb=st.columns(2)
        with ca:
            if st.button("← Back"): nav(2)
        with cb:
            if st.button("Continue to Training →"): nav(4)
        st.markdown('</div>',unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════════

    except Exception as _e:
        import traceback, gc; gc.collect(); plt.close('all')
        st.error(f"⚠️ Step error — {_e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())

# STEP 4 — TRAIN
# ══════════════════════════════════════════════════════════════════════════
elif step==4:
    plt.close('all')
    df_proc=st.session_state.df_proc; target=st.session_state.target
    problem_type=st.session_state.problem_type; is_ts=st.session_state.is_time_series
    st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 5 — Training All Models</div>',unsafe_allow_html=True)

    if df_proc is None:
        st.error("⚠️ Preprocessing data not found. Please go back and re-run preprocessing.")
        if st.button("← Back to Preprocessing"): nav(3)
        st.stop()

    feat_cols=[c for c in df_proc.columns if c!=target]
    X=df_proc[feat_cols].values; y=df_proc[target].values if target else None
    scaler=StandardScaler(); X_sc=scaler.fit_transform(X)
    st.session_state.scaler=scaler; st.session_state.feat_cols=feat_cols

    enable_tuning=st.toggle("🔧 Enable Hyperparameter Tuning (GridSearchCV)",value=True)

    # Check if already trained
    already_trained=(st.session_state.results is not None or
                     len(st.session_state.cluster_results)>0)

    if already_trained:
        # Show cached result — don't re-run training
        if st.session_state.results is not None and len(st.session_state.results):
            best_row=st.session_state.results.iloc[0]
            sc=st.session_state.sort_col
            st.success(f"✅ Training complete! Best: **{st.session_state.best_name}** — {sc}: **{best_row[sc]:.4f}**")
        else:
            st.success("✅ Clustering complete!")
        ca,cb,cc=st.columns(3)
        with ca:
            if st.button("← Back"): nav(3)
        with cb:
            if st.button("🔄 Re-train"):
                st.session_state.results=None
                st.session_state.cluster_results=[]
                st.session_state.best_model=None
                st.session_state.best_name=''
                st.rerun()
        with cc:
            if st.button("View Results →"): nav(5)
        st.markdown('</div>',unsafe_allow_html=True)
        st.stop()

    # Not yet trained — show Start Training button
    col_btn1,col_btn2=st.columns(2)
    with col_btn1:
        go=st.button("🚀 Start Training",use_container_width=True)
    with col_btn2:
        if st.button("← Back",use_container_width=True): nav(3)

    if not go:
        st.markdown('</div>',unsafe_allow_html=True)
        st.stop()

    # ── Training runs only when Start Training clicked ───────────────────
    results=[]; tuning_results={}

    if problem_type!='clustering':
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        Xtr_sc=scaler.transform(X_train); Xte_sc=scaler.transform(X_test)
        st.session_state.X_test=X_test; st.session_state.X_test_sc=Xte_sc
        st.session_state.y_test=y_test

    if problem_type=='classification':
        base_models={"Random Forest":(RandomForestClassifier(n_estimators=50,random_state=42,max_depth=12),False),
                     "Gradient Boosting":(GradientBoostingClassifier(n_estimators=50,random_state=42),False),
                     "Logistic Regression":(LogisticRegression(max_iter=500),True),
                     "KNN":(KNeighborsClassifier(),True),
                     "Decision Tree":(DecisionTreeClassifier(random_state=42,max_depth=10),False)}
        boost_models={}
        try: from xgboost import XGBClassifier; boost_models["XGBoost"]=(XGBClassifier(n_estimators=50,random_state=42,verbosity=0,eval_metric='logloss'),False)
        except: pass
        try: from lightgbm import LGBMClassifier; boost_models["LightGBM"]=(LGBMClassifier(n_estimators=50,random_state=42,verbose=-1),False)
        except: pass
        try: from catboost import CatBoostClassifier; boost_models["CatBoost"]=(CatBoostClassifier(iterations=50,random_seed=42,verbose=0),False)
        except: pass
        ml_models={**base_models,**boost_models}
        param_grids={"Random Forest":{'n_estimators':[50,100],'max_depth':[None,10]},
                     "Logistic Regression":{'C':[0.1,1,10]}}
        metric_name="Accuracy"; sort_col="Accuracy"
    elif problem_type=='regression':
        base_models={"Random Forest":(RandomForestRegressor(n_estimators=50,random_state=42,max_depth=12),False),
                     "Gradient Boosting":(GradientBoostingRegressor(n_estimators=50,random_state=42),False),
                     "Linear Regression":(LinearRegression(),True),
                     "Ridge":(Ridge(),True),
                     "KNN":(KNeighborsRegressor(),True),
                     "Decision Tree":(DecisionTreeRegressor(random_state=42,max_depth=10),False)}
        boost_models={}
        try: from xgboost import XGBRegressor; boost_models["XGBoost"]=(XGBRegressor(n_estimators=50,random_state=42,verbosity=0),False)
        except: pass
        try: from lightgbm import LGBMRegressor; boost_models["LightGBM"]=(LGBMRegressor(n_estimators=50,random_state=42,verbose=-1),False)
        except: pass
        try: from catboost import CatBoostRegressor; boost_models["CatBoost"]=(CatBoostRegressor(iterations=50,random_seed=42,verbose=0),False)
        except: pass
        ml_models={**base_models,**boost_models}
        param_grids={"Random Forest":{'n_estimators':[50,100],'max_depth':[None,10]},
                     "Ridge":{'alpha':[0.1,1,10,100]}}
        metric_name="R² Score"; sort_col="R² Score"
    else:
        ml_models={}; param_grids={}; metric_name="Silhouette"; sort_col="Silhouette"

    st.session_state.metric_name=metric_name; st.session_state.sort_col=sort_col
    boost_names={"XGBoost","LightGBM","CatBoost"}
    total=len(ml_models)+8; progress=st.progress(0); status=st.empty(); done=0

    for name,(model,use_sc) in ml_models.items():
        is_boost=name in boost_names
        emoji='🚀' if is_boost else '🤖'
        status.markdown(f'<div style="font-size:.85rem;color:{"#f87171" if is_boost else "#6b7280"};">{emoji} Training {name}...</div>',unsafe_allow_html=True)
        Xtr=Xtr_sc if use_sc else X_train; Xte=Xte_sc if use_sc else X_test
        if enable_tuning and name in param_grids:
            status.markdown(f'<div style="font-size:.85rem;color:#fbbf24;">🔧 Tuning {name}...</div>',unsafe_allow_html=True)
            gs=GridSearchCV(model,param_grids[name],cv=3,scoring='accuracy' if problem_type=='classification' else 'r2',n_jobs=-1)
            gs.fit(Xtr,y_train); model=gs.best_estimator_
            tuning_results[name]={'best_params':gs.best_params_,'best_score':round(gs.best_score_,4)}
        try:
            model.fit(Xtr,y_train); yp=model.predict(Xte); is_tuned=name in tuning_results
            if problem_type=='classification':
                results.append({'Model':name,'Accuracy':round(accuracy_score(y_test,yp),4),
                    'F1':round(f1_score(y_test,yp,average='weighted',zero_division=0),4),
                    'Tuned':'✓' if is_tuned else '—','Boost':'✓' if is_boost else '—',
                    '_model':model,'_scaled':use_sc,'_tuned':is_tuned,'_boost':is_boost})
            else:
                results.append({'Model':name,'R² Score':round(r2_score(y_test,yp),4),
                    'RMSE':round(np.sqrt(mean_squared_error(y_test,yp)),4),
                    'Tuned':'✓' if is_tuned else '—','Boost':'✓' if is_boost else '—',
                    '_model':model,'_scaled':use_sc,'_tuned':is_tuned,'_boost':is_boost})
        except: pass
        done+=1; progress.progress(done/total)

    st.session_state.tuning_results=tuning_results

    # Time series ARIMA
    ts_results={}
    if is_ts and target and problem_type=='regression':
        status.markdown('<div style="font-size:.85rem;color:#22d3ee;">📈 ARIMA...</div>',unsafe_allow_html=True)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            df_orig=st.session_state.df_clean; dc=st.session_state.date_col
            dft=df_orig.copy(); dft[dc]=pd.to_datetime(dft[dc],errors='coerce')
            dft=dft.dropna(subset=[dc]).sort_values(dc).set_index(dc)
            ts_s=dft[target].dropna(); sp=int(len(ts_s)*.8)
            tr_ts,te_ts=ts_s[:sp],ts_s[sp:]
            fit_ar=ARIMA(tr_ts,order=(2,1,2)).fit()
            fc=fit_ar.forecast(steps=len(te_ts))
            ts_results={'r2':round(r2_score(te_ts,fc),4),'rmse':round(np.sqrt(mean_squared_error(te_ts,fc)),4),
                        'forecast':fc,'test':te_ts,'train':tr_ts}
            results.append({'Model':'ARIMA(2,1,2)','R² Score':ts_results['r2'],'RMSE':ts_results['rmse'],
                'Tuned':'—','Boost':'—','_model':fit_ar,'_scaled':False,'_tuned':False,'_boost':False})
        except Exception as e: st.warning(f"ARIMA: {e}")
    st.session_state.ts_results=ts_results; done+=1; progress.progress(done/total)

    # Clustering
    n_cl=min(5,max(2,len(X_sc)//50)); cluster_results=[]
    # Cap data for clustering/DR to save memory
    MAX_CLUSTER=min(2000,len(X_sc))
    idx_cl=np.random.choice(len(X_sc),MAX_CLUSTER,replace=False) if len(X_sc)>MAX_CLUSTER else np.arange(len(X_sc))
    X_cl=X_sc[idx_cl]

    for cname,cmodel in [("K-Means",KMeans(n_clusters=n_cl,random_state=42,n_init=5)),
                          ("Gaussian Mixture",GaussianMixture(n_components=n_cl,random_state=42)),
                          ("DBSCAN",DBSCAN(eps=0.5,min_samples=5))]:
        try:
            lbl=cmodel.fit_predict(X_cl); n_u=len(set(lbl)-{-1})
            sil=round(silhouette_score(X_cl,lbl),4) if n_u>=2 else -1
            db=round(davies_bouldin_score(X_cl,lbl),4) if n_u>=2 else 999
            # Only store labels if small enough
            cluster_results.append({'Model':cname,'Silhouette':sil,'DB Index':db,'Clusters':n_u,
                                     '_labels':lbl if len(lbl)<5000 else None})
        except: pass
        done+=1; progress.progress(done/total)
    st.session_state.cluster_results=cluster_results

    # DR — only PCA (lightweight), skip t-SNE
    dr_results=[]; n_comp=min(2,X_cl.shape[1],X_cl.shape[0])
    try:
        pca2=PCA(n_components=n_comp); Xp=pca2.fit_transform(X_cl)
        pf=PCA(n_components=min(20,X_cl.shape[1],X_cl.shape[0])); pf.fit(X_cl)
        dr_results.append({'Method':'PCA','Var(2D)':round(sum(pf.explained_variance_ratio_[:2])*100,2),
                           '_X2d':Xp,'_pca_full':pf})
    except: pass
    try:
        svd=TruncatedSVD(n_components=n_comp,random_state=42); Xs2=svd.fit_transform(X_cl)
        dr_results.append({'Method':'SVD','Var(2D)':round(sum(svd.explained_variance_ratio_)*100,2),'_X2d':Xs2})
    except: pass
    st.session_state.dr_results=dr_results
    done=total; progress.progress(1.0); status.empty()
    # Free large arrays from memory
    import gc
    del X_sc, X_cl
    # Keep df_proc alive until results step reads feat_cols etc.
    # Only clear df_clean (no longer needed)
    st.session_state.df_clean=None
    gc.collect()

    if problem_type!='clustering' and results:
        rdf=pd.DataFrame(results).sort_values(sort_col,ascending=False).reset_index(drop=True)
        st.session_state.results=rdf; best=rdf.iloc[0]
        st.session_state.best_model=best['_model']; st.session_state.best_name=best['Model']
        st.session_state.best_scaled=best['_scaled']

    # Auto-navigate to results — st.rerun needed so session_state is visible
    st.session_state.step=5
    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
# STEP 5 — RESULTS (best visualisations)
# ══════════════════════════════════════════════════════════════════════════
elif step==5:
    plt.close('all')
    try:
        results_df=st.session_state.results; cluster_res=st.session_state.cluster_results
        dr_res=st.session_state.dr_results; feat_cols=st.session_state.feat_cols
        sort_col=st.session_state.sort_col; metric_name=st.session_state.metric_name
        problem_type=st.session_state.problem_type; X_test=st.session_state.X_test
        X_test_sc=st.session_state.X_test_sc; y_test=st.session_state.y_test
        scaler=st.session_state.scaler; df_proc=st.session_state.df_proc
        target=st.session_state.target; tuning_results=st.session_state.tuning_results
        ts_results=st.session_state.ts_results
        # Free df_proc now — feat_cols already extracted above
        if st.session_state.df_proc is not None:
            st.session_state.df_proc=None
            import gc; gc.collect()
        st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 6 — Results & Explainability</div>',unsafe_allow_html=True)

        if results_df is None and len(cluster_res)==0:
            st.warning('⚠️ No results yet — please go back and train first.')
            if st.button("← Go to Training"): nav(4)
            st.stop()
        best_score=results_df.iloc[0][sort_col] if len(results_df) else 0
        best_name=st.session_state.best_name or '—'
        n_boost=sum(1 for _,r in results_df.iterrows() if r.get('_boost')) if results_df is not None else 0

        # ── Gauge cards ───────────────────────────────────────────────────────
        if results_df is not None and len(results_df):
            st.markdown('<div class="sh">📊 Key Performance Gauges</div>',unsafe_allow_html=True)
            n_gauge=min(4,len(results_df))
            fig_g,axes_g=plt.subplots(1,n_gauge,figsize=(n_gauge*3.2,3),facecolor=BG)
            if n_gauge==1: axes_g=[axes_g]
            gauge_cols=[C1,C2,C3,C4]
            for i,(ax_g,(_,row)) in enumerate(zip(axes_g,results_df.head(n_gauge).iterrows())):
                gauge(ax_g,min(row[sort_col],1.0),row['Model'][:15],color=gauge_cols[i%4])
            plt.tight_layout(); st.pyplot(fig_g); plt.close()

        # ── Model comparison chart ────────────────────────────────────────────
        if results_df is not None and len(results_df):
            st.markdown('<div class="sh">🤖 Model Comparison</div>',unsafe_allow_html=True)
            fig_cmp,ax_cmp=make_fig(11,max(4,len(results_df)*0.58))
            m_names=results_df['Model'].tolist()
            m_scores=results_df[sort_col].tolist()
            m_boost=[r.get('_boost',False) for _,r in results_df.iterrows()]
            medals=['🥇','🥈','🥉']+['']*(len(m_names)-3)
            # Per-bar colour: gradient ramp per group
            bar_colors_cmp=[]
            for i,b in enumerate(m_boost):
                if b: bar_colors_cmp.append(mcolors.LinearSegmentedColormap.from_list('b',['#1a0a0a',C5])(0.65+i*.03))
                elif i==0: bar_colors_cmp.append(C1)
                elif i==1: bar_colors_cmp.append(C2)
                else: bar_colors_cmp.append(mcolors.LinearSegmentedColormap.from_list('n',['#111e30','#2a3a55'])(i/max(len(m_names),1)))
            bars_cmp=ax_cmp.barh(range(len(m_names)),m_scores,
                                  color=bar_colors_cmp[::-1],height=0.62,zorder=3)
            max_s=max(m_scores) if m_scores else 1
            for i,(bar,val,name,medal) in enumerate(zip(bars_cmp,m_scores[::-1],m_names[::-1],medals[::-1])):
                # Background track
                ax_cmp.barh(bar.get_y()+bar.get_height()/2,max_s*1.18,
                            height=0.62,left=0,color='#111827',zorder=2,alpha=.4)
                # Score label
                ax_cmp.text(val+max_s*.012, bar.get_y()+bar.get_height()/2,
                            f'{val:.4f}', va='center', ha='left', color='#e2e8f0', fontsize=8.5, fontweight='bold')
                # Name
                ax_cmp.text(max_s*.01, bar.get_y()+bar.get_height()/2,
                            f'{medal} {name}', va='center', ha='left', color='white',
                            fontsize=8, fontweight='600', zorder=4, clip_on=True)
                # % of best bar
                pct=val/max_s*100
                ax_cmp.text(max_s*1.17, bar.get_y()+bar.get_height()/2,
                            f'{pct:.1f}%', va='center', ha='right', color='#4b5563', fontsize=7)
            ax_cmp.set_xlim(0, max_s*1.22)
            ax_cmp.set_yticks([])
            legend_patches=[mpatches.Patch(color=C1,label='🏆 Best'),
                            mpatches.Patch(color=C2,label='2nd place'),
                            mpatches.Patch(color=C5,label='Boosting models')]
            ax_cmp.legend(handles=legend_patches,fontsize=7.5,labelcolor='#9ca3af',
                          facecolor=PANEL,edgecolor=GRID,loc='lower right',framealpha=.95)
            polish(ax_cmp,xlabel=sort_col,grid='x')
            plt.tight_layout(); st.pyplot(fig_cmp); plt.close()

            # ── Radar chart comparing top models ──────────────────────────────
            if len(results_df)>=3 and problem_type in ['classification','regression']:
                st.markdown('<div class="sh">🕸 Multi-Model Radar Comparison (Top 4)</div>',unsafe_allow_html=True)
                metric_cols=[c for c in results_df.columns if not c.startswith('_') and c not in ['Model','Tuned','Boost']]
                numeric_metrics=[c for c in metric_cols if results_df[c].dtype in [np.float64,np.float32,float]]
                if len(numeric_metrics)>=2:
                    top4=results_df.head(4)
                    fig_r,axes_r=plt.subplots(1,len(top4),figsize=(len(top4)*3.5,3.8),facecolor=BG,
                                               subplot_kw=dict(polar=False))
                    axes_r=np.array(axes_r).flatten()
                    for i,(ax_r,(_,row)) in enumerate(zip(axes_r,top4.iterrows())):
                        raw_vals=[float(row[m]) if m in row else 0 for m in numeric_metrics[:6]]
                        # normalise so all on 0-1
                        maxv=[results_df[m].max() for m in numeric_metrics[:6]]
                        minv=[results_df[m].min() for m in numeric_metrics[:6]]
                        norm_vals=[(v-mn)/(mx-mn) if (mx-mn)>0 else 0.5 for v,mx,mn in zip(raw_vals,maxv,minv)]
                        # For RMSE lower is better — invert
                        norm_vals=[1-v if m in ['RMSE','MSE'] else v for v,m in zip(norm_vals,numeric_metrics[:6])]
                        radar(ax_r,numeric_metrics[:6],norm_vals,
                              color=PAL[i%len(PAL)],title=row['Model'][:16])
                    plt.tight_layout(); st.pyplot(fig_r); plt.close()

            if tuning_results:
                with st.expander("🔧 Hyperparameter Tuning Details"):
                    try:
                        for mn,tr in tuning_results.items():
                            st.markdown(f"**{mn}** — CV: `{tr['best_score']}` — Params: `{tr['best_params']}`")

                    except Exception as e:
                        st.warning(f"⚠️ Chart error: {e}")
            best_model=st.session_state.best_model
            Xte=X_test_sc if st.session_state.best_scaled else X_test
            yp=best_model.predict(Xte)

            # ── Classification: styled confusion matrix + classification report ─
            if problem_type=='classification':
                c1,c2=st.columns([1.3,1])
                with c1:
                    with st.expander("🔲 Confusion Matrix",expanded=True):
                        try:
                            le_t=st.session_state.le_target; tn=list(le_t.classes_.astype(str)[:12]) if le_t else None
                            cm=confusion_matrix(y_test,yp)
                            fig_cm,ax_cm=make_fig(6,4.5)
                            draw_confusion(ax_cm,cm,tn)
                            acc=accuracy_score(y_test,yp)
                            ax_cm.set_title(f'Accuracy: {acc:.2%}  |  F1: {f1_score(y_test,yp,average="weighted",zero_division=0):.4f}',
                                            color=C1,fontsize=10,pad=10)
                            ax_cm.set_xlabel('Predicted',color='#4b5563',fontsize=8)
                            ax_cm.set_ylabel('Actual',color='#4b5563',fontsize=8)
                            plt.tight_layout(); st.pyplot(fig_cm); plt.close()
                        except Exception as e:
                            st.warning(f"⚠️ Chart error: {e}")
                with c2:
                    with st.expander("📋 Classification Report",expanded=True):
                        try:
                            rpt=classification_report(y_test,yp,target_names=tn,output_dict=True)
                            rpt_df=pd.DataFrame(rpt).transpose().round(3)
                            st.dataframe(rpt_df,use_container_width=True)
                            # Per-class F1 bar
                            class_rows={k:v for k,v in rpt.items() if k not in ['accuracy','macro avg','weighted avg']}
                            if class_rows:
                                fig_cls,ax_cls=make_fig(4,3)
                                cnames=list(class_rows.keys())[:10]; f1s=[class_rows[c]['f1-score'] for c in cnames]
                                bars_cls=ax_cls.barh(cnames,f1s,color=[C1 if v==max(f1s) else C2 if v>0.7 else '#2a3550' for v in f1s],height=0.55)
                                for bar,val in zip(bars_cls,f1s):
                                    ax_cls.text(val+.01,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',ha='left',color='#9ca3af',fontsize=7)
                                polish(ax_cls,xlabel='F1 Score',title='Per-Class F1',grid='x')
                                plt.tight_layout(); st.pyplot(fig_cls); plt.close()

                        except Exception as e:
                            st.warning(f"⚠️ Chart error: {e}")
            # ── Regression: actual vs predicted + residuals ────────────────────
            else:
                with st.expander("📈 Actual vs Predicted + Residual Analysis",expanded=True):
                    try:
                        fig_r2,axes_r2=make_figs(1,2,w=13,h=5)
                        # ── Scatter: Actual vs Predicted ──────────────────────────
                        ax_s=axes_r2[0]
                        mn2,mx2=min(y_test.min(),yp.min()),max(y_test.max(),yp.max())
                        # Colour by error magnitude
                        err_mag=np.abs(y_test-yp)
                        sc=ax_s.scatter(y_test,yp,c=err_mag,cmap='RdYlGn_r',
                                        s=20,alpha=.7,zorder=3,edgecolors='none')
                        cb=plt.colorbar(sc,ax=ax_s,shrink=.85)
                        cb.set_label('|Error|',color='#6b7280',fontsize=8)
                        cb.ax.yaxis.set_tick_params(color='#4b5563')
                        plt.setp(cb.ax.yaxis.get_ticklabels(),color='#6b7280',fontsize=7)
                        glow_line(ax_s,[mn2,mx2],[mn2,mx2],color='#fbbf24',lw=1.8,label='Perfect fit')
                        # ±10% band
                        ax_s.fill_between([mn2,mx2],[mn2*.9,mx2*.9],[mn2*1.1,mx2*1.1],
                                           alpha=.07,color='#fbbf24',label='±10%')
                        r2=r2_score(y_test,yp); rmse=np.sqrt(mean_squared_error(y_test,yp))
                        ax_s.set_title(f'R² = {r2:.4f}   |   RMSE = {rmse:.4f}',color=C1,fontsize=10,pad=8)
                        # Annotate outlier predictions
                        worst_idx=np.argsort(err_mag)[-3:]
                        for wi in worst_idx:
                            ax_s.annotate(f'{err_mag[wi]:.1f}',xy=(y_test[wi],yp[wi]),
                                xytext=(6,6),textcoords='offset points',fontsize=6.5,color=C5,
                                arrowprops=dict(arrowstyle='->',color=C5,lw=.7))
                        polish(ax_s,xlabel='Actual',ylabel='Predicted',legend=True)
                        # ── Residual distribution ─────────────────────────────────
                        ax_res=axes_r2[1]; residuals=y_test-yp
                        n_r,bins_r,patches_r=ax_res.hist(residuals,bins=35,edgecolor=BG,lw=.3,zorder=3)
                        # Gradient colouring: green near 0, red at extremes
                        mid=len(patches_r)//2
                        for i,p in enumerate(patches_r):
                            dist=abs(i-mid)/max(mid,1)
                            p.set_facecolor(mcolors.LinearSegmentedColormap.from_list('rg',[C1,C5])(dist))
                            p.set_alpha(.85)
                        # Kernel density overlay
                        try:
                            from scipy.stats import gaussian_kde
                            kde_x=np.linspace(residuals.min(),residuals.max(),300)
                            kde_y=gaussian_kde(residuals)(kde_x)
                            kde_y=kde_y/kde_y.max()*n_r.max()
                            glow_line(ax_res,kde_x,kde_y,color=C2,lw=2,label='KDE')
                        except: pass
                        ax_res.axvline(0,color='#fbbf24',lw=2,ls='--',
                                       label=f'μ = {residuals.mean():.3f}',zorder=5)
                        ax_res.axvspan(-residuals.std(),residuals.std(),alpha=.06,color=C1,label='±1σ')
                        ax_res.text(0.98,0.97,f'Skew: {pd.Series(residuals).skew():.3f}\nKurt: {pd.Series(residuals).kurt():.3f}',
                                    transform=ax_res.transAxes,ha='right',va='top',fontsize=7,color='#6b7280')
                        polish(ax_res,xlabel='Residual',ylabel='Count',title='Residual Distribution',legend=True)
                        plt.tight_layout(); st.pyplot(fig_r2); plt.close()

                    except Exception as e:
                        st.warning(f"⚠️ Chart error: {e}")
            # ── Feature importance ─────────────────────────────────────────────
            if hasattr(best_model,'feature_importances_'):
                with st.expander("🔍 Feature Importance — Top 20"):
                    try:
                        fi=pd.Series(best_model.feature_importances_,index=feat_cols).sort_values(ascending=False)[:20]
                        fig_fi,ax_fi=make_fig(10,6)
                        cmap_fi=mcolors.LinearSegmentedColormap.from_list('fi',['#0d1f30','#00e5a0','#3b82f6'])
                        bar_colors_fi=[cmap_fi(i/max(len(fi)-1,1)) for i in range(len(fi))]
                        bars_fi=ax_fi.barh(range(len(fi)),fi.values[::-1],
                                           color=bar_colors_fi,height=0.70,zorder=3)
                        for i,(bar,val) in enumerate(zip(bars_fi,fi.values[::-1])):
                            # Value label
                            ax_fi.text(val+.001,bar.get_y()+bar.get_height()/2,
                                       f'{val:.4f}',va='center',ha='left',color='#e2e8f0',fontsize=7.5,fontweight='bold')
                            # Feature name
                            ax_fi.text(val*.01,bar.get_y()+bar.get_height()/2,
                                       fi.index[::-1][i][:22],va='center',ha='left',
                                       color='white',fontsize=7,clip_on=True,zorder=4)
                        # Cumulative importance line (top axis)
                        cumsum=np.cumsum(fi.values[::-1])/fi.sum()
                        ax2_fi=ax_fi.twiny()
                        glow_line(ax2_fi,cumsum,range(len(fi)),color='#fbbf24',lw=2)
                        ax2_fi.scatter(cumsum,range(len(fi)),color='#fbbf24',s=20,zorder=5,edgecolors='white',lw=.8)
                        ax2_fi.axvline(0.8,color='#374151',lw=1,ls='--',alpha=.6)
                        ax2_fi.text(0.805,len(fi)-1,'80%',fontsize=6.5,color='#4b5563',va='top')
                        ax2_fi.set_xlim(0,1.3); ax2_fi.tick_params(colors='#4b5563',labelsize=7)
                        ax2_fi.set_xlabel('Cumulative Importance',color='#4b5563',fontsize=8)
                        ax_fi.set_yticks([])
                        polish(ax_fi,xlabel='Importance',title=f'Feature Importance — {best_name}  (Top {len(fi)})',grid='x')
                        plt.tight_layout(); st.pyplot(fig_fi); plt.close()

                    except Exception as e:
                        st.warning(f"⚠️ Chart error: {e}")
            # ── SHAP ──────────────────────────────────────────────────────────
            st.markdown('<div class="sep"></div><div class="sh">🔮 SHAP Explainability</div>',unsafe_allow_html=True)
            try:
                import shap
                with st.expander("🔮 SHAP Beeswarm + Bar Plot"):
                    try:
                        with st.spinner("Computing SHAP values..."):
                            sample=min(120,len(Xte)); Xs=Xte[:sample]
                            if hasattr(best_model,'feature_importances_'):
                                explainer=shap.TreeExplainer(best_model); sv=explainer.shap_values(Xs)
                                if isinstance(sv,list): sv=sv[0]
                            else:
                                explainer=shap.KernelExplainer(best_model.predict,shap.sample(Xs,30))
                                sv=explainer.shap_values(Xs,nsamples=50)
                            sv_mean=np.abs(sv).mean(axis=0)
                            top_idx=np.argsort(sv_mean)[-18:]; top_feats=[feat_cols[i] for i in top_idx]; top_vals=sv_mean[top_idx]
                            fig_sh,axes_sh=make_figs(1,2,w=13,h=5.5)
                            # Bar plot
                            ax_sh=axes_sh[0]
                            cmap_sh=mcolors.LinearSegmentedColormap.from_list('sh',[C2,C3])
                            bar_colors_sh=[cmap_sh(i/len(top_feats)) for i in range(len(top_feats))]
                            bars_sh=ax_sh.barh(top_feats,top_vals,color=bar_colors_sh,height=0.65,zorder=3)
                            for bar,val in zip(bars_sh,top_vals):
                                ax_sh.text(val+.001,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',ha='left',color='#9ca3af',fontsize=7)
                            polish(ax_sh,xlabel='Mean |SHAP|',title='Feature Impact (SHAP)',grid='x')
                            # Scatter SHAP for top feature
                            ax_sh2=axes_sh[1]; top_feat_idx=top_idx[-1]; feat_name=feat_cols[top_feat_idx]
                            feat_vals_s=Xs[:,top_feat_idx]; shap_vals_s=sv[:,top_feat_idx]
                            sc=ax_sh2.scatter(feat_vals_s,shap_vals_s,c=feat_vals_s,cmap='plasma',
                                             s=18,alpha=.7,zorder=3,edgecolors='none')
                            plt.colorbar(sc,ax=ax_sh2,label='Feature value',shrink=.8)
                            ax_sh2.axhline(0,color=GRID,lw=1,ls='--')
                            polish(ax_sh2,xlabel=feat_name[:20],ylabel='SHAP value',
                                   title=f'SHAP vs Feature Value — {feat_name[:20]}',grid='both')
                            plt.tight_layout(); st.pyplot(fig_sh); plt.close()
                    except Exception as e:
                        st.warning(f"⚠️ Chart error: {e}")
            except ImportError: st.info("Add `shap` to requirements.txt")

        # ── ARIMA ─────────────────────────────────────────────────────────────
        if ts_results:
            st.markdown('<div class="sep"></div><div class="sh">📈 ARIMA Forecast</div>',unsafe_allow_html=True)
            with st.expander("📈 ARIMA — Train / Actual / Forecast",expanded=True):
                try:
                    fig_ar,ax_ar=make_fig(12,4.8)
                    # Training shaded region
                    ax_ar.fill_between(ts_results['train'].index,ts_results['train'].values,
                                       alpha=.06,color='#374151')
                    glow_line(ax_ar,ts_results['train'].index,ts_results['train'].values,
                              color='#374151',lw=1,label='Train (historical)')
                    glow_line(ax_ar,ts_results['test'].index,ts_results['test'].values,
                              color=C2,lw=2.2,label='Actual (test)')
                    glow_line(ax_ar,ts_results['test'].index,ts_results['forecast'].values,
                              color=C1,lw=2.4,label='ARIMA Forecast')
                    # Forecast-error fill
                    ax_ar.fill_between(ts_results['test'].index,
                                       ts_results['test'].values,ts_results['forecast'].values,
                                       alpha=.12,color=C4,label='Forecast error')
                    # Gradient fill under forecast
                    grad_fill(ax_ar,ts_results['test'].index,ts_results['forecast'].values,C1,alpha=.1)
                    # Vertical divider
                    ax_ar.axvline(ts_results['test'].index[0],color='#4b5563',lw=1.5,ls=':',alpha=.8)
                    ylims=ax_ar.get_ylim()
                    ax_ar.text(ts_results['test'].index[0],ylims[1],
                               '  Forecast →',color='#6b7280',fontsize=8,va='top')
                    # Annotate final forecast point
                    last_fc=ts_results['forecast'].values[-1]
                    ax_ar.annotate(f'Final: {last_fc:.2f}',
                        xy=(ts_results['test'].index[-1],last_fc),
                        xytext=(-30,14),textcoords='offset points',
                        fontsize=7.5,color=C1,fontweight='bold',
                        arrowprops=dict(arrowstyle='->',color=C1,lw=.9))
                    polish(ax_ar,ylabel='Value',
                           title=f"ARIMA(2,1,2) — R²: {ts_results['r2']:.4f}   RMSE: {ts_results['rmse']:.4f}",
                           legend=True)
                    plt.tight_layout(); st.pyplot(fig_ar); plt.close()

                except Exception as e:
                    st.warning(f"⚠️ Chart error: {e}")
        # ── Clustering ────────────────────────────────────────────────────────
        if cluster_res:
            st.markdown('<div class="sep"></div><div class="sh">🔵 Clustering Results</div>',unsafe_allow_html=True)
            valid_cr=[r for r in cluster_res if r['Silhouette']>-1]
            if valid_cr:
                best_cr=max(valid_cr,key=lambda x:x['Silhouette'])
                # Cluster metrics bar
                fig_cl,ax_cl=make_fig(8,3)
                cl_n=[r['Model'] for r in valid_cr]; cl_s=[r['Silhouette'] for r in valid_cr]
                cl_d=[r['DB Index'] for r in valid_cr]
                x_cl=np.arange(len(cl_n)); w_cl=.38
                b1=ax_cl.bar(x_cl-w_cl/2,cl_s,w_cl,color=[C1 if r['Model']==best_cr['Model'] else '#2a3550' for r in valid_cr],label='Silhouette ↑',zorder=3)
                b2=ax_cl.bar(x_cl+w_cl/2,[d/max(cl_d) for d in cl_d],w_cl,color=[C4 if r['Model']==best_cr['Model'] else '#1f2d45' for r in valid_cr],alpha=.7,label='DB Index (normalised) ↓',zorder=3)
                for bar,val in zip(b1,cl_s): ax_cl.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.005,f'{val:.3f}',ha='center',va='bottom',color='#9ca3af',fontsize=7.5,fontweight='bold')
                ax_cl.set_xticks(x_cl); ax_cl.set_xticklabels(cl_n,color='#6b7280',fontsize=8)
                polish(ax_cl,title='Clustering Quality Metrics',legend=True)
                plt.tight_layout(); st.pyplot(fig_cl); plt.close()

                # 2D cluster scatter
                with st.expander(f"🗺 Cluster Map — {best_cr['Model']}",expanded=True):
                    try:
                        # Use pre-computed 2D projection from DR results (computed during training)
                        pca_dr_res=[r for r in dr_res if r.get('Method')=='PCA' and '_X2d' in r]
                        if pca_dr_res:
                            X2d=pca_dr_res[0]['_X2d']
                            lbl=best_cr['_labels']
                            if lbl is not None and len(lbl)==len(X2d):
                                fig_cs,ax_cs=make_fig(8,5.8)
                                for ci in sorted(set(lbl)):
                                    mask=lbl==ci; col=PAL[ci%len(PAL)] if ci>=0 else '#2a3550'; cnt=int(mask.sum())
                                    glow_scatter(ax_cs,X2d[mask,0],X2d[mask,1],color=col,s=16,
                                                 label=f'{"Noise" if ci==-1 else f"Cluster {ci}"} (n={cnt})')
                                    if ci>=0 and mask.sum()>0:
                                        cx,cy=X2d[mask,0].mean(),X2d[mask,1].mean()
                                        ax_cs.scatter([cx],[cy],color='white',s=90,zorder=10,
                                                      edgecolors=col,linewidths=2.5,marker='D')
                                        ax_cs.text(cx,cy,str(ci),ha='center',va='center',
                                                   fontsize=7,color=col,fontweight='bold',zorder=11)
                                polish(ax_cs,xlabel='PCA 1',ylabel='PCA 2',
                                       title=f'{best_cr["Model"]} — Silhouette: {best_cr["Silhouette"]:.4f}',legend=True)
                                plt.tight_layout(); st.pyplot(fig_cs); plt.close()
                            else:
                                st.info("Cluster map unavailable — label/data size mismatch.")
                        else:
                            st.info("Cluster map unavailable — PCA projection not found.")

                    except Exception as e:
                        st.warning(f"⚠️ Chart error: {e}")
                # Elbow chart
                with st.expander("📐 K-Means Elbow Chart"):
                    try:
                        # Use scaled test set as proxy for elbow chart
                        Xa2=X_test_sc if X_test_sc is not None else X_test
                        if Xa2 is not None and len(Xa2)>=4:
                            inertias=[]; ks=range(2,min(11,len(Xa2)//2+2))
                            for k in ks:
                                km=KMeans(n_clusters=k,random_state=42,n_init=5); km.fit(Xa2); inertias.append(km.inertia_)
                            fig_el,ax_el=make_fig(8,3.5)
                            glow_line(ax_el,list(ks),inertias,color=C1,lw=2.2)
                            ax_el.fill_between(list(ks),inertias,alpha=.1,color=C1)
                            for k,v in zip(ks,inertias):
                                ax_el.text(k,v+max(inertias)*.015,f'{v:,.0f}',ha='center',va='bottom',color='#6b7280',fontsize=7)
                            polish(ax_el,xlabel='K (number of clusters)',ylabel='Inertia',title='Elbow Chart — Choose Optimal K')
                            plt.tight_layout(); st.pyplot(fig_el); plt.close()
                        else:
                            st.info("Elbow chart unavailable.")

                    except Exception as e:
                        st.warning(f"⚠️ Chart error: {e}")
        # ── Dimensionality reduction ──────────────────────────────────────────
        if dr_res:
            st.markdown('<div class="sep"></div><div class="sh">🔷 Dimensionality Reduction</div>',unsafe_allow_html=True)
            pca_dr=[r for r in dr_res if r['Method']=='PCA' and '_pca_full' in r]
            if pca_dr:
                with st.expander("📊 PCA Scree Plot + Cumulative Variance"):
                    try:
                        pf=pca_dr[0]['_pca_full']; evr=pf.explained_variance_ratio_*100
                        fig_pca,ax_pca=make_fig(10,4)
                        # Gradient bar colours: purple→teal
                        cmap_pca=mcolors.LinearSegmentedColormap.from_list('pca',[C3,C1,'#06b6d4'])
                        bar_colors_pca=[cmap_pca(i/max(1,len(evr)-1)) for i in range(len(evr))]
                        bars_pca=ax_pca.bar(range(1,len(evr)+1),evr,color=bar_colors_pca,zorder=3,width=0.75)
                        for bar,val in zip(bars_pca,evr):
                            if val>1.5:
                                ax_pca.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.4,
                                            f'{val:.1f}%',ha='center',va='bottom',color='#9ca3af',fontsize=6.5,fontweight='bold')
                        # Cumulative glow line
                        ax2_pca=ax_pca.twinx()
                        cum=np.cumsum(evr)
                        glow_line(ax2_pca,range(1,len(evr)+1),cum,color='#fbbf24',lw=2)
                        ax2_pca.scatter(range(1,len(evr)+1),cum,color='#fbbf24',s=22,zorder=5,
                                        edgecolors='white',linewidths=.8)
                        # 80% and 95% threshold lines
                        for thresh,col_t in [(80,'#6b7280'),(95,C5)]:
                            ax2_pca.axhline(thresh,color=col_t,lw=1,ls='--',alpha=.7)
                            idx_thresh=next((i+1 for i,v in enumerate(cum) if v>=thresh),len(cum))
                            ax2_pca.text(idx_thresh+.2,thresh,f'{thresh}% ({idx_thresh} comps)',
                                         fontsize=6.5,color=col_t,va='bottom')
                        ax2_pca.set_ylim(0,115)
                        ax2_pca.tick_params(colors='#4b5563',labelsize=7.5)
                        ax2_pca.set_ylabel('Cumulative Variance %',color='#4b5563',fontsize=8)
                        polish(ax_pca,xlabel='Principal Component',ylabel='Variance Explained %',
                               title='PCA Scree — Variance per Component + Cumulative')
                        plt.tight_layout(); st.pyplot(fig_pca); plt.close()

                    except Exception as e:
                        st.warning(f"⚠️ Chart error: {e}")
            lbl_col=None
            if cluster_res:
                valid2=[r for r in cluster_res if r.get('_labels') is not None and r['Silhouette']>-1]
                if valid2: lbl_col=max(valid2,key=lambda x:x['Silhouette'])['_labels']

            # All DR methods — rich scatter side by side
            dr_plot=[r for r in dr_res if '_X2d' in r]
            if dr_plot:
                st.markdown('<div class="sh">🔷 Projection Maps — All Methods</div>',unsafe_allow_html=True)
                fig_dr,axes_dr=make_figs(1,len(dr_plot),w=len(dr_plot)*5,h=5)
                axes_dr=np.array(axes_dr).flatten()
                cmaps=['plasma','viridis','cool']
                for (ax_dr,r,cm) in zip(axes_dr,dr_plot,cmaps):
                    X2d=r['_X2d']; idx=r.get('_idx',None)
                    c_dr=lbl_col[idx] if lbl_col is not None and idx is not None else (
                         lbl_col[:len(X2d)] if lbl_col is not None else np.zeros(len(X2d)))
                    # Glow scatter: multi-layer
                    for si,a in [(80,.03),(30,.08),(10,.7)]:
                        ax_dr.scatter(X2d[:,0],X2d[:,1],c=c_dr,cmap=cm,
                                      s=si,alpha=a,zorder=2,edgecolors='none')
                    var=r['Var(2D)']
                    var_str=f'{var:.1f}%' if isinstance(var,float) else var
                    polish(ax_dr,xlabel='Dimension 1',ylabel='Dimension 2',
                           title=f'{r["Method"]}  |  Variance: {var_str}',grid='both')
                    # Point count
                    ax_dr.text(0.02,0.97,f'n={len(X2d):,}',transform=ax_dr.transAxes,
                               fontsize=7,color='#4b5563',va='top')
                plt.tight_layout(); st.pyplot(fig_dr); plt.close()

        ca,cb=st.columns(2)
        with ca:
            if st.button("← Back"): nav(4)
        with cb:
            if st.button("Neural Net Visualizer →"): nav(6)
        st.markdown('</div>',unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════════

    except Exception as _e:
        import traceback, gc; gc.collect(); plt.close('all')
        st.error(f"⚠️ Step error — {_e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())

# STEP 6 — NEURAL NETWORK VISUALIZER
# ══════════════════════════════════════════════════════════════════════════
elif step==6:
    plt.close('all')
    try:
        feat_cols=st.session_state.feat_cols; problem_type=st.session_state.problem_type
        target=st.session_state.target
        st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 7 — Neural Network Architecture Visualizer</div>',unsafe_allow_html=True)

        n_features=len(feat_cols)
        # Get n_classes from le_target (stored in session) instead of df_proc
        le_t=st.session_state.le_target
        n_classes=len(le_t.classes_) if (le_t is not None and problem_type=='classification') else 1

        c1,c2,c3=st.columns(3)
        with c1:
            n_layers=st.slider("Hidden Layers",1,6,3)
            activation=st.selectbox("Activation",["ReLU","Tanh","Sigmoid","LeakyReLU","ELU","GELU"])
        with c2:
            layer_sizes=[]
            for i in range(n_layers):
                s=st.number_input(f"Layer {i+1} neurons",min_value=4,max_value=512,
                                  value=max(8,256//(2**i)),step=8,key=f"ls_{i}")
                layer_sizes.append(int(s))
        with c3:
            dropout=st.slider("Dropout rate",0.0,0.5,0.2,0.05)
            batch_norm=st.toggle("Batch Normalization",value=True)
            optimizer=st.selectbox("Optimizer",["Adam","AdamW","SGD","RMSprop","AdaGrad"])
            lr=st.select_slider("Learning Rate",[0.0001,0.001,0.01,0.1],value=0.001)

        all_layers=[min(n_features,10)]+layer_sizes+[n_classes]
        layer_labels=(["Input"]+[f"Dense({s})" for s in layer_sizes]+["Output"])
        n_l=len(all_layers); MAX_NODES=10

        fig_nn=plt.figure(figsize=(max(12,n_l*2.2),7),facecolor=BG)
        ax_nn=fig_nn.add_subplot(111); ax_nn.set_facecolor(BG); ax_nn.axis('off')
        x_pos=np.linspace(.04,.96,n_l)

        for li,(n_nodes,xp,lbl) in enumerate(zip(all_layers,x_pos,layer_labels)):
            is_in=li==0; is_out=li==n_l-1
            col=C1 if is_in else C3 if is_out else C2
            disp=min(n_nodes,MAX_NODES)
            y_pos=np.linspace(.15,.85,disp)

            # Layer background panel
            ax_nn.add_patch(FancyBboxPatch((xp-.055,.1),.11,.82,
                boxstyle="round,pad=0.01",facecolor=(*[c/255 for c in bytes.fromhex(col[1:])],0.05),
                edgecolor=col,lw=.8,alpha=.5))

            # Connections first (behind nodes)
            if li<n_l-1:
                nx=x_pos[li+1]; nd=min(all_layers[li+1],MAX_NODES)
                ny_pos=np.linspace(.15,.85,nd)
                col_next=C1 if li+1==0 else C3 if li+1==n_l-1 else C2
                for yf in y_pos:
                    for yt in ny_pos:
                        ax_nn.plot([xp+.03,nx-.03],[yf,yt],color=GRID,lw=.35,alpha=.5,zorder=1)

            # Nodes
            for yi,y in enumerate(y_pos):
                # Glow
                ax_nn.add_patch(plt.Circle((xp,y),.028,color=col,alpha=.15,zorder=2))
                ax_nn.add_patch(plt.Circle((xp,y),.022,color=col,alpha=.3,zorder=3))
                ax_nn.add_patch(plt.Circle((xp,y),.016,color=col,alpha=.9,zorder=4))
                if disp<=6:
                    ax_nn.text(xp,y,str(yi+1),ha='center',va='center',fontsize=5,color='white',fontweight='bold',zorder=5)

            if n_nodes>MAX_NODES:
                ax_nn.text(xp,.5,'· · ·',ha='center',va='center',fontsize=14,color='#4b5563',zorder=5)

            # Layer label
            ax_nn.text(xp,.07,lbl,ha='center',va='center',fontsize=7,color='#9ca3af',fontweight='600')
            ax_nn.text(xp,.03,f'n={n_nodes}',ha='center',va='center',fontsize=6,color='#4b5563')

            # Batch norm + dropout annotations
            if not is_in and not is_out and batch_norm:
                ax_nn.text(xp,.94,'BN',ha='center',va='center',fontsize=5.5,color='#fbbf24',
                           bbox=dict(boxstyle='round,pad=0.15',facecolor=(245/255,158/255,11/255,.15),edgecolor=(245/255,158/255,11/255,.4),lw=.8))
            if not is_in and not is_out and dropout>0:
                ax_nn.text(xp,.88,f'D{dropout}',ha='center',va='center',fontsize=5.5,color='#f87171',
                           bbox=dict(boxstyle='round,pad=0.15',facecolor=(239/255,68/255,68/255,.1),edgecolor=(239/255,68/255,68/255,.35),lw=.8))

        ax_nn.set_xlim(0,1); ax_nn.set_ylim(0,1)
        arch_str=' → '.join([str(x) for x in all_layers])
        ax_nn.set_title(f'Architecture: {arch_str}   |   {optimizer} (lr={lr})   |   {activation}',
                        color='#9ca3af',fontsize=10,pad=10)
        plt.tight_layout(); st.pyplot(fig_nn); plt.close()

        # Param summary
        total_params=0; arch_rows=[]; prev=n_features
        for i,s in enumerate(layer_sizes):
            p=prev*s+s; total_params+=p
            arch_rows.append({'Layer':f'Dense {i+1}','Neurons':s,'Activation':activation,
                              'Params':f'{p:,}','BN':'✓' if batch_norm else '—',
                              'Dropout':str(dropout) if dropout>0 else '—'})
            prev=s
        op=prev*n_classes+n_classes; total_params+=op
        arch_rows.append({'Layer':'Output','Neurons':n_classes,'Activation':'Softmax/Sigmoid/Linear',
                          'Params':f'{op:,}','BN':'—','Dropout':'—'})
        st.dataframe(pd.DataFrame(arch_rows),use_container_width=True)
        st.markdown(f"""<div class="ibox"><div class="ititle">⚙️ Architecture Stats</div>
          <div style="display:flex;gap:2rem;flex-wrap:wrap">
            <div><div style="font-size:.65rem;color:#4b5563;text-transform:uppercase">Total Parameters</div>
              <div style="font-size:1.3rem;font-weight:700;color:#00e5a0;font-family:Space Mono">{total_params:,}</div></div>
            <div><div style="font-size:.65rem;color:#4b5563;text-transform:uppercase">Hidden Layers</div>
              <div style="font-size:1.3rem;font-weight:700;color:#d4d8f0;font-family:Space Mono">{n_layers}</div></div>
            <div><div style="font-size:.65rem;color:#4b5563;text-transform:uppercase">Optimizer</div>
              <div style="font-size:1.3rem;font-weight:700;color:#d4d8f0;font-family:Space Mono">{optimizer}  lr={lr}</div></div>
            <div><div style="font-size:.65rem;color:#4b5563;text-transform:uppercase">Regularisation</div>
              <div style="font-size:1.3rem;font-weight:700;color:#d4d8f0;font-family:Space Mono">Drop={dropout} {'+ BN' if batch_norm else ''}</div></div>
          </div></div>""",unsafe_allow_html=True)

        with st.expander("💻 Generated Keras Code"):
            try:
                out_act='softmax' if n_classes>2 else 'sigmoid' if problem_type=='classification' else 'linear'
                loss='sparse_categorical_crossentropy' if n_classes>2 else 'binary_crossentropy' if problem_type=='classification' else 'mse'
                code=f"import tensorflow as tf\nfrom tensorflow import keras\n\nmodel = keras.Sequential([\n    keras.layers.Input(shape=({n_features},)),"
                for s in layer_sizes:
                    code+=f"\n    keras.layers.Dense({s}, activation='{activation.lower()}'),"
                    if batch_norm: code+="\n    keras.layers.BatchNormalization(),"
                    if dropout>0: code+=f"\n    keras.layers.Dropout({dropout}),"
                code+=f"\n    keras.layers.Dense({n_classes}, activation='{out_act}')\n])\n\nmodel.compile(\n    optimizer=keras.optimizers.{optimizer}(learning_rate={lr}),\n    loss='{loss}',\n    metrics=[{'accuracy' if problem_type=='classification' else 'mae'}]\n)\nmodel.summary()"
                st.code(code,language='python')

            except Exception as e:
                st.warning(f"⚠️ Chart error: {e}")
        ca,cb=st.columns(2)
        with ca:
            if st.button("← Back"): nav(5)
        with cb:
            if st.button("AI Report →"): nav(7)
        st.markdown('</div>',unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════════

    except Exception as _e:
        import traceback, gc; gc.collect(); plt.close('all')
        st.error(f"⚠️ Step error — {_e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())

# STEP 7 — AI REPORT
# ══════════════════════════════════════════════════════════════════════════
elif step==7:
    plt.close('all')
    results_df=st.session_state.results; cluster_res=st.session_state.cluster_results
    problem_type=st.session_state.problem_type; best_name=st.session_state.best_name
    sort_col=st.session_state.sort_col; tuning_results=st.session_state.tuning_results
    feat_cols=st.session_state.feat_cols; target=st.session_state.target
    outlier_report=st.session_state.outlier_report; df=st.session_state.df
    if results_df is None and not st.session_state.cluster_results: st.warning('⚠️ No results — go back and train first.'); st.button('← Go to Training', on_click=nav, args=(4,)); st.stop()
    best_score=results_df.iloc[0][sort_col] if len(results_df) else 0
    st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 8 — AI Natural Language Report</div>',unsafe_allow_html=True)

    st.markdown("""<div style="background:rgba(168,85,247,.05);border:1px solid rgba(168,85,247,.18);border-radius:14px;padding:1.2rem;margin-bottom:1rem">
      <div style="font-family:Space Mono,monospace;font-size:.62rem;color:#c084fc;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem">🤖 Claude AI Report Generator</div>
      <div style="font-size:.83rem;color:#6b7280">Click below — Claude writes a full professional analysis in plain English covering findings, performance, and recommendations.</div>
    </div>""",unsafe_allow_html=True)

    if st.button("✨ Generate AI Report with Claude"):
        ctx=[f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns. Problem: {problem_type}. Target: {target}."]
        if results_df is not None and len(results_df):
            ctx.append(f"Models: {', '.join(results_df['Model'].tolist())}. Best: {best_name} ({sort_col}={best_score:.4f}).")
            ctx.append(f"Top 3: {results_df[['Model',sort_col]].head(3).to_string(index=False)}.")
        if tuning_results: ctx.append(f"Tuned: {', '.join(tuning_results.keys())}.")
        if outlier_report.get('zscore'): ctx.append(f"Outliers in: {', '.join(outlier_report['zscore'].keys())}.")
        if outlier_report.get('isolation_forest'): ctx.append(f"Isolation Forest: {outlier_report['isolation_forest']} anomalies.")
        if cluster_res:
            valid=[r for r in cluster_res if r['Silhouette']>-1]
            if valid:
                bc=max(valid,key=lambda x:x['Silhouette'])
                ctx.append(f"Best cluster: {bc['Model']} Silhouette={bc['Silhouette']:.4f} K={bc['Clusters']}.")
        ctx.append(f"Top features: {', '.join(feat_cols[:10])}.")
        prompt=f"""You are a senior data scientist writing a professional ML analysis report.

    Analysis summary:
{chr(10).join(ctx)}

Write a structured report with these sections using markdown headers (##):
1. Executive Summary — headline result in 2 sentences
2. Dataset Overview — describe data size, types, quality
3. Key Findings — most important discoveries with specific numbers
4. Model Performance Analysis — why best model won, compare approaches
5. Data Quality Notes — missing values, outliers, preprocessing
6. Actionable Recommendations — 4-5 concrete next steps
7. Risk Factors — overfitting, drift, imbalance, deployment risks

Professional tone. Specific numbers. Prose paragraphs not bullet lists. Under 650 words."""

        with st.spinner("Claude is writing your report..."):
            try:
                import requests
                api_key=st.secrets.get("ANTHROPIC_API_KEY","")
                if not api_key:
                    raise ValueError("No API key")
                resp=requests.post("https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type":"application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={"model":"claude-sonnet-4-5-20251001","max_tokens":1400,
                          "messages":[{"role":"user","content":prompt}]},
                    timeout=30)
                data=resp.json()
                if data.get('content'):
                    ai_text=data['content'][0]['text']
                else:
                    raise ValueError(data.get('error',{}).get('message','No content'))
            except Exception as api_err:
                st.warning(f"⚠️ AI API unavailable ({api_err}) — showing auto-generated report. To enable Claude AI reports, add `ANTHROPIC_API_KEY` to your Streamlit secrets (App settings → Secrets).")
                score_word="excellent" if best_score>.9 else "strong" if best_score>.8 else "moderate" if best_score>.7 else "baseline"
                ai_text=f"""## Executive Summary

This AutoML analysis processed a **{df.shape[0]:,}-row dataset** training {len(results_df) if results_df is not None else 0} models for a **{problem_type}** task. The best-performing model was **{best_name}** with a {score_word} {sort_col} of **{best_score:.4f}**.

## Dataset Overview

The dataset contains {df.shape[1]} columns — {df.select_dtypes(include=np.number).shape[1]} numeric and {df.select_dtypes(include='object').shape[1]} categorical. Total missing values: {df.isnull().sum().sum()}, handled via median/mode imputation during preprocessing.

## Key Findings

**{best_name}** outperformed all alternatives with {sort_col} = **{best_score:.4f}**. {"Boosting algorithms were included — XGBoost, LightGBM and CatBoost — offering ensemble-level gains over standard sklearn models." if any('XGBoost' in r.get('Model','') for _,r in (results_df.iterrows() if results_df is not None else [])) else ""}

## Model Performance Analysis

{"GridSearchCV hyperparameter tuning was applied to " + ', '.join(tuning_results.keys()) + ", improving cross-validated generalisation." if tuning_results else "Models ran with default hyperparameters — enabling tuning may improve results by 2–5%."} The performance spread suggests {"the data has strong predictable signal." if best_score>.85 else "the task has inherent complexity — further feature engineering is recommended."}

## Data Quality Notes

{"Outliers found in: " + ', '.join(outlier_report.get('zscore',{}).keys()) + ". " if outlier_report.get('zscore') else "No significant Z-score outliers. "}{"Isolation Forest flagged " + str(outlier_report.get('isolation_forest',0)) + " anomalies." if outlier_report.get('isolation_forest') else ""}

## Actionable Recommendations

1. Deploy **{best_name}** as the production model — package as a FastAPI endpoint returning real-time predictions.
2. Monitor for feature distribution drift monthly using statistical tests (KS-test, PSI).
3. Collect more labelled examples for underperforming classes to improve recall.
4. Engineer interaction features from the top-importance columns identified by SHAP.
5. Set up automated retraining triggered when accuracy drops >5% from baseline.

## Risk Factors

Watch for overfitting if retraining on small data batches. Ensure production data distribution matches training data before deployment. Class imbalance — if present — can inflate accuracy while hurting minority-class recall. Always monitor F1 per class, not just overall accuracy."""
            st.session_state.ai_report=ai_text
            st.rerun()

    if st.session_state.ai_report:
        report_html=st.session_state.ai_report
        report_html=re.sub(r'\*\*(.+?)\*\*',r'<strong style="color:#e2e8f0">\1</strong>',report_html)
        report_html=re.sub(r'## (.+)',r'<h3>\1</h3>',report_html)
        report_html=re.sub(r'\n\n',r'<br><br>',report_html)
        st.markdown(f'<div class="aibox">{report_html}</div>',unsafe_allow_html=True)
        st.download_button("⬇ Download Report (.txt)",st.session_state.ai_report.encode(),"ai_report.txt","text/plain")

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(6)
    with cb:
        if st.button("Business Insights →"): nav(8)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif step==8:
    plt.close('all')
    problem_type=st.session_state.problem_type; best_name=st.session_state.best_name
    sort_col=st.session_state.sort_col; results_df=st.session_state.results
    target=st.session_state.target; df=st.session_state.df
    if results_df is None and not st.session_state.cluster_results: st.warning('⚠️ No results — go back and train first.'); st.button('← Go to Training', on_click=nav, args=(4,)); st.stop()
    best_score=results_df.iloc[0][sort_col] if len(results_df) else 0
    st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 9 — Business Use Cases & Deployment</div>',unsafe_allow_html=True)

    st.markdown(f"""<div class="ibox"><div class="ititle">✦ Your Model at a Glance</div>
      <div style="display:flex;gap:2rem;flex-wrap:wrap">
        <div><div style="font-size:.64rem;color:#4b5563;text-transform:uppercase">Problem</div>
          <div style="font-size:1.1rem;font-weight:700;color:#00e5a0;font-family:Space Mono">{problem_type.title()}</div></div>
        <div><div style="font-size:.64rem;color:#4b5563;text-transform:uppercase">Best Model</div>
          <div style="font-size:1.1rem;font-weight:700;color:#d4d8f0;font-family:Space Mono">{best_name}</div></div>
        <div><div style="font-size:.64rem;color:#4b5563;text-transform:uppercase">{sort_col}</div>
          <div style="font-size:1.1rem;font-weight:700;color:#00e5a0;font-family:Space Mono">{best_score:.2%}</div></div>
        <div><div style="font-size:.64rem;color:#4b5563;text-transform:uppercase">Dataset</div>
          <div style="font-size:1.1rem;font-weight:700;color:#d4d8f0;font-family:Space Mono">{df.shape[0]:,} × {df.shape[1]}</div></div>
      </div></div>""",unsafe_allow_html=True)

    insights=get_biz(problem_type,best_name,best_score,target)
    html_grid='<div class="biz-grid">'
    for ins in insights:
        tags=''.join([f'<span class="btag">{t}</span>' for t in ins['tags']])
        html_grid+=f"""<div class="bcard">
          <div class="bicon">{ins['icon']}</div>
          <div class="btitle">{ins['title']}</div>
          <div style="display:inline-block;background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.18);
            border-radius:50px;padding:.1rem .5rem;font-size:.64rem;color:#00e5a0;margin-bottom:.5rem">{ins['roi']}</div>
          <div class="bdesc">{ins['desc']}</div><div>{tags}</div>
        </div>"""
    html_grid+='</div>'
    st.markdown(html_grid,unsafe_allow_html=True)

    st.markdown("""<br><div class="ibox"><div class="ititle">🚀 4-Step Deployment Roadmap</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;margin-top:.8rem">
        <div style="background:rgba(0,229,160,.04);border:1px solid rgba(0,229,160,.12);border-radius:12px;padding:1rem">
          <div style="font-size:.7rem;font-weight:700;color:#00e5a0;margin-bottom:.35rem">① Export Model</div>
          <div style="font-size:.78rem;color:#6b7280">Download .pkl → host on AWS S3 / GCP / Azure. Version with MLflow or DVC.</div>
        </div>
        <div style="background:rgba(59,130,246,.04);border:1px solid rgba(59,130,246,.12);border-radius:12px;padding:1rem">
          <div style="font-size:.7rem;font-weight:700;color:#60a5fa;margin-bottom:.35rem">② Build REST API</div>
          <div style="font-size:.78rem;color:#6b7280">FastAPI or Flask endpoint. JSON in → prediction out. Deploy on Railway, Render or Lambda.</div>
        </div>
        <div style="background:rgba(168,85,247,.04);border:1px solid rgba(168,85,247,.12);border-radius:12px;padding:1rem">
          <div style="font-size:.7rem;font-weight:700;color:#c084fc;margin-bottom:.35rem">③ Integrate</div>
          <div style="font-size:.78rem;color:#6b7280">Connect to CRM / app / dashboard via REST. Add Zapier or Make for no-code automation.</div>
        </div>
        <div style="background:rgba(245,158,11,.04);border:1px solid rgba(245,158,11,.12);border-radius:12px;padding:1rem">
          <div style="font-size:.7rem;font-weight:700;color:#fbbf24;margin-bottom:.35rem">④ Monitor & Retrain</div>
          <div style="font-size:.78rem;color:#6b7280">Evidently AI for drift. Monthly retraining. Alert when accuracy drops >5% from baseline.</div>
        </div>
      </div></div>""",unsafe_allow_html=True)

    ca,cb=st.columns(2)
    with ca:
        if st.button("← Back"): nav(7)
    with cb:
        if st.button("Export PDF Report →"): nav(9)
    st.markdown('</div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
# STEP 9 — EXPORT
# ══════════════════════════════════════════════════════════════════════════
elif step==9:
    plt.close('all')
    results_df=st.session_state.results; cluster_res=st.session_state.cluster_results
    best_model=st.session_state.best_model; best_name=st.session_state.best_name
    scaler=st.session_state.scaler; sort_col=st.session_state.sort_col
    problem_type=st.session_state.problem_type; target=st.session_state.target
    tuning_results=st.session_state.tuning_results; outlier_report=st.session_state.outlier_report
    ai_report=st.session_state.ai_report; df=st.session_state.df
    if results_df is None and not st.session_state.cluster_results: st.warning('⚠️ No results — go back and train first.'); st.button('← Go to Training', on_click=nav, args=(4,)); st.stop()
    best_score=results_df.iloc[0][sort_col] if len(results_df) else 0
    st.markdown('<div class="card fadein"><div class="ctitle">⬡ Step 10 — Export & Download</div>',unsafe_allow_html=True)

    st.markdown(f"""<div style="text-align:center;padding:1.5rem 0">
      <div style="font-size:2.8rem;margin-bottom:.6rem">🎉</div>
      <div style="font-family:Space Mono,monospace;font-size:1.05rem;color:#00e5a0;margin-bottom:.3rem">AutoML Pro Complete!</div>
      <div style="font-size:.9rem;color:#4b5563">Best: <strong style="color:#d4d8f0">{best_name}</strong>
      &nbsp;|&nbsp; {sort_col}: <strong style="color:#00e5a0">{best_score:.4f}</strong></div>
    </div>""",unsafe_allow_html=True)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
        from reportlab.lib.units import cm; from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,HRFlowable
        from reportlab.lib.enums import TA_CENTER

        buf_pdf=io.BytesIO()
        doc=SimpleDocTemplate(buf_pdf,pagesize=A4,topMargin=2*cm,bottomMargin=2*cm,leftMargin=2*cm,rightMargin=2*cm)
        sty=getSampleStyleSheet()
        TS=ParagraphStyle('t',parent=sty['Title'],fontSize=20,textColor=colors.HexColor('#00e5a0'),spaceAfter=5,alignment=TA_CENTER)
        H1=ParagraphStyle('h1',parent=sty['Heading1'],fontSize=13,textColor=colors.HexColor('#3b82f6'),spaceBefore=10,spaceAfter=4)
        H2=ParagraphStyle('h2',parent=sty['Heading2'],fontSize=10,textColor=colors.HexColor('#a855f7'),spaceBefore=7,spaceAfter=3)
        BS=ParagraphStyle('b',parent=sty['Normal'],fontSize=9,textColor=colors.HexColor('#374151'),spaceAfter=3,leading=14)
        story=[]
        story.append(Paragraph("AutoML Pro — Complete Analysis Report",TS))
        story.append(Paragraph(f"Problem: {problem_type.title()} | Best: {best_name} | Score: {best_score:.4f}",BS))
        story.append(HRFlowable(width="100%",thickness=1,color=colors.HexColor('#00e5a0')))
        story.append(Spacer(1,.3*cm))
        story.append(Paragraph("1. Dataset Summary",H1))
        ds=[['Metric','Value'],['Rows',f'{df.shape[0]:,}'],['Columns',str(df.shape[1])],
            ['Missing',str(df.isnull().sum().sum())],['Target',str(target) if target else 'None'],
            ['Problem',problem_type.title()],['Best Model',best_name],['Best Score',f'{best_score:.4f}']]
        t=Table(ds,colWidths=[7*cm,8*cm])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#00e5a0')),('FONTSIZE',(0,0),(-1,-1),9),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f9fafb')]),
            ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),5)]))
        story.append(t); story.append(Spacer(1,.3*cm))
        if outlier_report.get('zscore'):
            story.append(Paragraph("2. Outlier Detection",H1))
            od=[['Column','Z-Score Count']]+[[k,str(v)] for k,v in outlier_report['zscore'].items()]
            t2=Table(od,colWidths=[9*cm,6*cm])
            t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#ef4444')),
                ('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),5)]))
            story.append(t2); story.append(Spacer(1,.3*cm))
        if results_df is not None and len(results_df):
            story.append(Paragraph("3. ML Model Comparison",H1))
            disp=[c for c in results_df.columns if not c.startswith('_')]
            ml_data=[disp]+[[str(x) for x in row] for row in results_df[disp].values.tolist()]
            t3=Table(ml_data,colWidths=[17*cm/len(disp)]*len(disp))
            t3.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#00e5a0')),('FONTSIZE',(0,0),(-1,-1),8),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0fdf4')]),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),4)]))
            story.append(t3); story.append(Spacer(1,.3*cm))
        if tuning_results:
            story.append(Paragraph("4. Hyperparameter Tuning",H1))
            for mn,tr in tuning_results.items():
                story.append(Paragraph(f"<b>{mn}</b>: CV={tr['best_score']} | {tr['best_params']}",BS))
            story.append(Spacer(1,.3*cm))
        if cluster_res:
            story.append(Paragraph("5. Clustering Results",H1))
            cd=[['Model','Silhouette','DB Index','Clusters']]+[
                [r['Model'],str(r['Silhouette']),str(r['DB Index']),str(r['Clusters'])] for r in cluster_res]
            t4=Table(cd,colWidths=[6*cm,4*cm,4*cm,3*cm])
            t4.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1117')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#f59e0b')),('FONTSIZE',(0,0),(-1,-1),9),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#fffbeb')]),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),('PADDING',(0,0),(-1,-1),5)]))
            story.append(t4); story.append(Spacer(1,.3*cm))
        story.append(Paragraph("6. Business Use Cases",H1))
        for ins in get_biz(problem_type,best_name,best_score,target):
            story.append(Paragraph(f"{ins['icon']} <b>{ins['title']}</b> — {ins['roi']}",H2))
            story.append(Paragraph(ins['desc'],BS))
        story.append(Spacer(1,.3*cm))
        if ai_report:
            story.append(Paragraph("7. AI Analysis",H1))
            clean=re.sub(r'\*\*(.+?)\*\*',r'\1',ai_report); clean=re.sub(r'## ','',clean)
            for para in clean.split('\n\n'):
                if para.strip(): story.append(Paragraph(para.strip(),BS))
        doc.build(story); buf_pdf.seek(0)
        st.session_state.pdf_buf=buf_pdf.read()
        st.success("✅ PDF Report ready!")
    except ImportError:
        st.warning("Add `reportlab` to requirements.txt"); st.session_state.pdf_buf=None

    c1,c2,c3,c4,c5=st.columns(5)
    with c1:
        if best_model:
            buf=io.BytesIO(); joblib.dump(best_model,buf); buf.seek(0)
            st.download_button("⬇ Model .pkl",buf,f"{best_name.replace(' ','_')}.pkl","application/octet-stream",use_container_width=True)
    with c2:
        buf2=io.BytesIO(); joblib.dump(scaler,buf2); buf2.seek(0)
        st.download_button("⬇ Scaler .pkl",buf2,"scaler.pkl","application/octet-stream",use_container_width=True)
    with c3:
        rows=[]
        if results_df is not None:
            for _,r in results_df.iterrows(): rows.append({k:v for k,v in r.items() if not k.startswith('_')})
        if cluster_res:
            for r in cluster_res: rows.append({k:v for k,v in r.items() if not k.startswith('_')})
        if rows: st.download_button("⬇ Results .csv",pd.DataFrame(rows).to_csv(index=False),"results.csv","text/csv",use_container_width=True)
    with c4:
        if ai_report: st.download_button("⬇ AI Report",ai_report.encode(),"ai_report.txt","text/plain",use_container_width=True)
    with c5:
        if st.session_state.get('pdf_buf'):
            st.download_button("⬇ PDF Report",st.session_state.pdf_buf,"automl_pro_report.pdf","application/pdf",use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🔄 Start Over"):
        for k in list(DEFS.keys()):
            st.session_state[k] = DEFS[k]
        st.session_state.step = 0
        st.rerun()
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin-top:3rem;padding-top:1.4rem;
  border-top:1px solid rgba(255,255,255,.03);font-family:Space Mono,monospace;
  font-size:.58rem;color:#111827;letter-spacing:.1em">
  AUTOML PRO · XGBoost · LightGBM · CatBoost · ARIMA · SHAP · NN Viz · AI Report · Business Insights
</div>""",unsafe_allow_html=True)
