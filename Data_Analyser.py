import os
import glob
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import warnings
warnings.filterwarnings("ignore")

start_total=time.time()

BASE_DIR=os.getcwd()
INPUT_DIR=BASE_DIR
OUTPUT_DIR=os.path.join(BASE_DIR,"Output")
CHART_DIR=os.path.join(OUTPUT_DIR,"charts")
DATA_DIR=os.path.join(OUTPUT_DIR,"data")
MODEL_DIR=os.path.join(OUTPUT_DIR,"models")

os.makedirs(CHART_DIR,exist_ok=True)
os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)

plt.style.use("dark_background")

DARK_BG="#0e1117"
GRID_COLOR="#2a2f3a"
TEXT_COLOR="#e6edf3"
ACCENT="#00d4ff"

files=glob.glob(os.path.join(INPUT_DIR,"*.csv"))

def process_file(file):
    df=pd.read_csv(file)
    df=df.drop_duplicates()
    df.columns=[c.lower().strip().replace(" ","_") for c in df.columns]
    year=int(os.path.basename(file).split(".")[0])
    df["year"]=year
    return df

frames=[]
for file in tqdm(files):
    frames.append(process_file(file))

combined=pd.concat(frames,ignore_index=True)
combined=combined.sort_values("year")
combined.to_excel(os.path.join(DATA_DIR,"master_combined.xlsx"),index=False)

numeric_cols=combined.select_dtypes(include=np.number).columns.tolist()
numeric_cols=[c for c in numeric_cols if c!="year"]

def clip_series(s):
    lower=s.quantile(0.01)
    upper=s.quantile(0.99)
    return s.clip(lower,upper)

def apply_axis_style(ax,xlabel,ylabel,title):
    ax.set_title(title,fontsize=18,color=TEXT_COLOR,pad=15,fontweight="bold")
    ax.set_xlabel(xlabel,fontsize=13,color=TEXT_COLOR,labelpad=10)
    ax.set_ylabel(ylabel,fontsize=13,color=TEXT_COLOR,labelpad=10)
    ax.tick_params(axis='both',colors=TEXT_COLOR,labelsize=10)
    ax.grid(True,color=GRID_COLOR,alpha=0.4)

def save_fig(fig,name):
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR,name),dpi=300,facecolor=fig.get_facecolor())
    plt.close()

corr=combined[numeric_cols].corr()

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(corr,cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Pearson Correlation Matrix")
save_fig(fig,"corr_pearson.png")

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(combined[numeric_cols].corr(method="spearman"),cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Spearman Correlation Matrix")
save_fig(fig,"corr_spearman.png")

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(combined[numeric_cols].corr(method="kendall"),cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Kendall Correlation Matrix")
save_fig(fig,"corr_kendall.png")

cov=combined[numeric_cols].cov()

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(cov,cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Covariance Matrix")
save_fig(fig,"covariance_matrix.png")

year_group=combined.groupby("year")[numeric_cols].mean()

for col in tqdm(numeric_cols[:6]):
    s=clip_series(year_group[col].dropna())
    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(s.index,s.values,color=ACCENT,marker="o")
    apply_axis_style(ax,"Year",col.upper(),f"{col.upper()} TIME SERIES TREND")
    save_fig(fig,f"{col}_timeseries.png")

for col in tqdm(numeric_cols[:5]):
    s=clip_series(combined[col].dropna())

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    ax.hist(s,bins=30,color=ACCENT)
    apply_axis_style(ax,col,"Frequency",f"{col.upper()} HISTOGRAM")
    save_fig(fig,f"{col}_hist.png")

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    sns.boxplot(x=s,ax=ax,color=ACCENT)
    apply_axis_style(ax,col,"Distribution",f"{col.upper()} BOX")
    save_fig(fig,f"{col}_box.png")

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    sns.kdeplot(s,ax=ax,color=ACCENT)
    apply_axis_style(ax,col,"Density",f"{col.upper()} DENSITY")
    save_fig(fig,f"{col}_density.png")

scaled=StandardScaler().fit_transform(combined[numeric_cols].fillna(combined[numeric_cols].mean()))

pca=PCA(n_components=2)
proj=pca.fit_transform(scaled)

fig,ax=plt.subplots(figsize=(14,7))
fig.patch.set_facecolor(DARK_BG)
ax.scatter(proj[:,0],proj[:,1],color=ACCENT)
apply_axis_style(ax,"PC1","PC2","PCA Projection")
save_fig(fig,"pca_projection.png")

iso=IsolationForest(contamination=0.02,random_state=42)
anomaly=iso.fit_predict(scaled)

fig,ax=plt.subplots(figsize=(14,7))
fig.patch.set_facecolor(DARK_BG)
ax.scatter(range(len(anomaly)),combined[numeric_cols[0]].fillna(0),c=anomaly,cmap="coolwarm")
apply_axis_style(ax,"Index",numeric_cols[0],"Anomaly Detection")
save_fig(fig,"anomaly.png")

for target in numeric_cols[:4]:

    df_reg=year_group[[target]].dropna()

    if len(df_reg)<3:
        continue

    X=df_reg.index.values.reshape(-1,1)
    y=df_reg[target].values

    lin=LinearRegression()
    lin.fit(X,y)
    lin_pred=lin.predict(X)

    ridge=Ridge(alpha=1.0)
    ridge.fit(X,y)
    ridge_pred=ridge.predict(X)

    poly_model=Pipeline([
        ("poly",PolynomialFeatures(degree=2)),
        ("reg",LinearRegression())
    ])
    poly_model.fit(X,y)
    poly_pred=poly_model.predict(X)

    rf=RandomForestRegressor(n_estimators=300,random_state=42)
    rf.fit(X,y)
    rf_pred=rf.predict(X)

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(X,y,label="Actual",color=ACCENT,marker="o")
    ax.plot(X,lin_pred,label="Linear",color="orange")
    ax.plot(X,ridge_pred,label="Ridge",color="yellow")
    ax.plot(X,poly_pred,label="Polynomial",color="green")
    ax.plot(X,rf_pred,label="RandomForest",color="red")

    ax.legend()

    apply_axis_style(ax,"Year",target.upper(),f"{target.upper()} Regression Forecast")

    save_fig(fig,f"{target}_forecast.png")

summary={
"Total Records":len(combined),
"Years":combined["year"].nunique(),
"Metrics":len(numeric_cols),
"Charts":len(glob.glob(os.path.join(CHART_DIR,"*.png")))
}

doc=SimpleDocTemplate(os.path.join(OUTPUT_DIR,"Executive_Report.pdf"))
styles=getSampleStyleSheet()

elements=[]
elements.append(Paragraph("<b>World Happiness Index Intelligence Report</b>",styles["Heading1"]))
elements.append(Spacer(1,20))

for k,v in summary.items():
    elements.append(Paragraph(f"{k}: {v}",styles["Normal"]))
    elements.append(Spacer(1,10))

images=glob.glob(os.path.join(CHART_DIR,"*.png"))

for img in images:
    elements.append(Image(img,width=450,height=300))
    elements.append(Spacer(1,20))

doc.build(elements)

print("\nExecution time:",round(time.time()-start_total,2),"seconds")
print("Charts:",len(images))
print("Complete")