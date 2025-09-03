# app.py
# ================================================================
# "Calouro - Pagante (DELTAS) | 5 Dispersões + 2 Heatmaps + Barras (IC95%)"
# App Streamlit — pronto p/ publicar no Streamlit Cloud
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import atanh, tanh, sqrt
from scipy.stats import pearsonr, spearmanr, norm

st.set_page_config(page_title="Calouro - Pagante (DELTAS)", layout="wide")

ROXO = "#9900FF"
NUM_COLS = ["D_CONV","D_REND100","D_MENOR40","D_REPROV","D_NAO_AI","D_MIX_INAD","AUM","D_DEV_BOLSA"]
ALL_COLS = ["MARCA"] + NUM_COLS

st.title("Calouro - Pagante (DELTAS) • Dispersões + Heatmaps + Barras (IC95%)")
st.caption("Interativo • upload de CSV • barras em roxo • escala Viridis_r nas dispersões")

# ---------------------------
# Helpers
# ---------------------------
def to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("%","").replace("pp","").replace(",",".").strip()
    try:
        return float(s)
    except:
        return np.nan

def fisher_ci(r, n, alpha=0.05):
    """IC 95% para r de Pearson (Fisher)."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = atanh(r); se = 1.0 / sqrt(n - 3); zcrit = norm.ppf(1 - alpha/2.0)
    return tanh(z - zcrit*se), tanh(z + zcrit*se)

def add_quadrant_lines(fig, x0, x1, y0, y1):
    fig.add_hline(y=0, line_dash="dash", opacity=0.6)
    fig.add_vline(x=0, line_dash="dash", opacity=0.6)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[y0, y1])

def scatter_corr(df, y_col, y_label, title):
    x = df["D_CONV"]; y = df[y_col]
    dx = (x.max() - x.min())*0.1 if np.isfinite(x.max()-x.min()) else 1
    dy = (y.max() - y.min())*0.1 if np.isfinite(y.max()-y.min()) else 1
    x0, x1 = x.min()-dx, x.max()+dx
    y0, y1 = y.min()-dy, y.max()+dy

    fig = px.scatter(
        df, x="D_CONV", y=y_col, text="MARCA",
        color="AUM", color_continuous_scale="Viridis_r",
        labels={"D_CONV":"Δ % Conversão vs AA (pp)", y_col:y_label, "AUM":"Δ % AUM Percebido"},
        title=title
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=0.6, color="black")),
                      textposition="top center")
    add_quadrant_lines(fig, x0, x1, y0, y1)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    return fig

def heatmap_corr(df, method="pearson"):
    corr = df[NUM_COLS].corr(method=method)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        colorbar=dict(title="Correlação")
    ))
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            fig.add_annotation(
                x=col, y=row, text=f"{corr.iloc[i,j]:.2f}",
                showarrow=False, font=dict(size=12, color="black"),
                xanchor="center", yanchor="middle",
                bgcolor="rgba(255,255,255,0.7)"
            )
    fig.update_layout(
        title=f"Heatmap de Correlação — {method.capitalize()} (Deltas)",
        xaxis=dict(tickangle=45),
        margin=dict(l=80,r=20,t=60,b=120)
    )
    return fig

def barras_correlacoes(df):
    var_map = {
        "D_REND100": "REND",
        "D_MENOR40": "Média <40",
        "D_REPROV" : "Reprovação",
        "D_NAO_AI" : "NAO_AI",
        "D_MIX_INAD":"Inadimplência",
        "AUM"      : "AUM"
    }
    x_var = "D_CONV"
    labels, rvals, lo_list, hi_list = [], [], [], []
    for col, lbl in var_map.items():
        sub = df[[x_var, col]].dropna(); n = len(sub)
        if n < 3:
            r, lo, hi = np.nan, np.nan, np.nan
        else:
            r, _ = pearsonr(sub[x_var], sub[col])
            lo, hi = fisher_ci(r, n)
        labels.append(lbl); rvals.append(r); lo_list.append(lo); hi_list.append(hi)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rvals,
        marker_color=ROXO, marker_line=dict(color="black", width=0.8),
        name="Correlação (Pearson)"
    ))
    y_err_plus  = [ (hi - r) if np.isfinite(hi) and np.isfinite(r) else 0 for r, hi in zip(rvals, hi_list) ]
    y_err_minus = [ (r - lo) if np.isfinite(lo) and np.isfinite(r) else 0 for r, lo in zip(rvals, lo_list) ]
    fig.update_traces(error_y=dict(
        type='data', symmetric=False, array=y_err_plus, arrayminus=y_err_minus,
        thickness=1.2, width=5, color="black"
    ))
    fig.add_hline(y=0, line_dash="dash", opacity=0.7)
    fig.update_layout(
        title="Correlação com Δ % Conversão (IC 95%)",
        yaxis_title="Coeficiente de Correlação (Pearson)",
        margin=dict(l=40,r=20,t=60,b=40)
    )
    return fig

# ---------------------------
# Dados padrão (se não subir CSV)
# ---------------------------
raw_default = [
 ["ÂNIMA BR","-1,2 pp","-3,9 pp","1,7 pp","3,3 pp","-24,8 pp","-3,1 pp","36,1%","-37,1 pp"],
 ["AGES","-2,0 pp","-5,1 pp","1,8 pp","5,8 pp","-36,1 pp","-2,5 pp","102,9%","-45,7 pp"],
 ["UNIFG - BA","1,5 pp","-3,7 pp","1,1 pp","3,4 pp","-17,8 pp","-3,1 pp","126,9%","-49,8 pp"],
 ["UNF","-5,5 pp","-3,2 pp","1,7 pp","2,8 pp","-24,6 pp","-3,6 pp","88,4%","-28,1 pp"],
 ["UNP","-2,1 pp","-4,5 pp","2,7 pp","4,2 pp","-26,9 pp","-3,8 pp","88,3%","-33,6 pp"],
 ["FPB","-8,4 pp","1,7 pp","0,0 pp","-2,6 pp","-0,2 pp","-6,7 pp","95,6%","-28,1 pp"],
 ["UNIFG - PE","-4,5 pp","-0,4 pp","-1,7 pp","-0,9 pp","-16,1 pp","-6,4 pp","100,0%","-30,3 pp"],
 ["UAM","1,0 pp","-5,3 pp","2,3 pp","4,4 pp","-26,6 pp","-2,3 pp","12,1%","0,0 pp"],
 ["USJT","1,2 pp","-4,7 pp","1,8 pp","3,8 pp","-25,1 pp","-3,3 pp","11,6%","0,0 pp"],
 ["UNA","0,5 pp","-3,3 pp","1,9 pp","2,5 pp","-19,7 pp","-2,8 pp","11,7%","0,0 pp"],
 ["UNIBH","0,1 pp","-1,8 pp","-0,5 pp","1,4 pp","-33,4 pp","-4,2 pp","20,7%","0,0 pp"],
 ["IBMR","-6,6 pp","-2,0 pp","-0,7 pp","0,8 pp","-26,2 pp","-4,9 pp","109,0%","-49,5 pp"],
 ["FASEH","-1,2 pp","-9,9 pp","7,9 pp","8,8 pp","-11,6 pp","0,7 pp","7,3%","0,0 pp"],
 ["MIL. CAMPOS","7,1 pp","7,8 pp","-2,4 pp","-7,8 pp","0,0 pp","-7,5 pp","17,7%","0,0 pp"],
 ["UNISUL","-8,8 pp","-10,5 pp","6,9 pp","10,2 pp","-32,5 pp","5,7 pp","-4,2%","0,0 pp"],
 ["UNICURITIBA","3,7 pp","-2,5 pp","-0,4 pp","2,0 pp","-16,3 pp","-7,1 pp","13,5%","0,0 pp"],
 ["UNISOCIESC","-5,1 pp","-9,8 pp","6,8 pp","9,1 pp","-25,4 pp","9,3 pp","11,7%","0,0 pp"],
 ["UNR","0,0 pp","-1,5 pp","-1,9 pp","1,0 pp","-27,3 pp","-11,7 pp","88,4%","-47,5 pp"],
 ["FAD","2,7 pp","5,1 pp","-5,0 pp","-5,3 pp","-23,2 pp","-13,8 pp","80,9%","-26,5 pp"]
]
df_default = pd.DataFrame(raw_default, columns=ALL_COLS)
for c in NUM_COLS:
    df_default[c] = df_default[c].apply(to_float)

# ---------------------------
# Sidebar: Upload CSV
# ---------------------------
st.sidebar.header("Dados")
st.sidebar.write("Envie um CSV com as colunas:")
st.sidebar.code(", ".join(ALL_COLS), language="text")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded, dtype=str)
        missing = [c for c in ALL_COLS if c not in df.columns]
        if missing:
            st.sidebar.error(f"Faltam colunas no CSV: {missing}. Usando base de exemplo.")
            df = df_default.copy()
        else:
            for c in NUM_COLS:
                df[c] = df[c].apply(to_float)
    except Exception as e:
        st.sidebar.error(f"Erro ao ler CSV: {e}. Usando base de exemplo.")
        df = df_default.copy()
else:
    df = df_default.copy()

# ---------------------------
# Abas e gráficos
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Dispersões", "Heatmaps", "Barras (IC95%)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            scatter_corr(df, "D_MENOR40", "Δ % Média < 40 (pp)", "Δ Conversão (X) vs Δ % Média < 40 (Y)"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            scatter_corr(df, "D_REPROV", "Δ % Reprovado (pp)", "Δ Conversão (X) vs Δ % Reprovado (Y)"),
            use_container_width=True
        )
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            scatter_corr(df, "D_REND100", "Δ % Rendimento 100% (pp)", "Δ Conversão (X) vs Δ % Rendimento 100% (Y)"),
            use_container_width=True
        )
    with c4:
        st.plotly_chart(
            scatter_corr(df, "D_MIX_INAD", "Δ % Mix Inadimplência (pp)", "Δ Conversão (X) vs Δ % Mix Inadimplência (Y)"),
            use_container_width=True
        )
    st.plotly_chart(
        scatter_corr(df, "D_NAO_AI", "Δ % Não Realizou AI (pp)", "Δ Conversão (X) vs Δ % Não Realizou AI (Y)"),
        use_container_width=True
    )

with tab2:
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(heatmap_corr(df, method="pearson"), use_container_width=True)
    with colB:
        st.plotly_chart(heatmap_corr(df, method="spearman"), use_container_width=True)

with tab3:
    st.plotly_chart(barras_correlacoes(df), use_container_width=True)

st.markdown("---")
st.caption(f"Tema: barras em roxo {ROXO}; dispersões com escala Viridis invertida (Δ AUM).")
