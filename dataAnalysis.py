import os
import re
import glob
import io
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import streamlit as st

# --- Helper functions ---
def load_and_clean_csv(path):
    # Extract device ID from filename
    fn = os.path.basename(path)
    match = re.match(r"(AS\d+)_export_.*\.csv", fn)
    device = match.group(1) if match else "Unknown"

    # Load
    df = pd.read_csv(path)
    # Rename columns
    df = df.rename(columns={
        df.columns[0]: "Timestamp",
        df.columns[1]: "Temp_F",
        df.columns[2]: "RH"
    })
    # Parse timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Device'] = device
    return df


def interpolate_gaps(df, col, max_gap=5):
    # Linear interpolate up to max_gap consecutive NaNs
    return df[col].interpolate(method='linear', limit=max_gap, limit_direction='both')


def compute_summary_stats(df, group_col='Device'):
    stats = df.groupby(group_col).agg(
        count=('Temp_F', 'count'),
        mean=('Temp_F', 'mean'),
        std=('Temp_F', 'std'),
        min=('Temp_F', 'min'),
        max=('Temp_F', 'max'),
        median=('Temp_F', lambda x: x.quantile(0.5)),
        p90=('Temp_F', lambda x: x.quantile(0.9)),
        missing=('Temp_F', lambda x: x.isna().sum()),
    ).reset_index()
    return stats


def compute_correlations(df, ref_dev='AS10'):
    # Pivot Temp_F by device
    pivot = df.pivot(index='Timestamp', columns='Device', values='Temp_F')
    corr = pivot.corr(method='pearson')
    # Extract correlations vs AS10
    corr_vs_ref = corr[ref_dev].drop(ref_dev)
    return corr_vs_ref


def make_correlation_heatmap(corr_df):
    fig, ax = plt.subplots()
    cax = ax.matshow(corr_df.values.reshape(-1,1), aspect='auto')
    fig.colorbar(cax)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_xticks([])
    ax.set_title('Pearson Corr vs AS10 (Temp_F)')
    return fig


def generate_pdf(buffer, logo_path, title, subtitle, summary_df, corr_series):
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = []
    # Logo
    if os.path.exists(logo_path):
        elems.append(Image(logo_path, width=150, height=75))
    elems.append(Spacer(1, 12))
    # Title/Sub
    elems.append(Paragraph(f"<strong>{title}</strong>", styles['Title']))
    elems.append(Paragraph(subtitle, styles['Heading2']))
    elems.append(Spacer(1, 12))
    # Summary stats table
    data = [summary_df.columns.tolist()] + summary_df.values.tolist()
    tbl = Table(data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#cccccc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))
    # Correlation table
    corr_data = [['Device', 'Corr_vs_AS10']] + [[dev, f"{val:.3f}"] for dev,val in corr_series.items()]
    corr_tbl = Table(corr_data, hAlign='LEFT')
    corr_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#cccccc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))
    elems.append(corr_tbl)
    doc.build(elems)
    buffer.seek(0)
    return buffer

# --- Streamlit App ---
st.set_page_config(page_title="All Souls Cathedral: Q1 Environmental Data Analysis", layout="wide")
st.sidebar.title("Settings")

# 1. Folder input
folder = st.sidebar.text_input("Data folder path", value="./data")
# 2. Date range
start_date = st.sidebar.date_input("Start date", value=datetime(2025,1,1))
end_date = st.sidebar.date_input("End date", value=datetime.today())
# 3. Load data
if st.sidebar.button("Load & Analyze"):
    # Ingest
    files = glob.glob(os.path.join(folder, "AS*_export_*.csv"))
    dfs = [load_and_clean_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values('Timestamp')

    # Apply date filter
    mask = (df_all['Timestamp'].dt.date >= start_date) & (df_all['Timestamp'].dt.date <= end_date)
    df_all = df_all.loc[mask]

    # Gap fill per device
    df_all = df_all.set_index('Timestamp')
    for dev in df_all['Device'].unique():
        idx = df_all['Device']==dev
        df_all.loc[idx, 'Temp_F'] = interpolate_gaps(df_all[idx], 'Temp_F')
        df_all.loc[idx, 'RH'] = interpolate_gaps(df_all[idx], 'RH')
    df_all = df_all.reset_index()

    # Compute summary and correlations
    summary = compute_summary_stats(df_all)
    corr_vs = compute_correlations(df_all)

    # Sidebar device selector after load
    devices = df_all['Device'].unique().tolist()
    selected = st.sidebar.multiselect("Select devices", devices, default=devices)
    df_sel = df_all[df_all['Device'].isin(selected)]

    # Charts
    st.header("Time Series Plots")
    st.line_chart(data=df_sel.pivot(index='Timestamp', columns='Device', values='Temp_F'), height=300)
    st.line_chart(data=df_sel.pivot(index='Timestamp', columns='Device', values='RH'), height=300)

    # Normalized difference plot
    # Merge with outdoor AS10
    df_out = df_sel[df_sel['Device']=='AS10'][['Timestamp','Temp_F','RH']].rename(columns={'Temp_F':'Temp_out','RH':'RH_out'})
    df_merged = pd.merge(df_sel, df_out, on='Timestamp', how='left')
    df_merged['Norm_Temp'] = df_merged['Temp_F'] - df_merged['Temp_out']
    df_merged['Norm_RH'] = df_merged['RH'] - df_merged['RH_out']
    st.header("Normalized Differences vs Outdoor (AS10)")
    st.line_chart(data=df_merged.pivot(index='Timestamp', columns='Device', values='Norm_Temp'), height=300)
    st.line_chart(data=df_merged.pivot(index='Timestamp', columns='Device', values='Norm_RH'), height=300)

    # Summary and correlation tables
    st.header("Summary Statistics (Temp_F)")
    st.dataframe(summary)
    st.header("Pearson Correlation vs AS10 (Temp_F)")
    fig = make_correlation_heatmap(corr_vs)
    st.pyplot(fig)

    # PDF Export
    buffer = io.BytesIO()
    pdf_buffer = generate_pdf(buffer,
                              logo_path="BSD_Logo_FINAL_MAIN.png",
                              title="All Souls Cathedral",
                              subtitle="Q1 Environmental Data Analysis",
                              summary_df=summary,
                              corr_series=corr_vs)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="All_Souls_Q1_Analysis.pdf",
        mime="application/pdf"
    )

else:
    st.info("Enter the data folder path and click 'Load & Analyze' to begin.")
