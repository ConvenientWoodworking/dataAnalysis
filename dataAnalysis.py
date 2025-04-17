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
    """
    Load a CSV file, rename columns, parse timestamps, and tag with device ID.
    """
    fn = os.path.basename(path)
    match = re.match(r"(AS\d+)_export_.*\.csv", fn)
    device = match.group(1) if match else "Unknown"

    df = pd.read_csv(path)
    df = df.rename(columns={
        df.columns[0]: "Timestamp",
        df.columns[1]: "Temp_F",
        df.columns[2]: "RH"
    })
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Device'] = device
    return df


def find_contiguous_nans(mask):
    """
    Identify start and end indices of contiguous True values in mask.
    """
    gaps = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if not val and start is not None:
            gaps.append((start, i-1))
            start = None
    if start is not None:
        gaps.append((start, len(mask)-1))
    return gaps


def fill_small_gaps(series, max_gap=10, n_neighbors=4):
    """
    Interpolate small gaps (<= max_gap) using surrounding n_neighbors points.
    """
    s = series.copy()
    mask = s.isna().values
    idxs = s.index
    gaps = find_contiguous_nans(mask)
    for start, end in gaps:
        length = end - start + 1
        if length <= max_gap:
            pad_start_idx = max(start - n_neighbors, 0)
            pad_end_idx = min(end + n_neighbors, len(idxs)-1)
            segment = s.iloc[pad_start_idx:pad_end_idx+1]
            s.iloc[pad_start_idx:pad_end_idx+1] = segment.interpolate(method='linear')
    return s


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
    pivot = df.pivot(index='Timestamp', columns='Device', values='Temp_F')
    corr = pivot.corr(method='pearson')
    return corr[ref_dev].drop(ref_dev)


def make_correlation_heatmap(corr_series, title='Pearson Corr vs AS10'):
    fig, ax = plt.subplots()
    data = corr_series.values.reshape(-1, 1)
    cax = ax.matshow(data, aspect='auto')
    fig.colorbar(cax)
    ax.set_yticks(range(len(corr_series)))
    ax.set_yticklabels(corr_series.index)
    ax.set_xticks([])
    ax.set_title(title)
    return fig


def generate_pdf(buffer, logo_path, title, subtitle, summary_df, corr_series):
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = []
    if os.path.exists(logo_path):
        elems.append(Image(logo_path, width=150, height=75))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(f"<strong>{title}</strong>", styles['Title']))
    elems.append(Paragraph(subtitle, styles['Heading2']))
    elems.append(Spacer(1, 12))
    data = [summary_df.columns.tolist()] + summary_df.values.tolist()
    tbl = Table(data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#cccccc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))
    elems.append(tbl)
    elems.append(Spacer(1, 12))
    corr_data = [['Device', 'Corr_vs_AS10']] + [[dev, f"{val:.3f}"] for dev, val in corr_series.items()]
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

folder = st.sidebar.text_input("Data folder path", value="./data")
start_date = st.sidebar.date_input("Start date", value=datetime(2025,1,1))
end_date = st.sidebar.date_input("End date", value=datetime.today())

if st.sidebar.button("Load & Plot"):
    # Ingest all CSVs
    files = glob.glob(os.path.join(folder, "AS*_export_*.csv"))
    device_dfs = {df['Device'].iloc[0]: df for df in [load_and_clean_csv(f) for f in files]}

    # Determine master by row count
    master = max(device_dfs, key=lambda d: len(device_dfs[d]))
    master_index = device_dfs[master].sort_values('Timestamp')['Timestamp']

    # Reindex and fill gaps
    processed = []
    for device, df in device_dfs.items():
        df = df.set_index('Timestamp').reindex(master_index)
        df['Temp_F'] = fill_small_gaps(df['Temp_F'])
        df['RH'] = fill_small_gaps(df['RH'])
        df['Device'] = device
        df = df.reset_index().rename(columns={'index': 'Timestamp'})
        processed.append(df)
    df_all = pd.concat(processed, ignore_index=True)

    # Filter by date range
    mask = (df_all['Timestamp'].dt.date >= start_date) & (df_all['Timestamp'].dt.date <= end_date)
    df_all = df_all.loc[mask]

    # Single time series plot for all devices
    st.header("Temperature (°F) Time Series for All Devices")
    ts = df_all.pivot(index='Timestamp', columns='Device', values='Temp_F')
    st.line_chart(ts)
else:
    st.info("Enter the data folder path and click 'Load & Plot' to display the time series.")
