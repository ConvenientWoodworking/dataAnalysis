import os
import re
import glob
import io
from datetime import datetime

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# --- Helper functions ---
def load_and_clean_csv(path):
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
    gaps, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            gaps.append((start, i-1))
            start = None
    if start is not None:
        gaps.append((start, len(mask)-1))
    return gaps


def fill_and_flag(series, max_gap=10, n_neighbors=4):
    orig = series.copy()
    s = series.copy()
    mask = s.isna().values
    idxs = s.index
    gaps = find_contiguous_nans(mask)
    for start, end in gaps:
        length = end - start + 1
        if length <= max_gap:
            pad_start = max(start - n_neighbors, 0)
            pad_end = min(end + n_neighbors, len(idxs)-1)
            segment = s.iloc[pad_start:pad_end+1]
            s.iloc[pad_start:pad_end+1] = segment.interpolate()
    interpolated = orig.isna() & s.notna()
    return s, interpolated


def compute_summary_stats(df, field='Temp_F'):
    stats = df.groupby('Device').agg(
        count=(field, 'count'),
        avg=(field, 'mean'),
        std=(field, 'std'),
        min=(field, 'min'),
        max=(field, 'max'),
        median=(field, lambda x: x.quantile(0.5)),
        missing=(field, lambda x: x.isna().sum())
    ).reset_index()
    return stats


def compute_correlations(df, field='Temp_F', ref_dev='AS10'):
    pivot = df.pivot(index='Timestamp', columns='Device', values=field)
    corr = pivot.corr(method='pearson')
    return corr

# --- Streamlit App ---
st.set_page_config(page_title="All Souls Cathedral: Q1 Environmental Data Analysis", layout="wide")
st.sidebar.title("Settings")

folder = "./data"
start_date = st.sidebar.date_input("Start date", value=datetime(2025,1,1))
end_date = st.sidebar.date_input("End date", value=datetime.today())

# Load Data
if st.sidebar.button("Load Data"):
    files = glob.glob(os.path.join(folder, "AS*_export_*.csv"))
    device_dfs = {}
    for f in files:
        df = load_and_clean_csv(f)
        device_dfs[df['Device'].iloc[0]] = df

    master = max(device_dfs, key=lambda d: len(device_dfs[d]))
    master_idx = device_dfs[master].sort_values('Timestamp')['Timestamp']

    processed = []
    for dev, df in device_dfs.items():
        tmp = df.set_index('Timestamp').reindex(
            master_idx,
            method='nearest',
            tolerance=pd.Timedelta(minutes=30)
        )
        filled_temp, flag_temp = fill_and_flag(tmp['Temp_F'])
        filled_rh, flag_rh = fill_and_flag(tmp['RH'])
        tmp['Temp_F'] = filled_temp
        tmp['RH'] = filled_rh
        tmp['Interpolated'] = flag_temp
        tmp['Device'] = dev
        processed.append(tmp.reset_index().rename(columns={'index':'Timestamp'}))

    st.session_state.df_all = pd.concat(processed, ignore_index=True)
    st.session_state.devices = sorted(st.session_state.df_all['Device'].unique())

# Device selector
devices = st.session_state.get('devices', [])
attic = [f"AS{i:02d}" for i in range(15,24)]
main = [f"AS{i:02d}" for i in range(1,15) if i != 10]
crawlspace = [f"AS{i:02d}" for i in range(24,34)]
outdoorref = ["AS10"]

# Group checkboxes omitted for brevity
# ...
selected = [dev for dev in devices if st.session_state.get(f"chk_{dev}")]

if st.sidebar.button("Analyze"):
    if 'df_all' not in st.session_state:
        st.error("Please load data first.")
    else:
        df = st.session_state.df_all
        df = df[df['Device'].isin(selected)]
        df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]

        # Standard plots... (Temperature, RH, Normalized)
        # ...

        # Correlation matrix for Temperature
        st.header("Correlation Matrix (Temperature)")
        corr_temp = compute_correlations(df, field='Temp_F')
        corr_temp_df = corr_temp.reset_index().melt('Device', var_name='Device2', value_name='Corr')
        heatmap_temp = alt.Chart(corr_temp_df).mark_rect().encode(
            x='Device2:O', y='Device:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heatmap_temp, use_container_width=False)

        # Correlation matrix for RH
        st.header("Correlation Matrix (Relative Humidity)")
        corr_rh = compute_correlations(df, field='RH')
        corr_rh_df = corr_rh.reset_index().melt('Device', var_name='Device2', value_name='Corr')
        heatmap_rh = alt.Chart(corr_rh_df).mark_rect().encode(
            x='Device2:O', y='Device:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heatmap_rh, use_container_width=False)

        # Summary Statistics
        st.header("Summary Statistics (Temperature)")
        summary_temp = compute_summary_stats(df, field='Temp_F')
        st.dataframe(summary_temp)

        st.header("Summary Statistics (Relative Humidity)")
        summary_rh = compute_summary_stats(df, field='RH')
        st.dataframe(summary_rh)

        # Correlations vs AS10
        st.header("Pearson Correlation vs Outdoor Reference (Temperature)")
        corr_vs_ref_temp = compute_correlations(df, field='Temp_F')['AS10']
        st.table(corr_vs_ref_temp.reset_index().rename(columns={"index":"Device", "AS10":"Corr"}))

        st.header("Pearson Correlation vs Outdoor Reference (RH)")
        corr_vs_ref_rh = compute_correlations(df, field='RH')['AS10']
        st.table(corr_vs_ref_rh.reset_index().rename(columns={"index":"Device", "AS10":"Corr"}))
else:
    st.info("Use 'Load Data' then select devices and 'Analyze' to view time series with interpolated points flagged.")
