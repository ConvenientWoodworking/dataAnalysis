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
    return corr[ref_dev].drop(ref_dev)

# --- Streamlit App ---
st.set_page_config(page_title="All Souls Cathedral: Q1 Environmental Data Analysis", layout="wide")
st.sidebar.title("Settings")

# Hardcoded data folder path
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

# Device selector with grouped checkboxes
devices = st.session_state.get('devices', [])
attic = [f"AS{i:02d}" for i in range(15,24)]
main = [f"AS{i:02d}" for i in range(1,15) if i != 10]
crawlspace = [f"AS{i:02d}" for i in range(24,34)]
outdoorref = ["AS10"]

st.sidebar.markdown("### Attic")
if devices:
    if st.sidebar.button("Select All Attic", key="select_all_attic"):
        for dev in attic:
            if dev in devices: st.session_state[f"chk_{dev}"] = True
    if st.sidebar.button("Deselect All Attic", key="deselect_all_attic"):
        for dev in attic:
            if dev in devices: st.session_state[f"chk_{dev}"] = False
for dev in attic:
    if dev in devices:
        key = f"chk_{dev}"; st.session_state.setdefault(key, True)
        st.sidebar.checkbox(dev, key=key)

st.sidebar.markdown("### Main")
if devices:
    if st.sidebar.button("Select All Main", key="select_all_main"):
        for dev in main:
            if dev in devices: st.session_state[f"chk_{dev}"] = True
    if st.sidebar.button("Deselect All Main", key="deselect_all_main"):
        for dev in main:
            if dev in devices: st.session_state[f"chk_{dev}"] = False
for dev in main:
    if dev in devices:
        key = f"chk_{dev}"; st.session_state.setdefault(key, True)
        st.sidebar.checkbox(dev, key=key)

st.sidebar.markdown("### Crawlspace")
if devices:
    if st.sidebar.button("Select All Crawlspace", key="select_all_crawl"):
        for dev in crawlspace:
            if dev in devices: st.session_state[f"chk_{dev}"] = True
    if st.sidebar.button("Deselect All Crawlspace", key="deselect_all_crawl"):
        for dev in crawlspace:
            if dev in devices: st.session_state[f"chk_{dev}"] = False
for dev in crawlspace:
    if dev in devices:
        key = f"chk_{dev}"; st.session_state.setdefault(key, True)
        st.sidebar.checkbox(dev, key=key)

st.sidebar.markdown("### Outdoor Reference")
for dev in outdoorref:
    if dev in devices:
        key = f"chk_{dev}"; st.session_state.setdefault(key, True)
        st.sidebar.checkbox(dev, key=key)

selected = [dev for dev in devices if st.session_state.get(f"chk_{dev}")]

# Analyze & Plot & Stats
if st.sidebar.button("Analyze"):
    if 'df_all' not in st.session_state:
        st.error("Please load data first.")
    else:
        df = st.session_state.df_all
        df = df[df['Device'].isin(selected)]
        mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)
        df = df.loc[mask]

        # Temperature plot
        df_temp = df.melt(id_vars=['Timestamp','Device','Interpolated'], value_vars=['Temp_F'], var_name='Metric')
        line_temp = alt.Chart(df_temp).mark_line().encode(x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')), y='value:Q', color='Device:N')
        points_temp = alt.Chart(df_temp[df_temp['Interpolated']]).mark_circle(size=50).encode(x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')), y='value:Q', color=alt.value('red'))
        st.altair_chart((line_temp+points_temp).properties(title="Temperature Data", width=800, height=400), use_container_width=True)

        # Relative Humidity plot
        df_rh = df.melt(id_vars=['Timestamp','Device','Interpolated'], value_vars=['RH'], var_name='Metric')
        line_rh = alt.Chart(df_rh).mark_line().encode(x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')), y='value:Q', color='Device:N')
        points_rh = alt.Chart(df_rh[df_rh['Interpolated']]).mark_circle(size=50).encode(x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')), y='value:Q', color=alt.value('red'))
        st.altair_chart((line_rh+points_rh).properties(title="Relative Humidity Data", width=800, height=400), use_container_width=True)

        # Normalized Temperature Diff
        df_out = df[df['Device']=='AS10'][['Timestamp','Temp_F','RH']].rename(columns={'Temp_F':'Temp_out','RH':'RH_out'})
        df_norm = pd.merge(df, df_out, on='Timestamp', how='left')
        df_norm['Norm_Temp'] = df_norm['Temp_F'] - df_norm['Temp_out']
        df_norm_temp = df_norm.melt(id_vars=['Timestamp','Device'], value_vars=['Norm_Temp'], var_name='Metric')
        chart_norm_t = alt.Chart(df_norm_temp).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y='value:Q',
            color='Device:N'
        ).properties(title="Normalized Temperature Difference", width=800, height=400)
        st.altair_chart(chart_norm_t, use_container_width=True)

        # Normalized RH Diff
        df_norm['Norm_RH'] = df_norm['RH'] - df_norm['RH_out']
        df_norm_rh = df_norm.melt(id_vars=['Timestamp','Device'], value_vars=['Norm_RH'], var_name='Metric')
        chart_norm_r = alt.Chart(df_norm_rh).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y='value:Q',
            color='Device:N'
        ).properties(title="Normalized Relative Humidity Difference", width=800, height=400)
        st.altair_chart(chart_norm_r, use_container_width=True)

        # Statistical Analysis
        st.header("Summary Statistics (Temperature)")
        summary_temp = compute_summary_stats(df, field='Temp_F')
        st.dataframe(summary_temp)

        st.header("Summary Statistics (Relative Humidity)")
        summary_rh = compute_summary_stats(df, field='RH')
        st.dataframe(summary_rh)

        st.header("Pearson Correlation vs Outdoor Reference (Temperature)")
        corr_temp = compute_correlations(df, field='Temp_F', ref_dev='AS10')
        st.table(corr_temp.reset_index().rename(columns={"index":"Device"}))

        st.header("Pearson Correlation vs Outdoor Reference (RH)")
        corr_rh = compute_correlations(df, field='RH', ref_dev='AS10')
        st.table(corr_rh.reset_index().rename(columns={"index":"Device"}))
else:
    st.info("Use 'Load Data' then select devices and 'Analyze' to view time series with interpolated points flagged.")
