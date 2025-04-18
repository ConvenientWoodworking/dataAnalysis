import os
import re
import glob
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
            gaps.append((start, i-1)); start = None
    if start is not None:
        gaps.append((start, len(mask)-1))
    return gaps


def fill_and_flag(series, max_gap=10, n_neighbors=4):
    orig = series.copy()
    s = series.copy()
    mask = s.isna().values
    idxs = s.index
    for start, end in find_contiguous_nans(mask):
        if (end - start + 1) <= max_gap:
            left = max(start - n_neighbors, 0)
            right = min(end + n_neighbors, len(idxs)-1)
            segment = s.iloc[left:right+1]
            s.iloc[left:right+1] = segment.interpolate()
    interpolated = orig.isna() & s.notna()
    return s, interpolated


def compute_summary_stats(df, field='Temp_F'):
    return df.groupby('Device').agg(
        count=(field, 'count'),
        avg=(field, 'mean'),
        std=(field, 'std'),
        min=(field, 'min'),
        max=(field, 'max'),
        median=(field, lambda x: x.quantile(0.5)),
        missing=(field, lambda x: x.isna().sum())
    ).reset_index()


def compute_correlations(df, field='Temp_F'):
    pivot = df.pivot(index='Timestamp', columns='Device', values=field)
    return pivot.corr(method='pearson')

# --- Streamlit App ---
st.set_page_config(page_title="All Souls Cathedral: Q1 Environmental Data Analysis", layout="wide")
st.sidebar.title("Settings")

# Constants
FOLDER = "./data"

# Date selectors
date_cols = st.sidebar.columns(2)
date_cols[0].write("Start date")
start_date = date_cols[0].date_input("", value=datetime(2025,1,1))
date_cols[1].write("End date")
end_date = date_cols[1].date_input("", value=datetime.today())

# Load Data button
if st.sidebar.button("Load Data"):
    files = glob.glob(os.path.join(FOLDER, "AS*_export_*.csv"))
    device_dfs = {}
    for f in files:
        df = load_and_clean_csv(f)
        device_dfs[df['Device'].iloc[0]] = df

    master = max(device_dfs, key=lambda d: len(device_dfs[d]))
    master_times = device_dfs[master].sort_values('Timestamp')['Timestamp']

    records = []
    for dev, df in device_dfs.items():
        tmp = df.set_index('Timestamp').reindex(
            master_times, method='nearest', tolerance=pd.Timedelta(minutes=30)
        )
        filled_t, flag_t = fill_and_flag(tmp['Temp_F'])
        filled_r, flag_r = fill_and_flag(tmp['RH'])
        tmp['Temp_F']      = filled_t
        tmp['RH']          = filled_r
        tmp['Interpolated']= flag_t
        tmp['Device']      = dev
        records.append(tmp.reset_index().rename(columns={'index':'Timestamp'}))

    st.session_state.df_all  = pd.concat(records, ignore_index=True)
    st.session_state.devices = sorted(st.session_state.df_all['Device'].unique())

# Device groupings
devices    = st.session_state.get('devices', [])
attic      = [f"AS{i:02d}" for i in range(15,24)]
main       = [f"AS{i:02d}" for i in range(1,15) if i != 10]
crawlspace = [f"AS{i:02d}" for i in range(24,34)]
outdoor    = ["AS10"]

# Grouped checkboxes with dual-column buttons
def group_ui(group, label):
    st.sidebar.markdown(f"**{label}**")
    c1, c2 = st.sidebar.columns([1,1])
    if c1.button(f"Select All {label}"):
        for d in group:
            if d in devices: st.session_state[f"chk_{d}"] = True
    if c2.button(f"Deselect All {label}"):
        for d in group:
            if d in devices: st.session_state[f"chk_{d}"] = False
    for d in group:
        if d in devices:
            key = f"chk_{d}"
            st.session_state.setdefault(key, True)
            st.sidebar.checkbox(d, key=key)

group_ui(attic,      "Attic")
group_ui(main,       "Main")
group_ui(crawlspace, "Crawlspace")
for d in outdoor:
    if d in devices:
        key = f"chk_{d}"
        st.session_state.setdefault(key, True)
        st.sidebar.checkbox(d, key=key)

selected = [d for d in devices if st.session_state.get(f"chk_{d}")]

# Analyze & Display
if st.sidebar.button("Analyze"):
    if 'df_all' not in st.session_state:
        st.error("Please load data first.")
    else:
        df = st.session_state.df_all
        df = df[df['Device'].isin(selected)]
        df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]

                # Temperature plot
        df_t = df.melt(
            id_vars=['Timestamp','Device','Interpolated'],
            value_vars=['Temp_F'],
            var_name='Metric'
        )
        line_temp = alt.Chart(df_t).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q', title='Temperature (°F)'),
            color='Device:N'
        ).properties(title='Temperature Data')
        points_temp = alt.Chart(df_t[df_t['Interpolated']]).mark_circle(size=50, color='red').encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q')
        )
        st.altair_chart(line_temp + points_temp, use_container_width=True)

        # Relative Humidity plot
        df_r = df.melt(
            id_vars=['Timestamp','Device','Interpolated'],
            value_vars=['RH'],
            var_name='Metric'
        )
        line_rh = alt.Chart(df_r).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q', title='Relative Humidity (%)'),
            color='Device:N'
        ).properties(title='Relative Humidity Data')
        points_rh = alt.Chart(df_r[df_r['Interpolated']]).mark_circle(size=50, color='red').encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q')
        )
        st.altair_chart(line_rh + points_rh, use_container_width=True)

        # Normalized diffs
        df_out = df[df['Device']=='AS10'][['Timestamp','Temp_F','RH']].rename(columns={'Temp_F':'T_out','RH':'RH_out'})
        df_norm = df.merge(df_out, on='Timestamp')
        df_norm['Norm_T']  = df_norm['Temp_F'] - df_norm['T_out']
        df_norm['Norm_RH'] = df_norm['RH']     - df_norm['RH_out']
        st.altair_chart(
            alt.Chart(df_norm).mark_line().encode(
                x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
                y='Norm_T:Q', color='Device:N'
            )
        , use_container_width=True)
        st.altair_chart(
            alt.Chart(df_norm).mark_line().encode(
                x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
                y='Norm_RH:Q', color='Device:N'
            )
        , use_container_width=True)

        # Correlation matrices
        st.header("Correlation Matrix (Temperature)")
        corr_t = compute_correlations(df, field='Temp_F')
        df_ct = (
            corr_t.reset_index().rename(columns={'index':'Device'})
            .melt(id_vars='Device', var_name='Device2', value_name='Corr')
        )
        heat_t = alt.Chart(df_ct).mark_rect().encode(
            x='Device2:O', y='Device:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heat_t, use_container_width=False)

        st.header("Correlation Matrix (Relative Humidity)")
        corr_h = compute_correlations(df, field='RH')
        df_ch = (
            corr_h.reset_index().rename(columns={'index':'Device'})
            .melt(id_vars='Device', var_name='Device2', value_name='Corr')
        )
        heat_h = alt.Chart(df_ch).mark_rect().encode(
            x='Device2:O', y='Device:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heat_h, use_container_width=False)

        # Corr vs AS10 tables
        st.header("Pearson Corr vs AS10 (Temp)")
        cvt = compute_correlations(df, field='Temp_F')['AS10']
        st.table(cvt.reset_index().rename(columns={"index":"Device","AS10":"Corr"}))
        st.header("Pearson Corr vs AS10 (RH)")
        cvr = compute_correlations(df, field='RH')['AS10']
        st.table(cvr.reset_index().rename(columns={"index":"Device","AS10":"Corr"}))

        # Summary Stats
        st.header("Summary Statistics (Temperature)")
        st.dataframe(compute_summary_stats(df, field='Temp_F'))
        st.header("Summary Statistics (Relative Humidity)")
        st.dataframe(compute_summary_stats(df, field='RH'))
else:
    st.info("Use 'Load Data' then 'Analyze' to run the full analysis.")
