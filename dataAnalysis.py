import os
import re
import glob
from datetime import datetime

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# --- Device name mapping ---
# Mapping from AS codes to friendly names
DEVICE_LABELS = {
    'AS01': "Zabriskie-Main",
    'AS02': "Kitchen-Main",
    'AS03': "Women's Restroom-Main",
    'AS04': "Music Room-Main",
    'AS05': "Children's Wing - Room 2-Main",
    'AS06': "Children's Wing - Room 5-Main",
    'AS07': "CE Room-Main",
    'AS08': "Owen Library-Main",
    'AS09': "Reception Room-Main",
    'AS10': "Outdoor Reference",
    'AS11': "Tower-Main",
    'AS12': "Robing Access-Main",
    'AS13': "Men's Robing-Main",
    'AS14': "Women's Robing-Main",
    'AS15': "Robing-Attic",
    'AS16': "East Organ-Attic",
    'AS17': "Front Porch-Attic",
    'AS18': "Owen Library-Attic",
    'AS19': "CE Room-Attic",
    'AS20': "Kitchen-Attic",
    'AS21': "Music Room-Attic",
    'AS22': "Children's Wing - Room 5-Attic",
    'AS23': "Zabriskie-Attic",
    'AS24': "Zabriskie - North-Crawlspace",
    'AS25': "Kitchen-Crawlspace",
    'AS26': "Zabriskie - Central-Crawlspace",
    'AS27': "Apse-Crawlspace",
    'AS28': "East Transept-Crawlspace",
    'AS29': "West Organ-Crawlspace",
    'AS30': "Baptistery-Crawlspace",
    'AS31': "East Nave-Crawlspace",
    'AS32': "Women's Robing-Crawlspace",
    'AS33': "Men's Robing-Crawlspace"
}

# --- Helper functions ---
def load_and_clean_csv(path):
    fn = os.path.basename(path)
    match = re.match(r"(AS\d+)_export_.*\.csv", fn)
    device = match.group(1) if match else "Unknown"

    df = pd.read_csv(path)
    df = df.rename(columns={
        df.columns[0]: 'Timestamp',
        df.columns[1]: 'Temp_F',
        df.columns[2]: 'RH'
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
    return df.groupby('DeviceName').agg(
        count=(field, 'count'),
        avg=(field, 'mean'),
        std=(field, 'std'),
        min=(field, 'min'),
        max=(field, 'max'),
        median=(field, lambda x: x.quantile(0.5)),
        missing=(field, lambda x: x.isna().sum())
    ).reset_index()


def compute_correlations(df, field='Temp_F'):
    pivot = df.pivot(index='Timestamp', columns='DeviceName', values=field)
    return pivot.corr(method='pearson')

# --- Streamlit App ---
st.set_page_config(page_title='All Souls Cathedral: Q1 Environmental Data Analysis', layout='wide')
st.sidebar.title('Settings')

# Constants
FOLDER = './data'

# Date selectors
date_cols = st.sidebar.columns(2)
date_cols[0].write('Start date')
start_date = date_cols[0].date_input('', value=datetime(2025,1,1))
date_cols[1].write('End date')
end_date   = date_cols[1].date_input('', value=datetime.today())

# Load Data button
if st.sidebar.button('Load Data'):
    files = glob.glob(os.path.join(FOLDER, 'AS*_export_*.csv'))
    device_dfs = {load_and_clean_csv(f)['Device'].iloc[0]: load_and_clean_csv(f) for f in files}

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
        # Map to friendly name
        tmp['DeviceName']  = DEVICE_LABELS.get(dev, dev)
        records.append(tmp.reset_index().rename(columns={'index':'Timestamp'}))

    df_all = pd.concat(records, ignore_index=True)
    st.session_state.df_all    = df_all
    st.session_state.devices   = sorted(df_all['Device'])

# Device groupings
devices    = st.session_state.get('devices', [])
attic      = [f'AS{i:02d}' for i in range(15,24)]
main       = [f'AS{i:02d}' for i in range(1,15) if i != 10]
crawlspace = [f'AS{i:02d}' for i in range(24,34)]
outdoor    = ['AS10']

# Grouped checkboxes
def group_ui(group, label):
    st.sidebar.markdown(f'**{label}**')
    col1, col2 = st.sidebar.columns(2)
    if col1.button(f'Select All {label}'):
        for d in group:
            if d in devices: st.session_state[f'chk_{d}'] = True
    if col2.button(f'Deselect All {label}'):
        for d in group:
            if d in devices: st.session_state[f'chk_{d}'] = False
    for d in group:
        if d in devices:
            key = f'chk_{d}'
            st.session_state.setdefault(key, True)
            label = DEVICE_LABELS.get(d, d)
            st.sidebar.checkbox(label, key=key)

group_ui(attic,      'Attic')
group_ui(main,       'Main')
group_ui(crawlspace, 'Crawlspace')
# Outdoor Reference (no select/deselect buttons)
st.sidebar.markdown("**Outdoor Reference**")
for d in outdoor:
    if d in devices:
        key = f"chk_{d}"
        st.session_state.setdefault(key, True)
        st.sidebar.checkbox(DEVICE_LABELS.get(d, d), key=key)

selected = [d for d in devices if st.session_state.get(f'chk_{d}')]

# Analyze & Display
if st.sidebar.button('Analyze'):
    if 'df_all' not in st.session_state:
        st.error('Please load data first.')
    else:
        df = st.session_state.df_all
        df = df[df['Device'].isin(selected)]
        df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]

        # Temperature plot
        df['DeviceName'] = df['Device'].map(DEVICE_LABELS).fillna(df['Device'])
        df_t = df.melt(
            id_vars=['Timestamp','DeviceName','Interpolated'],
            value_vars=['Temp_F'], var_name='Metric'
        )
        line_temp = alt.Chart(df_t).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q', title='Temperature (°F)'), color='DeviceName:N'
        ).properties(title='Temperature Data')
        pts_temp = alt.Chart(df_t[df_t['Interpolated']]).mark_circle(size=50, color='red').encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y='value:Q'
        )
        st.altair_chart(line_temp + pts_temp, use_container_width=True)

        # Relative Humidity plot
        df_r = df.melt(id_vars=['Timestamp','DeviceName','Interpolated'], value_vars=['RH'], var_name='Metric')
        line_rh = alt.Chart(df_r).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('value:Q', title='Relative Humidity (%)'), color='DeviceName:N'
        ).properties(title='Relative Humidity Data')
        pts_rh = alt.Chart(df_r[df_r['Interpolated']]).mark_circle(size=50, color='red').encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y='value:Q'
        )
        st.altair_chart(line_rh + pts_rh, use_container_width=True)

        # Normalized Temperature Difference plot
        df_out = df[df['Device']=='AS10'][['Timestamp','Temp_F','RH']].rename(columns={'Temp_F':'T_out','RH':'RH_out'})
        df_norm = df.merge(df_out, on='Timestamp')
        df_norm['DeviceName'] = df_norm['Device'].map(DEVICE_LABELS).fillna(df_norm['Device'])
        df_norm['Norm_T']  = df_norm['Temp_F'] - df_norm['T_out']
        chart_norm_t = alt.Chart(df_norm).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('Norm_T:Q', title='Temp Difference (°F)'), color='DeviceName:N'
        ).properties(title='Normalized Temperature Difference')
        st.altair_chart(chart_norm_t, use_container_width=True)

        # Normalized Relative Humidity Difference plot
        df_norm['Norm_RH'] = df_norm['RH'] - df_norm['RH_out']
        chart_norm_rh = alt.Chart(df_norm).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y('Norm_RH:Q', title='RH Difference (%)'), color='DeviceName:N'
        ).properties(title='Normalized Relative Humidity Difference')
        st.altair_chart(chart_norm_rh, use_container_width=True)

        # Correlation matrices
        st.header('Correlation Matrix (Temperature)')
        corr_t = compute_correlations(df, field='Temp_F')
        df_ct = (
            corr_t.reset_index().rename(columns={'index':'DeviceName'})
            .melt(id_vars='DeviceName', var_name='DeviceName2', value_name='Corr')
        )
        heat_t = alt.Chart(df_ct).mark_rect().encode(
            x='DeviceName2:O', y='DeviceName:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heat_t, use_container_width=False)

        st.header('Correlation Matrix (Relative Humidity)')
        corr_h = compute_correlations(df, field='RH')
        df_ch = (
            corr_h.reset_index().rename(columns={'index':'DeviceName'})
            .melt(id_vars='DeviceName', var_name='DeviceName2', value_name='Corr')
        )
        heat_h = alt.Chart(df_ch).mark_rect().encode(
            x='DeviceName2:O', y='DeviceName:O', color='Corr:Q'
        ).properties(width=400, height=400)
        st.altair_chart(heat_h, use_container_width=False)

        # Corr vs AS10 tables
        st.header('Pearson Corr vs Outdoor Reference (Temp)')
        cvt = compute_correlations(df, field='Temp_F')['Outdoor-Reference']
        st.table(cvt.reset_index().rename(columns={'index':'DeviceName','Outdoor-Reference':'Corr'}))

        st.header('Pearson Corr vs Outdoor Reference (RH)')
        cvr = compute_correlations(df, field='RH')['Outdoor-Reference']
        st.table(cvr.reset_index().rename(columns={'index':'DeviceName','Outdoor-Reference':'Corr'}))

        # Summary Stats
        st.header('Summary Statistics (Temperature)')
        st.dataframe(compute_summary_stats(df, field='Temp_F'))
        st.header('Summary Statistics (Relative Humidity)')
        st.dataframe(compute_summary_stats(df, field='RH'))
else:
    st.info("Use 'Load Data' then 'Analyze' to run the full analysis.")
