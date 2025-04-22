import os
import re
import glob
from datetime import datetime

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# --- Device name mapping ---
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

# --- Zone mapping for group summaries ---
ZONE_MAP = {
    **{f'AS{i:02d}': 'Main' for i in range(1,15) if i != 10},
    **{f'AS{i:02d}': 'Attic' for i in range(15,24)},
    **{f'AS{i:02d}': 'Crawlspace' for i in range(24,34)},
    'AS10': 'Outdoor'
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


def compute_dew_point_f(temp_f, rh):
    # Magnus formula
    temp_c = (temp_f - 32) * 5/9
    a, b = 17.625, 243.04
    alpha = np.log(rh/100) + (a * temp_c)/(b + temp_c)
    dew_c = (b * alpha) / (a - alpha)
    return dew_c * 9/5 + 32


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

@st.cache_data
def make_calendar_pivot(df, metric):
    df = df.assign(
        hour=df.Timestamp.dt.hour,
        date=df.Timestamp.dt.date
    )
    pivot = df.pivot_table(
        index='hour', columns='date', values=metric, aggfunc='mean'
    ).reset_index().melt(id_vars='hour', var_name='date', value_name=metric)
    return pivot

@st.cache_data
def compute_group_means(df):
    df = df.copy()
    df['Zone'] = df['Device'].map(ZONE_MAP)
    return df.groupby(['Timestamp','Zone']).Temp_F.mean().reset_index()

# --- Streamlit App ---
st.set_page_config(page_title='All Souls Cathedral: Q1 Environmental Data Analysis', layout='wide')
st.header('All Souls Cathedral: Q1 Environmental Data Analysis')

# Sidebar settings
st.sidebar.title('Settings')
FOLDER = './data'

# Date selectors
start_date = st.sidebar.date_input('Start Date', value=datetime(2025,1,1))
end_date   = st.sidebar.date_input('End Date',   value=datetime.today())

# Metric selection
metrics = st.sidebar.multiselect(
    'Select Metrics',
    ['Temperature','RH','Dew Point'],
    default=['Temperature','RH']
)

# Threshold inputs
st.sidebar.markdown('**Thresholds**')
temp_min = st.sidebar.number_input('Temp Min (°F)', value=68.0)
temp_max = st.sidebar.number_input('Temp Max (°F)', value=74.0)
rh_min   = st.sidebar.number_input('RH Min (%)',   value=30.0)
rh_max   = st.sidebar.number_input('RH Max (%)',   value=60.0)

# Load data
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
        ft, fl = fill_and_flag(tmp['Temp_F'])
        fr, frl = fill_and_flag(tmp['RH'])
        tmp['Temp_F'], tmp['RH'] = ft, fr
        tmp['Interpolated'] = fl | frl
        tmp['Device'] = dev
        tmp['DeviceName'] = DEVICE_LABELS.get(dev, dev)
        # Dew point
        tmp['Dew_Point'] = compute_dew_point_f(tmp['Temp_F'], tmp['RH'])
        records.append(tmp.reset_index())
    df_all = pd.concat(records, ignore_index=True)
    st.session_state.df_all = df_all
    st.session_state.devices = sorted(df_all['Device'])

# Ensure data loaded
if 'df_all' not in st.session_state:
    st.info("Use 'Load Data' to begin.")
    st.stop()

# Filter by date & device
df = st.session_state.df_all
devices = st.session_state.devices
# Device selection (existing checkboxes)
for dev in devices:
    key = f'chk_{dev}'
    st.sidebar.checkbox(DEVICE_LABELS.get(dev,dev), key=key, value=st.session_state.get(key,True))
selected = [d for d in devices if st.session_state.get(f'chk_{d}', False)]
df = df[df['Device'].isin(selected)]
df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]

# Tabs for Tiered Dashboard
tabs = st.tabs(['Summary','Time Series & Trends','Diagnostics'])

# --- Tab 1: Summary Cards ---
with tabs[0]:
    st.subheader('Insights at a Glance')
    # Comfort % time in zone
    comfort_df = df.copy()
    comfort_df['InZone'] = (
        (comfort_df.Temp_F.between(temp_min,temp_max)) &
        (comfort_df.RH.between(rh_min,rh_max))
    )
    pct_in = comfort_df.groupby('DeviceName').InZone.mean().reset_index()
    cols = st.columns(3)
    for i, row in pct_in.iterrows():
        if i<3:
            cols[i].metric(row['DeviceName'], f"{row['InZone']*100:.1f}%")
    # Condensation events
    df['Condensation'] = (df['Temp_F'] <= df['Dew_Point'])
    cond_events = df[df['Condensation']].groupby('DeviceName').size().reset_index(name='Events')
    for j, row in cond_events.iterrows():
        if j<3:
            cols[j].metric(f"{row['DeviceName']} Cond Events", int(row['Events']))

# --- Tab 2: Interactive Time Series & Trends ---
with tabs[1]:
    st.subheader('Time Series')
    # Generic plot function
    def plot_timeseries(df, metric, threshold=None):
        y_field = {
            'Temperature':'Temp_F',
            'RH':'RH',
            'Dew Point':'Dew_Point'
        }[metric]
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('Timestamp:T', axis=alt.Axis(format='%m/%d')),
            y=alt.Y(f'{y_field}:Q', title=f'{metric}') ,
            color='DeviceName:N'
        )
        # threshold bands
        if threshold:
            low, high = threshold
            band = alt.Chart(df).mark_rect(opacity=0.1).encode(
                x='Timestamp:T', x2='Timestamp:T',
                y=alt.value(low), y2=alt.value(high)
            )
            return band + chart
        return chart

    for m in metrics:
        thr = None
        if m=='Temperature': thr=(temp_min,temp_max)
        if m=='RH': thr=(rh_min,rh_max)
        st.altair_chart(plot_timeseries(df, m, thr), use_container_width=True)
    # Daily aggregates
    daily = df.set_index('Timestamp').groupby([pd.Grouper(freq='D'))]['Temp_F','RH'].agg(['min','max','mean'])
    st.line_chart(daily)

# --- Tab 3: Diagnostics ---
with tabs[2]:
    st.subheader('Calendar Heatmap')
    cal = make_calendar_pivot(df, 'Temp_F')
    heat = alt.Chart(cal).mark_rect().encode(
        x='hour:O', y='date:O', color='Temp_F:Q'
    ).properties(height=300)
    st.altair_chart(heat, use_container_width=True)

    st.subheader('Zone Averages ± Std Dev')
    grp = compute_group_means(df)
    band = alt.Chart(grp).mark_errorband().encode(
        x='Timestamp:T', y='Temp_F:Q', color='Zone:N'
    )
    line = alt.Chart(grp).mark_line().encode(
        x='Timestamp:T', y='Temp_F:Q', color='Zone:N'
    )
    st.altair_chart(band+line, use_container_width=True)

    st.subheader('Boxplots & Histograms')
    box = alt.Chart(df).mark_boxplot().encode(
        x='DeviceName:N', y='Temp_F:Q'
    ).properties(height=200)
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X('Temp_F:Q', bin=True), y='count()'
    ).properties(height=200)
    st.altair_chart(box, use_container_width=True)
    st.altair_chart(hist, use_container_width=True)
