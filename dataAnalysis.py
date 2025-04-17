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
            s.iloc[pad_start:pad_end+1] = segment.interpolate(method='linear')
    return s

# --- Streamlit App ---
st.set_page_config(page_title="All Souls Cathedral: Q1 Environmental Data Analysis", layout="wide")
st.sidebar.title("Settings")

# Sidebar inputs
folder = st.sidebar.text_input("Data folder path", value="./data")
start_date = st.sidebar.date_input("Start date", value=datetime(2025,1,1))
end_date = st.sidebar.date_input("End date", value=datetime.today())

# Load button: ingest and preprocess data
if st.sidebar.button("Load Data"):
    files = glob.glob(os.path.join(folder, "AS*_export_*.csv"))
    device_dfs = {}
    for f in files:
        df = load_and_clean_csv(f)
        device_dfs[df['Device'].iloc[0]] = df

    # Determine master timeline
    master = max(device_dfs, key=lambda d: len(device_dfs[d]))
    master_index = device_dfs[master].sort_values('Timestamp')['Timestamp']

    # Reindex and fill gaps
    processed = []
    for device, df in device_dfs.items():
        tmp = df.set_index('Timestamp').reindex(master_index)
        tmp['Temp_F'] = fill_small_gaps(tmp['Temp_F'])
        tmp['RH'] = fill_small_gaps(tmp['RH'])
        tmp['Device'] = device
        processed.append(tmp.reset_index().rename(columns={'index': 'Timestamp'}))
    df_all = pd.concat(processed, ignore_index=True)

    # Store in session state
    st.session_state['df_all'] = df_all
    st.session_state['devices'] = sorted(df_all['Device'].unique())

# Device selection
devices = st.session_state.get('devices', [])
selected = st.sidebar.multiselect("Select devices to analyze", options=devices, default=devices)

# Analyze button: filter and plot
if st.sidebar.button("Analyze"):
    if 'df_all' not in st.session_state:
        st.error("Please click 'Load Data' first to ingest CSV files.")
    else:
        df_all = st.session_state['df_all']
        # Filter by selected devices and date range
        df_sel = df_all[df_all['Device'].isin(selected)]
        mask = (df_sel['Timestamp'].dt.date >= start_date) & (df_sel['Timestamp'].dt.date <= end_date)
        df_sel = df_sel.loc[mask]

        # Plot time series for all selected devices
        st.header("Temperature (°F) Time Series for Selected Devices")
        ts = df_sel.pivot(index='Timestamp', columns='Device', values='Temp_F')
        st.line_chart(ts)
else:
    if 'df_all' not in st.session_state:
        st.info("Enter the data folder path, then click 'Load Data' to begin.")
    else:
        st.info("Devices loaded. Use the multiselect and click 'Analyze' to view the plot.")
