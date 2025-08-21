# -*- coding: utf-8 -*-

# imports and utilities
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import zscore
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Apply a Butterworth low-pass filter to the input data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def corner_detection(data, column_name):
    zscore_threshold = 1
    data['z-axis_normalized'] = (data[column_name] - data[column_name].min()) / (data[column_name].max() - data[column_name].min())
    data['z-axis_normalized_rate_of_change'] = data['z-axis_normalized'].diff()
    data['z-axis_normalized_zscore'] = zscore(data['z-axis_normalized'])

    positive_potential_peaks_normalized = data[(data['z-axis_normalized_rate_of_change'].abs() > data['z-axis_normalized_rate_of_change'].quantile(0.7)) &
                                            (data['z-axis_normalized_zscore'].abs() < zscore_threshold)]

    peaks_indices = find_peaks(positive_potential_peaks_normalized['z-axis_normalized'], height=0.5)[0]
    
    return positive_potential_peaks_normalized.iloc[peaks_indices].index

def process_data(df, chart_name_1, chart_name_2):
    """
    Process the acceleration data and generate plots.
    """
    # Define columns to be used
    columns_to_process = ['Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)']
    
    # Check for required columns
    if not all(col in df.columns for col in columns_to_process):
        st.error("The uploaded file does not contain the required columns: 'Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)'.")
        return

    # Convert columns to numeric, coercing errors to NaN
    for col in columns_to_process:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in the processed columns
    df.dropna(subset=columns_to_process, inplace=True)
    
    # Check if DataFrame is empty after dropping rows
    if df.empty:
        st.error("No valid numeric data found in the required columns after cleaning. Please check your data file.")
        return

    # Filter and process data for push analysis
    df['combined acceleration'] = np.sqrt(df[columns_to_process[0]]**2 + df[columns_to_process[1]]**2 + df[columns_to_process[2]]**2)
    fs = 120
    cutoff = 5
    order = 2
    df['combined acceleration filtered'] = butter_lowpass_filter(df['combined acceleration'], cutoff, fs, order)

    push_peaks = corner_detection(df, 'Acceleration Z(g)')
    
    if len(push_peaks) > 1:
        fig = go.Figure()
        trace = go.Scatter(x=df.index, y=df['combined acceleration'], mode='lines', fill='tozeroy',name='Time Series')
        marked_trace = go.Scatter(x=df.index[push_peaks], y=df['combined acceleration'].iloc[push_peaks],
                                mode='markers', marker=dict(size=10, color='Red', opacity=0.8), name='Marked Points')
        fig = go.Figure([trace, marked_trace])
        fig.update_layout(title= chart_name_1, xaxis_title='Time', yaxis_title='Combined Acceleration')
        st.plotly_chart(fig)

        result = []
        for j in range(len(push_peaks) - 1):
            start_index = push_peaks[j] + 1
            end_index = push_peaks[j + 1]
            slice_df = df.iloc[start_index:end_index]
            result.append(slice_df['combined acceleration'].sum())
        fig_bar = go.Figure([go.Bar(y=result, width=0.5)])
        fig_bar.update_layout(title= chart_name_2, yaxis_title='Total Acceleration from One Push', template='plotly_white')
        st.plotly_chart(fig_bar)


st.title('Body-Mounted Accelerometer Data Visualizer')
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=['csv', 'txt'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        process_data(df, 'Pushing Detection', 'Total Acceleration from Each Push')
    except Exception as e:
        st.error(f"Error reading the file: {e}")
