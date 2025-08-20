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
    # Corrected code to convert column to numeric
    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
    data['z-axis_normalized'] = (data[column_name] - data[column_name].min()) / (data[column_name].max() - data[column_name].min())
    data['z-axis_normalized_rate_of_change'] = data['z-axis_normalized'].diff()
    data['z-axis_normalized_zscore'] = zscore(data['z-axis_normalized'])
    positive_potential_peaks_normalized = data[(data['z-axis_normalized_rate_of_change'].abs() > data['z-axis_normalized_rate_of_change'].quantile(0.7)) &
                                            (data['z-axis_normalized_zscore'] > zscore_threshold) &
                                            (data['z-axis_normalized'] > 0)]['z-axis_normalized']
    filtered_peaks_indices_normalized = []
    last_peak_index = 0
    min_distance_between_peaks = 700
    for index in positive_potential_peaks_normalized.index:
        if index - last_peak_index >= min_distance_between_peaks:
            filtered_peaks_indices_normalized.append(index)
            last_peak_index = index
    refined_peaks_normalized = data.loc[filtered_peaks_indices_normalized, column_name]
    return refined_peaks_normalized


def zero_crossing(df, column_name):
    df['Shifted'] = df[column_name].shift(1)
    df['Value'] = df[column_name]
    df['Neg_to_Pos'] = (df['Value'] > 0) & (df['Shifted'] < 0)
    df['Pos_to_Neg'] = (df['Value'] < 0) & (df['Shifted'] > 0)
    neg_to_pos_df = df[df['Neg_to_Pos']]
    pos_to_neg_df = df[df['Pos_to_Neg']]
    result = pd.concat([pos_to_neg_df, neg_to_pos_df])
    result = result.sort_index()
    result.rename(columns={'Neg_to_Pos': 'corner_entrance'}, inplace=True)
    result.rename(columns={'Pos_to_Neg': 'corner_exit'}, inplace=True)
    return result


def find_entrance_exit(peaks, entrance_exit):
    df = entrance_exit
    start_idx = min(peaks.index)
    end_idx = max(peaks.index)
    selected_df = df.loc[start_idx:end_idx]
    all_indices = df.index.tolist()
    selected_indices = selected_df.index.tolist()
    before_after_indices = []
    for idx in selected_indices:
        idx_pos = all_indices.index(idx)
        if idx_pos > 0:
            before_after_indices.append(all_indices[idx_pos - 1])
        before_after_indices.append(idx)
        if idx_pos < len(all_indices) - 1:
            before_after_indices.append(all_indices[idx_pos + 1])
    final_indices = sorted(set(before_after_indices))
    final_selection = df.loc[final_indices]
    return final_selection


def apply_filter_corner(data, column_name, cutoff):
    cutoff = cutoff
    fs = 200
    order = 1
    filtered_signal = butter_lowpass_filter(data[column_name], cutoff, fs, order)
    result = data.copy()
    result[column_name] = filtered_signal
    return result


def push_detection(data, column_name, sampling_rate):
    peaks, _ = find_peaks(data[column_name], prominence=0.5, distance=sampling_rate / 4)
    return peaks


def apply_filter_push(data, column_name, sampling_rate):
    cutoff = 5
    fs = sampling_rate
    order = 5
    filtered_signal = butter_lowpass_filter(data[column_name], cutoff, fs, order)
    data['filtered_signal'] = filtered_signal
    return data


def entrance_exit_function(df_source):
    data = apply_filter_corner(df_source.copy(), 'Angle Y(°)', 0.5)
    peaks = corner_detection(df_source, 'Angle Y(°)')
    zero_cross = zero_crossing(data, 'Angle Y(°)')
    entrance_exit = find_entrance_exit(peaks, zero_cross)
    return entrance_exit


def straight_push_index(entrance_exit):
    df = entrance_exit
    exit_indices = df.index[df['corner_exit']].tolist()
    entrance_indices = df.index[df['corner_entrance']].tolist()
    pairs = []
    for i in range(len(exit_indices) - 1):
        current_exit_idx = exit_indices[i]
        for j in range(len(entrance_indices)):
            if entrance_indices[j] > current_exit_idx:
                pairs.append((current_exit_idx, entrance_indices[j]))
                break
    return pairs


def combined_frontal_acceleration(df):
    # The following line has been updated with the correct column names.
    df['combined acceleration'] = (df['Acceleration Z(g)']**2 + df['Acceleration Y(g)']**2 + df['Acceleration X(g)']**2)**0.5
    return df


def visualization(df):
    fig = make_subplots(rows=7, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df['Acceleration Z(g)'], name='z-axis (g)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Acceleration X(g)'], name='x-axis (g)', marker_color='orange'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=-df['Acceleration Y(g)'], name='-y-axis (g)', marker_color='green'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Angular velocity Z(°/s)'], name='z-axis (Deg/s)', marker_color='red'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Angle X(°)'], name='Angle X(°)', marker_color='cyan'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Angle Y(°)'], name='Angle Y(°)', marker_color='yellow'), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['combined acceleration'], name='combined acceleration', marker_color='black'), row=7, col=1)
    fig.update_layout(height=900, width=700, showlegend=False)
    fig.update_xaxes(title_text="センサーデータの可視化", row=3, col=1)
    fig.update_yaxes(title_text="z-axis (g)", row=1, col=1)
    fig.update_yaxes(title_text="x-axis (g)", row=2, col=1)
    fig.update_yaxes(title_text="-y-axis (g)", row=3, col=1)
    fig.update_yaxes(title_text="z-axis (Deg/s)", row=4, col=1)
    fig.update_yaxes(title_text="Angle X(°)", row=5, col=1)
    fig.update_yaxes(title_text="Angle Y(°)", row=6, col=1)
    fig.update_yaxes(title_text="combined acceleration", row=7, col=1)
    fig.update_layout(width=1500, height=1000)
    return fig


st.title('Body-Mounted Accelerometer Data Visualizer')
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=['csv', 'txt'])


if uploaded_file is not None:
    try:
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
        df = df.reset_index()
        df = combined_frontal_acceleration(df)
        entrance_exit = entrance_exit_function(df)
        st.write('Data loaded')

        st.subheader("Visualizations")
        visualization(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
