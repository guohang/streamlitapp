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
import re


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
                                            (data['z-axis_normalized_zscore'].abs() > zscore_threshold) &
                                            (data['z-axis_normalized_rate_of_change'] > 0)]
    positive_potential_peaks_indices_normalized = positive_potential_peaks_normalized.index

    negative_potential_peaks_normalized = data[(data['z-axis_normalized_rate_of_change'].abs() > data['z-axis_normalized_rate_of_change'].quantile(0.7)) &
                                            (data['z-axis_normalized_zscore'].abs() > zscore_threshold) &
                                            (data['z-axis_normalized_rate_of_change'] < 0)]
    negative_potential_peaks_indices_normalized = negative_potential_peaks_normalized.index

    return (positive_potential_peaks_indices_normalized, negative_potential_peaks_indices_normalized)

def combined_acceleration_and_time(df):
    
    if 'Acceleration X(g)' in df.columns and 'Acceleration Y(g)' in df.columns and 'Acceleration Z(g)' in df.columns:
        df['combined acceleration'] = np.sqrt(df['Acceleration X(g)']**2 + df['Acceleration Y(g)']**2 + df['Acceleration Z(g)']**2)
        st.write("Successfully calculated combined acceleration.")

    else:
        st.error("Required columns ('Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)') not found.")
        st.stop()
        
    if 'Time' in df.columns:
        # Convert 'Time' column to datetime objects.
        # errors='coerce' will turn unparsable values into NaT (Not a Time).
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        
        # Check if the first value is a valid timestamp before calculating the time difference.
        if pd.notna(df['Time'][0]):
            # Calculate time difference from the start of the data
            # This will result in a Timedelta object.
            df['Time'] = (df['Time'] - df['Time'][0]).dt.total_seconds()
        else:
            st.error("The first timestamp in the 'Time' column is not a valid format.")
            st.stop()
    else:
        st.error("The 'Time' column is not found in the uploaded data.")
        st.stop()
    
    return df

def peak_and_chart(df, find_peaks_parameter, chart_name_1, chart_name_2):
    
    peaks, _ = find_peaks(df['combined acceleration'], **find_peaks_parameter)

    st.write("Number of detected events: " + str(len(peaks)))

    # Visualization
    fig = go.Figure()
    trace = go.Scatter(x=df['Time'], y=df['combined acceleration'], mode='lines', fill='tozeroy', name='Time Series')
    
    # Check if peaks were found
    if len(peaks) > 0:
        marked_trace = go.Scatter(x=df['Time'].iloc[peaks], y=df['combined acceleration'].iloc[peaks],
                                mode='markers', marker=dict(size=10, color='Red', opacity=0.8), name='Marked Points')
        fig = go.Figure([trace, marked_trace])
    else:
        fig = go.Figure([trace])

    fig.update_layout(title=chart_name_1, xaxis_title='Time', yaxis_title='Combined Acceleration')
    st.plotly_chart(fig)

    result = []
    if len(peaks) > 1:
        for j in range(len(peaks) - 1):
            start_index = peaks[j] + 1
            end_index = peaks[j + 1]
            slice_df = df.iloc[start_index:end_index]
            result.append(slice_df['combined acceleration'].sum())
        
        if result:
            fig_bar = go.Figure([go.Bar(y=result, width=0.5)])
            fig_bar.update_layout(title=chart_name_2, yaxis_title='Total Acceleration from One Push', template='plotly_white')
            st.plotly_chart(fig_bar)
        else:
            st.write("Not enough peaks to generate a bar chart for total acceleration.")

def detect_push_peaks(df, find_peaks_parameter, chart_name_1, chart_name_2):
    
    if 'combined acceleration' not in df.columns:
        st.error("Please ensure combined acceleration is calculated before detecting peaks.")
        return

    push_peaks, _ = find_peaks(df['combined acceleration'], **find_peaks_parameter)
    st.write("Number of detected events (Push): " + str(len(push_peaks)))

    if len(push_peaks) > 0:
        fig = go.Figure()
        trace = go.Scatter(x=df['Time'], y=df['combined acceleration'], mode='lines', fill='tozeroy',name='Time Series')
        marked_trace = go.Scatter(x=df['Time'].iloc[push_peaks], y=df['combined acceleration'].iloc[push_peaks],
                                mode='markers', marker=dict(size=10, color='Red', opacity=0.8), name='Marked Points')
        fig = go.Figure([trace, marked_trace])
        fig.update_layout(title= chart_name_1, xaxis_title='Time', yaxis_title='Combined Acceleration')
        st.plotly_chart(fig)

        result = []
        if len(push_peaks) > 1:
            for j in range(len(push_peaks) - 1):
                start_index = push_peaks[j] + 1
                end_index = push_peaks[j + 1]
                slice_df = df.iloc[start_index:end_index]
                result.append(slice_df['combined acceleration'].sum())
            
            if result:
                fig_bar = go.Figure([go.Bar(y=result, width=0.5)])
                fig_bar.update_layout(title= chart_name_2, yaxis_title='Total Acceleration from One Push', template='plotly_white')
                st.plotly_chart(fig_bar)
            else:
                st.write("Not enough push peaks to generate a bar chart.")
    else:
        st.write("No peaks detected. Cannot generate chart.")


st.title('Body-Mounted Accelerometer Data Visualizer')
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=['csv', 'txt'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, skipinitialspace=True)
        # Clean column names by stripping leading/trailing whitespace
        df.columns = df.columns.str.strip()
        st.write("### Raw Data")
        st.dataframe(df)

        df = combined_acceleration_and_time(df)
        
        st.write("### Filtered and Preprocessed Data")
        st.dataframe(df)
        
        # Streamlit sidebar for filter controls
        st.sidebar.header('Filter Controls')
        selected_option = st.sidebar.selectbox('Select Data Type:', ['Combined Acceleration', 'Z-Axis Data'])
        
        if selected_option == 'Combined Acceleration':
            st.write("### Combined Acceleration Analysis")
            
            # Use `st.columns` for side-by-side layout
            col1, col2 = st.columns(2)
            with col1:
                height = st.number_input('Height (m):', value=0.1, step=0.01)
            with col2:
                distance = st.number_input('Distance (m):', value=0.1, step=0.01)
            
            find_peaks_parameter = {
                'height': height,
                'distance': int(distance)
            }
            
            peak_and_chart(df, find_peaks_parameter, 'Combined Acceleration Peaks Over Time', 'Total Acceleration for Each Peak')

        elif selected_option == 'Z-Axis Data':
            st.write("### Z-Axis Corner Detection Analysis")
            
            if 'Acceleration Z(g)' in df.columns:
                
                # Use `st.columns` for side-by-side layout
                col1, col2 = st.columns(2)
                with col1:
                    height = st.number_input('Height (m):', value=0.1, step=0.01, key='z_height')
                with col2:
                    distance = st.number_input('Distance (m):', value=0.1, step=0.01, key='z_distance')
                
                find_peaks_parameter = {
                    'height': height,
                    'distance': int(distance)
                }

                # Assuming 'Acceleration Z(g)' is the correct column name based on the CSV
                detect_push_peaks(df, find_peaks_parameter, 'Push Peaks Over Time', 'Total Acceleration from One Push')
            else:
                st.error("The 'Acceleration Z(g)' column is not found in the uploaded data.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
