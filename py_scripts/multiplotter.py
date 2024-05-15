# multiplotter.py

import pandas as pd
import plotly.graph_objects as go
import random

def random_color(existing_colors):
    while True:
        color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
        if color not in existing_colors:
            return color

def plot_combined(files=None, dfs=None, labels=None):
    """
    Plots data from multiple sources (files and/or DataFrames) on the same canvas.
    
    Parameters:
    - files (list, optional): List of file paths to be read and plotted.
    - dfs (list, optional): List of DataFrames to be plotted.
    - labels (list, optional): List of labels corresponding to each source for the legend. 
                               If not provided for DataFrames, it will use indices as labels.
    """
    # Predefined set of well-separated colors
    predefined_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Create a blank figure
    fig = go.Figure()

    # Used colors set to ensure we don't repeat colors
    used_colors = set()

    all_data = []

    # Load data from provided file paths
    if files:
        for file in files:
            df = pd.read_csv(file, delimiter='\t')
            all_data.append((df, file))

    # Add directly provided DataFrames
    if dfs:
        if not labels:
            labels = range(len(dfs))
        for df, label in zip(dfs, labels):
            all_data.append((df, label))

    # Iterate over all data sources to plot
    for df, label in all_data:
        # Choose a color from the predefined set or generate a new one if we've exhausted the predefined set
        if predefined_colors:
            color = predefined_colors.pop()
        else:
            color = random_color(used_colors)
        used_colors.add(color)

        for index, row in df.iterrows():
            show_in_legend = True if index == 0 else False  # Show legend only for the first trace of each source
            fig.add_trace(go.Scatter(x=[row['time_start'], row['time_stop']],
                                     y=[row['freq_start'], row['freq_stop']],
                                     mode='lines+markers',
                                     line=dict(width=2, color=color),
                                     name=str(label),               # Set the name of the trace to the label for the legend
                                     legendgroup=str(label),        # Use legendgroup to group traces of the same source
                                     showlegend=show_in_legend      # Show legend only for the first trace of each source
                                    ))

    # Update layout
    fig.update_layout(
        title='Time vs Frequency for Each Event',
        xaxis_title='Time (s)',
        yaxis_title='Frequency',
        template='plotly_white',
        width=900,
        height=700
    )

    # Show figure
    fig.show()

def plot_combined_3d(files=None, dfs=None, labels=None):
    """
    Plots data from multiple sources (files and/or DataFrames) on a 3D canvas.
    
    Parameters:
    - files (list, optional): List of file paths to be read and plotted.
    - dfs (list, optional): List of DataFrames to be plotted.
    - labels (list, optional): List of labels corresponding to each source for the legend. 
                               If not provided for DataFrames, it will use indices as labels.
    """
    predefined_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    used_colors = set()
    all_data = []

    if files:
        for file in files:
            df = pd.read_csv(file, delimiter='\t')
            all_data.append((df, file))

    if dfs:
        if not labels:
            labels = range(len(dfs))
        for df, label in zip(dfs, labels):
            all_data.append((df, label))

    fig_3d = go.Figure()

    for df, label in all_data:
        if predefined_colors:
            color = predefined_colors.pop()
        else:
            color = random_color(used_colors)
        used_colors.add(color)

        for _, row in df.iterrows():
            show_in_legend = True if _ == 0 else False
            fig_3d.add_trace(go.Scatter3d(
                x=[row['time_start'], row['time_stop']],
                y=[row['amp_min'], row['amp_max']],
                z=[row['freq_start'], row['freq_stop']],
                mode='lines+markers',
                line=dict(width=2, color=color),
                marker=dict(size=4, color=color),
                name=str(label),
                legendgroup=str(label),
                showlegend=show_in_legend
            ))

    fig_3d.update_layout(
        title='Time vs Amplitude vs Frequency for Each Event (3D)',
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            zaxis_title='Frequency',
            xaxis=dict(range=[0, max(df['time_stop'].max() for df, _ in all_data)]),
            yaxis=dict(range=[0, max(df['amp_max'].max() for df, _ in all_data)]),
            zaxis=dict(range=[0, max(df['freq_stop'].max() for df, _ in all_data)]),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.5)
            )
        ),
        template='plotly_white',
        showlegend=True,
        width=900,
        height=700
    )

    fig_3d.show()

def plot_scatter(files=None, dfs=None, labels=None, mode='markers+lines'):
    """
    Plots data from multiple sources (files and/or DataFrames) on a scatter plot.
    
    Parameters:
    - files (list, optional): List of file paths to be read and plotted.
    - dfs (list, optional): List of DataFrames to be plotted.
    - labels (list, optional): List of labels corresponding to each source for the legend. 
                               If not provided for DataFrames, it will use indices as labels.
    - mode (str, optional): Mode for plotting. Can be 'lines', 'markers', or 'markers+lines'.
                             Default is 'markers+lines'.
    """
    predefined_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    used_colors = set()
    all_data = []

    if files:
        for file in files:
            df = pd.read_csv(file, delimiter='\t')
            all_data.append((df, file))

    if dfs:
        if not labels:
            labels = range(len(dfs))
        for df, label in zip(dfs, labels):
            all_data.append((df, label))

    scatter_fig = go.Figure()

    for df, label in all_data:
        if predefined_colors:
            color = predefined_colors.pop()
        else:
            color = random_color(used_colors)
        used_colors.add(color)

        scatter_fig.add_trace(go.Scatter(
            x=df['Frequency (Hz)'], 
            y=df['Amplitude'], 
            mode=mode, 
            name=str(label), 
            line=dict(color=color)
        ))

    # Calculate the maximum values with 10% padding
    max_x = max(df['Frequency (Hz)'].max() for df, _ in all_data) * 1.10
    max_y = max(df['Amplitude'].max() for df, _ in all_data) * 1.10

    scatter_fig.update_layout(title="Scatter Plot of Frequency vs. Amplitude")
    scatter_fig.update_xaxes(range=[0, max_x])
    scatter_fig.update_yaxes(range=[0, max_y])
    scatter_fig.show()