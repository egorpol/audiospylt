# multiplotter.py

import pandas as pd
import plotly.graph_objects as go
import random
from typing import List, Optional, Tuple, Any, Iterator, Dict

# Predefined set of well-separated colors
PREDEFINED_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    # Added a few more common ones
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]

def _color_palette_generator() -> Iterator[str]:
    """
    Yields colors from a predefined list, then generates random unique RGB colors.
    """
    used_colors = set()
    for color in PREDEFINED_COLORS:
        if color not in used_colors:
            used_colors.add(color)
            yield color
    
    # Fallback to random colors if predefined are exhausted
    while True:
        color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"
        if color not in used_colors:
            used_colors.add(color)
            yield color

def _load_and_prepare_data(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None
) -> List[Tuple[pd.DataFrame, str]]:
    """
    Loads data from files and combines with provided DataFrames.
    Generates labels if necessary.
    """
    all_data: List[Tuple[pd.DataFrame, str]] = []

    if files:
        for i, file_path in enumerate(files):
            try:
                df = pd.read_csv(file_path, delimiter='\t')
                # Use filename as label if no specific labels for files are handled elsewhere
                label = file_path # Or some derivation like os.path.basename(file_path)
                all_data.append((df, label))
            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except pd.errors.EmptyDataError:
                print(f"Warning: File is empty: {file_path}")
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")


    if dfs:
        # Generate default labels for DataFrames if not provided or insufficient
        num_dfs = len(dfs)
        current_df_labels: List[str] = []
        if df_labels and len(df_labels) >= num_dfs :
            current_df_labels = df_labels[:num_dfs]
        elif df_labels: # some labels provided, but not enough
             current_df_labels = df_labels + [f"DataFrame {i}" for i in range(len(df_labels), num_dfs)]
        else: # no labels provided
            current_df_labels = [f"DataFrame {i}" for i in range(num_dfs)]

        for df, label in zip(dfs, current_df_labels):
            all_data.append((df, str(label))) # Ensure label is string

    return all_data


def plot_combined(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None
) -> None:
    """
    Plots data from multiple sources (files and/or DataFrames) on the same 2D canvas.
    Each row in a DataFrame is treated as a separate line segment.

    Parameters:
    - files (list, optional): List of file paths (TSV) to be read and plotted.
    - dfs (list, optional): List of DataFrames to be plotted.
    - df_labels (list, optional): List of labels corresponding to each DataFrame in `dfs`.
                                 If not provided, DataFrames will be labeled "DataFrame 0", "DataFrame 1", etc.
                                 File paths are used as labels for file-based sources.
    """
    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot.")
        return

    fig = go.Figure()
    color_gen = _color_palette_generator()

    for df, label in all_data:
        if df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping.")
            continue
        
        # Check for required columns
        required_cols = ['time_start', 'time_stop', 'freq_start', 'freq_stop']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing one or more required columns: {required_cols}. Skipping.")
            continue

        current_color = next(color_gen)
        for index, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['time_start'], row['time_stop']],
                y=[row['freq_start'], row['freq_stop']],
                mode='lines+markers',
                line=dict(width=2, color=current_color),
                marker=dict(size=5),
                name=str(label),
                legendgroup=str(label),
                showlegend=(index == 0)  # Show legend only for the first trace of this source
            ))

    fig.update_layout(
        title_text='Time vs Frequency for Each Event',
        xaxis_title_text='Time (s)',
        yaxis_title_text='Frequency (Hz)',
        template='plotly_white',
        width=900,
        height=700,
        legend_title_text='Sources'
    )
    fig.show()


def plot_combined_3d(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None
) -> None:
    """
    Plots data from multiple sources (files and/or DataFrames) on a 3D canvas.
    Each row in a DataFrame is treated as a separate line segment in 3D.

    Parameters:
    - files (list, optional): List of file paths (TSV) to be read and plotted.
    - dfs (list, optional): List of DataFrames to be plotted.
    - df_labels (list, optional): List of labels corresponding to each DataFrame in `dfs`.
                                 If not provided, DataFrames will be labeled "DataFrame 0", "DataFrame 1", etc.
                                 File paths are used as labels for file-based sources.
    """
    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot for 3D.")
        return

    fig_3d = go.Figure()
    color_gen = _color_palette_generator()

    # For axis range calculation
    max_time, max_amp, max_freq = 0.0, 0.0, 0.0
    has_data_for_axes = False

    for df, label in all_data:
        if df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping for 3D plot.")
            continue
        
        required_cols = ['time_start', 'time_stop', 'amp_min', 'amp_max', 'freq_start', 'freq_stop']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing one or more required columns for 3D plot: {required_cols}. Skipping.")
            continue

        current_color = next(color_gen)
        for index, row in df.iterrows():
            fig_3d.add_trace(go.Scatter3d(
                x=[row['time_start'], row['time_stop']],
                y=[row['amp_min'], row['amp_max']],
                z=[row['freq_start'], row['freq_stop']],
                mode='lines+markers',
                line=dict(width=2, color=current_color),
                marker=dict(size=3, color=current_color),
                name=str(label),
                legendgroup=str(label),
                showlegend=(index == 0)
            ))
        
        # Update max values for axis ranges
        if not df.empty:
            has_data_for_axes = True
            max_time = max(max_time, df['time_stop'].max())
            max_amp = max(max_amp, df['amp_max'].max())
            max_freq = max(max_freq, df['freq_stop'].max()) # Assuming freq_stop is relevant for Z axis max

    scene_dict: Dict[str, Any] = dict(
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        zaxis_title='Frequency (Hz)',
        camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=0.5))
    )
    if has_data_for_axes:
        scene_dict['xaxis'] = dict(range=[0, max_time * 1.05]) # 5% padding
        scene_dict['yaxis'] = dict(range=[0, max_amp * 1.05])
        scene_dict['zaxis'] = dict(range=[0, max_freq * 1.05])


    fig_3d.update_layout(
        title_text='Time vs Amplitude vs Frequency (3D)',
        scene=scene_dict,
        template='plotly_white',
        showlegend=True,
        width=900,
        height=700,
        legend_title_text='Sources'
    )
    fig_3d.show()


def plot_scatter(
    files: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
    df_labels: Optional[List[str]] = None,
    mode: str = 'markers+lines'
) -> None:
    """
    Plots data from multiple sources (files and/or DataFrames) on a 2D scatter plot.
    Each DataFrame/file contributes one trace to the scatter plot.

    Parameters:
    - files (list, optional): List of file paths (TSV) to be read and plotted.
                               Expected columns: 'Frequency (Hz)', 'Amplitude'.
    - dfs (list, optional): List of DataFrames to be plotted.
                            Expected columns: 'Frequency (Hz)', 'Amplitude'.
    - df_labels (list, optional): List of labels corresponding to each DataFrame in `dfs`.
                                 If not provided, "DataFrame 0", "DataFrame 1", etc.
                                 File paths are used as labels for file-based sources.
    - mode (str, optional): Plotly mode ('lines', 'markers', 'lines+markers'). Default 'markers+lines'.
    """
    all_data = _load_and_prepare_data(files, dfs, df_labels)
    if not all_data:
        print("No data to plot for scatter.")
        return

    scatter_fig = go.Figure()
    color_gen = _color_palette_generator()

    max_x_val, max_y_val = 0.0, 0.0
    has_data_for_axes = False

    for df, label in all_data:
        if df.empty:
            print(f"Warning: DataFrame for '{label}' is empty. Skipping for scatter plot.")
            continue

        required_cols = ['Frequency (Hz)', 'Amplitude']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: DataFrame for '{label}' is missing one or more required columns for scatter: {required_cols}. Skipping.")
            continue

        current_color = next(color_gen)
        scatter_fig.add_trace(go.Scatter(
            x=df['Frequency (Hz)'],
            y=df['Amplitude'],
            mode=mode,
            name=str(label),
            line=dict(color=current_color), # For lines mode
            marker=dict(color=current_color) # For markers mode
        ))
        
        if not df.empty:
            has_data_for_axes = True
            max_x_val = max(max_x_val, df['Frequency (Hz)'].max())
            max_y_val = max(max_y_val, df['Amplitude'].max())

    scatter_fig.update_layout(
        title_text="Scatter Plot of Frequency vs. Amplitude",
        xaxis_title_text="Frequency (Hz)",
        yaxis_title_text="Amplitude",
        template='plotly_white',
        width=900,
        height=700,
        legend_title_text='Sources'
    )
    if has_data_for_axes:
        scatter_fig.update_xaxes(range=[0, max_x_val * 1.10]) # 10% padding
        scatter_fig.update_yaxes(range=[0, max_y_val * 1.10])

    scatter_fig.show()