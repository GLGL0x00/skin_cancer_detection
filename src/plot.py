import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np 
import matplotlib as plt
import cv2
import h5py
LABELS = ["Benign", "Malignant"]
COLORS = ['#66c2a5', '#fc8d62']
LBL_MAP_I2S = {i:l for i,l in enumerate(LABELS)}
nb_palette = sns.color_palette(palette='tab20')

def plot_target_distribution(df: pd.DataFrame, log_y: bool = True) -> None:
    """Plot the distribution of the target variable.

    Args:
        df (pd.DataFrame): 
            The input dataframe containing the target column.
        log_y (bool, optional):
            Whether to log the y-axis (helpful for visualizing large class imbalance)

    Returns:
        None; 
            This function doesn't return anything, it displays a plot.
    """
    # Count the occurrences of each target value
    target_counts = df['target'].value_counts().sort_index()
    
    # Calculate percentages
    total = len(df)
    percentages = [f"{count/total:.3%}" for count in target_counts]
    
    # Create the bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=LABELS,  # Assume we have access to this
            y=target_counts,
            text=percentages,
            textposition='auto',
            marker_color=COLORS  # Assume we have access to this
        )
    ])
    
    # Customize the layout
    fig.update_layout(
        title='<b>DISTRIBUTION OF BENIGN VS MALIGNANT LESIONS',
        xaxis_title='<b>Lesion Classification</b>', yaxis_title=f'<b>Count {"<sub>"+"(Log Scale)"+"</sub>" if log_y else ""}</b>',
        template='plotly_white', height=600, width=1200,
    )
    
    if log_y:
        fig.update_layout(yaxis=dict(type='log'))
    
    # Add annotation for total count
    fig.add_annotation(
        text=f"<b>TOTAL SAMPLES:  {total:,}</b>",
        xref="paper", yref="paper",
        x=0.98, y=1.05,
        showarrow=False,
        font=dict(size=12)
    )
    
    # Show the plot
    fig.show()

def plot_categorical_feature_distribution(
    df: pd.DataFrame, 
    feature_col: str,
    target_col: str = "target",
    target_as_str: bool = True,
    log_y: bool = False, 
    color_sequence: list[str] | None = None,
    template_theme: str = "plotly_white",
    group_by_target: bool = True,
    stack_bars: bool = False
) -> None:
    """Plot the distribution of a feature, optionally grouped by the target variable.

    This function creates a histogram of the feature distribution,
    with options for log scale, custom color schemes, and grouping by target.

    Args:
        df (pd.DataFrame): 
            The input dataframe containing feature and target columns.
        feature_col (str): 
            Name of the feature column to plot.
        target_col (str, optional): 
            Name of the target column.
        target_as_str (bool, optional): 
            Whether to convert target labels to strings.
        log_y (bool, optional): 
            Whether to use log scale for y-axis.
        color_sequence (list[str], optional): 
            Custom color sequence for the bars.
        template_theme (str, optional): 
            Plotly template theme for visual styling.
            Available options include: 
                'ggplot2', 'seaborn', 'simple_white', 'plotly',
                'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff',
                'gridon', 'none'.
        group_by_target (bool, optional): 
            Whether to group bars by target.
        stack_bars (bool, optional): 
            Whether to stack bars when grouped.

    Returns:
        None: This function displays a plot and doesn't return anything.
    """
    # Prevent accidental edits to the original dataframe
    _df = df.copy().sort_values(by=[feature_col, target_col]).reset_index(drop=True)
        
    if target_as_str and group_by_target:
        _df[target_col] = _df[target_col].map(LBL_MAP_I2S)
        
    # Set default color sequence if not provided
    if not color_sequence:
        color_sequence = list(nb_palette.as_hex())
    
    # Prepare the histogram data
    if group_by_target:
        fig = px.histogram(
            _df, x=feature_col, color=target_col, 
            color_discrete_sequence=COLORS,  # Use target colors for grouping
            log_y=log_y, height=500, width=1200, template=template_theme,
            title=f'<b>DISTRIBUTION OF {feature_col.replace("_", " ").upper()} BY TARGET',
            barmode='group' if not stack_bars else 'stack'
        )
        
        # Add border to bars using the target colors
        for i, trace in enumerate(fig.data):
            trace.marker.line.color = COLORS[i]
            trace.marker.line.width = 1.5
    else:
        fig = px.histogram(
            _df, x=feature_col, color=feature_col, 
            color_discrete_sequence=color_sequence,
            log_y=log_y, height=500, width=1200, template=template_theme,
            title=f'<b>DISTRIBUTION OF {feature_col.replace("_", " ").upper()}',
        )
    
    # Customize the layout
    fig.update_layout(
        bargap=0.1,  # Add space between bars
        xaxis_title=f'<b>{feature_col.replace("_", " ").title()}</b>', 
        yaxis_title=f'<b>Count {"<sub>(Log Scale)</sub>" if log_y else ""}</b>',
        showlegend=group_by_target  # Show legend only when grouped by target
    )
    
    # Apply log scale to y-axis if requested
    if log_y:
        fig.update_layout(yaxis_type='log')
    
    # Display the plot
    fig.show()

def show_bar_plot(df, figsize=(12,6)):
    # Probability of melanoma with respect to age and sex
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Helper function to show values on bars
    def show_values_on_bars(axs, h_v="v", space=0.4, v_space=0.02):
        def _show_on_single_plot(ax):
            if h_v == "v":  # Vertical bars
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height() + v_space
                    value = float(p.get_height())
                    ax.text(_x, _y, f'{value:.1f}', ha="center")
            elif h_v == "h":  # Horizontal bars
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + space
                    _y = p.get_y() + p.get_height() + v_space
                    value = int(p.get_width())
                    ax.text(_x, _y, value, ha="left")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    # Set up the figure size and title
    plt.figure(figsize=figsize)
    sns.set()
    plt.title('Probability')

    # Convert values to percentage
    prob = df * 100

    # Define the color palette and rank the values for coloring
    pal = sns.color_palette(palette='Blues_r', n_colors=len(prob))
    rank = prob.values.argsort().argsort()

    # Correct the sns.barplot call by using keyword arguments for x and y
    br = sns.barplot(x=prob.index, y=prob.values, palette=np.array(pal[::-1])[rank])

    # Add values on top of the bars
    # show_values_on_bars(br, "v", 0.50)

    # Display the plot
    plt.show()
