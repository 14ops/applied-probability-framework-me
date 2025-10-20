"""Module for advanced data visualizations using Plotly."""

import plotly.graph_objects as go


def plot_interactive_heatmap(data):
    """Generates and saves an interactive heatmap from the given data.

    Args:
        data (list of list): A 2D list or numpy array representing the heatmap data.
    """
    fig = go.Figure(data=go.Heatmap(z=data))
    fig.update_layout(title="Interactive Heatmap")
    fig.write_html("interactive_heatmap.html")

