import plotly.graph_objects as go

def plot_interactive_heatmap(data):
    fig = go.Figure(data=go.Heatmap(z=data))
    fig.update_layout(title="Interactive Heatmap")
    fig.write_html("interactive_heatmap.html")
