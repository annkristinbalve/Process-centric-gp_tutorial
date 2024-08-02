from ipywidgets import widgets, VBox, HBox, Layout
import plotly.graph_objects as go

### marginalisation plot 
common_layout_cov = dict(
        margin=dict(l=30, r=30, b=30, t=30),
        yaxis=dict(scaleanchor="x", scaleratio=1, showticklabels=True, ticks='outside'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=300,
        height=300
    )


common_layout_mean = dict(
    margin=dict(l=30, r=10, b=40, t=40),
    yaxis=dict(showticklabels=False, showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    width=300,
    height=100
)

layout_data_and_gp = lambda x_range: go.Layout(
        title='Data and GP Fit', 
        height=500,
        width=2*500,
        xaxis=dict(title='X', range=x_range),  
        yaxis=dict(title='y'),
        xaxis_showgrid=False, 
        yaxis_showgrid=False,
        paper_bgcolor='rgba(0,0,0,0)',  
        plot_bgcolor='rgba(0,0,0,0)'    
    )

layout_colorbar = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        yaxis=dict(showticklabels=False, showgrid=False),
        xaxis=dict(showticklabels=False, showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=50,
        height=250
    )

layout_data = dict(
    xaxis=dict(range=[-10, 10], showticklabels=True, showgrid = False),
    margin=dict(l=120, r=20, b=20, t=20),
    yaxis=dict(showticklabels=False, showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=True,
    width=360,
    height=60,
)