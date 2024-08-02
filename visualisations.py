import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import base64
from io import BytesIO
import numpy as np
from matplotlib import cm
from scipy.stats import multivariate_normal, norm
import io
import plotly.graph_objects as go
import math

from kernels import rbf_kernel, lin_kernel, periodic_kernel, white_kernel
from means import zero_mean, sine_mean,lin_mean, step_mean ## importing mean functions 

from gp_functions import gp_inference, GP_marginal, gp_inference_iterative, draw_samples
import plotly.subplots as sp
from plotly.subplots import make_subplots
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal, norm
import math
import ipywidgets as widgets
from IPython.display import display

from ipywidgets import Label, Layout
import plotly.graph_objects as go


### marginalisation plot
import numpy as np
import plotly.graph_objects as go
from ipywidgets import widgets, VBox, HBox, Layout, interactive
from kernels import *
from means import *
from gp_functions import *
from data import *
from IPython.display import display
from widgets_helper import * #range_slider, kernel_dropdown, mean_dropdown
from layout_helper import *
from visualisations import *


#### MULTIVARIATE GAUSSIAN PLOTTING CODE #######

def is_positive_semi_definite_2x2(matrix):
    if matrix.shape != (2, 2):
        raise ValueError("The matrix is not 2x2")

    if matrix[0, 1] != matrix[1, 0]:
        raise ValueError("The matrix is not symmetric.")

    a = matrix[0, 0]
    b = matrix[0, 1]
    d = matrix[1, 1]

    if a < 0:
        return False
    if a * d - b * b < 0:
        return False

    return True

def conditional_distribution(x, y_value, mean, cov):
    mean_x_given_y = mean[0] + cov[0, 1] / cov[1, 1] * (y_value - mean[1])
    var_x_given_y = cov[0, 0] - cov[0, 1]**2 / cov[1, 1]
    return norm.pdf(x, mean_x_given_y, np.sqrt(var_x_given_y))
    
def plot_multivariate_Gaussian_2d(mean, cov, y_value=None):    
    x = np.linspace(-5,5,50)
    y = np.linspace(-5,5,50)
    
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    rv = multivariate_normal(mean, cov, allow_singular=True)
    Z = rv.pdf(pos)
    
    # Marginal distributions
    marginal_x = norm(loc=mean[0], scale=np.sqrt(cov[0, 0]))
    marginal_y = norm(loc=mean[1], scale=np.sqrt(cov[1, 1]))
    
    x_kde = np.linspace(-5,5,50)
    y_kde = np.linspace(-5,5,50)
    
    marginal_x_vals = marginal_x.pdf(x_kde)
    marginal_y_vals = marginal_y.pdf(y_kde)
    
    fig = make_subplots(rows=2, cols=2, 
                        shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.05, horizontal_spacing=0.05,
                        row_heights=[0.2, 0.8], column_widths=[0.8, 0.2],
                        specs=[[{"type": "xy"}, {"type": "xy"}], 
                               [{"type": "xy"}, {"type": "xy"}]])
    
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z, colorscale='RdYlBu', contours_coloring='fill', line_width=1, opacity=0.7,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(x=marginal_y_vals, y=y_kde, mode='lines', line=dict(color='green', width=2), name='Marginal Y'), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_kde, y=marginal_x_vals, mode='lines', line=dict(color='blue', width=2), name='Marginal X'), row=1, col=1)

    if y_value is not None:
        conditional_x = np.linspace(mean[0] - 3*np.sqrt(cov[0, 0]), mean[0] + 3*np.sqrt(cov[0, 0]), 100)
        conditional_z = conditional_distribution(conditional_x, y_value, mean, cov) + y_value
        fig.add_trace(go.Scatter(x=conditional_x, y=conditional_z, mode='lines', line=dict(color='purple'), name=f'Conditional X|Y={y_value}'), row=2, col=1)

    fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode='markers', name='Mean', marker=dict(color='blue', size=12)), row=2, col=1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Plot the eigenvectors scaled by the square root of eigenvalues
    for i in range(2):
        fig.add_trace(go.Scatter(
            x=[mean[0], mean[0] +  2 * np.sqrt(eigvals[i]) * eigvecs[0, i]], 
            y=[mean[1], mean[1] + 2 * np.sqrt(eigvals[i]) * eigvecs[1, i]],
            mode='lines+markers', name=f'Eigenvector {i+1}', line=dict(color=['red', 'blue'][i])
        ), row=2, col=1)
        
    fig.update_layout(
        title_text='Bivariate Gaussian Distribution with Marginals',
        title_y=0.95,
        autosize=False,
        width=700,
        height=700,
        legend=dict(x=0.7, y=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        uirevision='true'  )
    fig.update_xaxes(title_text="X", row=2, col=1, showgrid=False)
    fig.update_yaxes(title_text="Y", row=2, col=1, showgrid=False)
    fig.show()
    

def plot_multivariate_Gaussian_3d(mean, cov, y_value = None):
    n_samples = 100
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    
    x = np.linspace(mean[0] - 6*np.sqrt(cov[0, 0]), mean[0] + 6*np.sqrt(cov[0, 0]), 100)
    y = np.linspace(mean[1] - 6*np.sqrt(cov[1, 1]), mean[1] + 6*np.sqrt(cov[1, 1]), 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    try:
        rv = multivariate_normal(mean, cov, allow_singular=True)
    except np.linalg.LinAlgError:
        raise ValueError("The covariance matrix should be semi-positive definite") 
    Z = rv.pdf(pos)
    
    # Marginal distributions
    marginal_x = norm(loc=mean[0], scale=np.sqrt(cov[0, 0]))
    marginal_y = norm(loc=mean[1], scale=np.sqrt(cov[1, 1]))
    
    # Create the 3D plot
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='RdYlBu', opacity=0.7, cmin=0, cmax=Z.max()))

        # Plot KDE for smoother marginal distributions
    x_kde = np.linspace(mean[0] - 6*np.sqrt(cov[0, 0]), mean[0] + 6*np.sqrt(cov[0, 0]), 100)
    y_kde = np.linspace(mean[1] - 6*np.sqrt(cov[1, 1]), mean[1] + 6*np.sqrt(cov[1, 1]), 100)
    
    marginal_x_vals = marginal_x.pdf(x_kde)
    marginal_y_vals = marginal_y.pdf(y_kde)
    
    # Specify a y_value for the conditional distribution
    if y_value != None:
        #y_value = 1.5  # You can choose any value you like
        conditional_x = np.linspace(-6, 6, 100)
        conditional_z = conditional_distribution(conditional_x, y_value, mean, cov) #+ y_value
        fig.add_trace(go.Scatter3d(
            x=conditional_x,
            y=np.full_like(conditional_x,y_value),
            z=0.4*conditional_z-0.4,
            mode='lines',
            line=dict(color='purple',width=4),
            name=f"Conditional (X | Y = {y_value})" ))
 
# Plot marginal distributions
    fig.add_trace(go.Scatter3d(
        x=x_kde, 
        y=np.full_like(x_kde, mean[1] - 6*np.sqrt(cov[1, 1])), 
        z=marginal_x_vals, 
        mode='lines',
        line=dict(color='red', width=4),
        name='Marginal X'))
    fig.add_trace(go.Scatter3d(
        x=np.full_like(y_kde, mean[0] + 6*np.sqrt(cov[0, 0])), 
        y=y_kde, 
        z=marginal_y_vals, 
        mode='lines',
        line=dict(color='blue', width=4),
        name='Marginal Y' ))
    
    # Scatter plot of the samples
    fig.add_trace(go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=np.full_like(samples[:, 0], -0.4), name="Samples", mode='markers', marker=dict(size=3, color='black')))
    
    fig.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Probability Density'
        ),
        title='Bivariate Gaussian Distribution with Marginals',
        autosize=True,
        width=800,
        height=800,
        legend=dict(
            x=0.02,
            y=0.98,
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='#000'
            ),
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(0, 0, 0, 0)',
            borderwidth=1  ) )
    fig.show()

##### VISUALISATIONS OF MEAN AND COVARIANCE SAMPLES ###########
def plot_prior_samples(prior_functions, prior_names, params, kernel_f=rbf_kernel, mean_f=zero_mean, n=10, prior_type="mean", layout=(1, 4)):
    x = np.linspace(-3, 3, 100).reshape(-1, 1)
    plt.rcParams.update({'axes.titlesize': 24, 'axes.labelsize': 22})  
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(5 * layout[1], 5 * layout[0]))
    for i, (prior_f, prior_name) in enumerate(zip(prior_functions, prior_names)):
        if prior_type == "mean":
            m_f, k_f = prior_f, kernel_f
            title = f"m(x) = {prior_name}"
        elif prior_type == "covariance":
            m_f, k_f = mean_f, prior_f
            title = f"{prior_name} Kernel"
        else:
            raise ValueError("Please specify 'mean' or 'covariance' depending on whether you provide a list of mean or covariances")
        K = k_f(x, x, params) + params["noise"]**2 * np.eye(len(x))
        m = m_f(x)
        f = draw_samples(m, K, n)  # Draw n samples
        ax = axes[i] if layout != (1, 1) else axes
        ax.plot(x, f, alpha=0.7)
        ax.plot(x, m, 'k', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_ylim([-4, 4])
    plt.suptitle(f"Samples drawn from different prior {prior_type} functions", fontsize=26, y=1.025)
    plt.tight_layout()
    plt.show()


##### KERNEL AND SAMPLES ###########

def plot_samples_and_kernels(mu_f, K_f, params, n, k_name=""):
    x=np.linspace(-6, 6, 100).reshape(-1,1)
    K = K_f(x, x, params) + params["noise"]**2 * np.eye(len(x))
    m = mu_f(x)
    f = draw_samples(m, K, n)
    
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Samples from GP", f"{k_name} Kernel Matrix"), 
        column_widths=[0.6, 0.4], 
        horizontal_spacing=0.15 )
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot Samples
    for i in range(f.shape[1]):
        fig.add_trace(go.Scatter(
            x=x.flatten(), y=f[:, i], 
            showlegend=False, 
            line=dict(width=1.25, color=colors[i % len(colors)]), 
            mode='lines',
            name=f'Sample {i+1}'
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x.flatten(), 
        y=m, 
        mode='lines', 
        line=dict(color='black', width=3),
        name='Mean Function',
        showlegend=True
    ), row=1, col=1)
    
    # Plot Covariance Matrix
    fig.add_trace(go.Heatmap(
        z=K, 
        x=np.linspace(-6, 6, K.shape[1]), 
        y=np.linspace(-6, 6, K.shape[0]), 
        colorscale='RdYlBu',
        colorbar_x=1,
        showlegend=False,
        colorbar_thickness=20,
        colorbar_len = 1
    ), row=1, col=2)

    fig.update_xaxes(title_text = "x",scaleanchor="y", scaleratio=1, row=1, col = 2, tickvals=[-6, 0, 6])
    fig.update_yaxes(title_text = "x",scaleanchor="y", scaleratio=1, row=1, col=2, range = [-7,7], tickvals=[-6, 0,6])
    fig.update_layout(
        height=400,
        width=1000,
        plot_bgcolor='rgba(0,0,0,0)',
        autosize = False,
         legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01 ),
        margin = {'t':45} )
    fig.update_xaxes(showgrid=False, showline=True, mirror=True, title_text="x",  tickvals=[-6, 0,6],row=1, col=1)
    fig.update_yaxes(showgrid=False, showline=True, mirror=True, title_text="f(x)",  tickvals=[-6, 0,6],row=1, col=1, range=[-7, 7])
    fig.show()



##### CONDITIONING OPERATION ###########

# Function to create prior traces
def create_prior_traces(fig, Xtest, mu_prior, k_prior, X, y, ytest, ci_color, true_color):
    fig.add_trace(go.Scatter(visible=True, x=Xtest.flatten(), y=mu_prior, mode='lines', name='Prior Mean',
                             line=dict(color="blue", width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(visible=True, x=np.concatenate([Xtest.flatten(), Xtest.flatten()[::-1]]),
                             y=np.concatenate([mu_prior - 1.96 * k_prior, (mu_prior + 1.96 * k_prior)[::-1]]),
                             fill='toself', fillcolor=ci_color, opacity=0.2, name='95% CI (Prior Variance)', showlegend=True))
    fig.add_trace(go.Scatter(visible=True, x=X.flatten(), y=y, mode='markers', marker=dict(color='red'), name='Observations'))
    fig.add_trace(go.Scatter(visible=True, x=Xtest.flatten(), y=ytest, mode='lines', line=dict(color=true_color, width=2), name='Ground truth'))

# Function to create incremental traces
def create_incremental_traces(fig, mu, k, Xtest, ytest, X, y, Xi, yi, ci_color, true_color):
    fig.add_trace(go.Scatter(visible=False, x=Xtest.flatten(), y=mu, mode='lines', name='Predictive Mean',
                             line=dict(color="blue", width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(visible=False, x=np.concatenate([Xtest.flatten(), Xtest.flatten()[::-1]]),
                             y=np.concatenate([mu - 1.96 * k, (mu + 1.96 * k)[::-1]]),
                             fill='toself', fillcolor=ci_color, opacity=0.2, name='95% CI (Predictive Variance)', showlegend=True))
    fig.add_trace(go.Scatter(visible=False, x=X.flatten(), y=y, mode='markers', marker=dict(color='red'), name='Observations', showlegend=False))
    fig.add_trace(go.Scatter(visible=False, x=Xi.flatten(), y=yi, mode='markers', marker=dict(color='red', size=10), name='Added data point'))
    fig.add_trace(go.Scatter(visible=False, x=Xtest.flatten(), y=ytest, mode='lines', line=dict(color=true_color, width=2), name='Ground truth', showlegend=False))

def conditional_gps_visualisation(batch_size, N, X, y, Xtest, ytest, status_label, mean_f_inc, cov_f_inc, params):
    status_label.value = "UPDATING THE PLOT..."
    num_batches = (N + batch_size - 1) // batch_size
    log_Z_inc = 0
    mu_prior = mean_f_inc(Xtest)
    cov_prior = cov_f_inc(Xtest, Xtest, params)
    k_prior = np.sqrt(np.diag(cov_prior))

    ymin = min(mu_prior - 1.96 * k_prior)
    ymax = max(mu_prior + 1.96 * k_prior)

    fig = go.Figure()

    steps = []
    traces_per_step = 5
    initial_traces = 4

    create_prior_traces(fig, Xtest, mu_prior, k_prior, X, y, ytest, ci_color='#1f77b4', true_color='green')
    step_title = "Prior"
    visible = [True] * initial_traces + [False] * (traces_per_step * num_batches)
    steps.append(dict(method="update", args=[{"visible": visible}, {"title": step_title}]))

    for i in range(0, N, batch_size):
        Xi = X[i:i + batch_size].reshape(-1, 1)
        yi = y[i:i + batch_size]
        mean_f_inc, cov_f_inc, log_Z = gp_inference(mean_f_inc, cov_f_inc, params, Xi, yi, False)
        log_Z_inc += log_Z

        mu_pred_inc = mean_f_inc(Xtest)
        cov_pred_inc = cov_f_inc(Xtest, Xtest)
        k_pred_inc = np.sqrt(np.diag(cov_pred_inc))
        create_incremental_traces(fig, mu_pred_inc, k_pred_inc, Xtest, ytest, X, y, Xi, yi, ci_color='#1f77b4', true_color='green')
        
        step_title = f"Step {i // batch_size + 1} (log_Z: {log_Z_inc:.2f})"
        visible = [False] * initial_traces + [False] * (traces_per_step * (i // batch_size)) + [True] * traces_per_step + [False] * (traces_per_step * (num_batches - (i // batch_size + 1)))
        steps.append(dict(method="update", args=[{"visible": visible}, {"title": step_title}]))

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Step: "},
        pad={"t": 50},
        steps=steps )]
    
    fig.update_layout(
        sliders=sliders,
        title=dict(
        text="Prior",
        x=0.5,  # Center title
        y=0.95, # Adjust title position to avoid overlap with legend
        xanchor='center',
        yanchor='top'  ),
        xaxis_title="Input X",
        yaxis_title="Output y",
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend layout
            yanchor="top",
            y=1.15,
            xanchor="center",
            x=0.5 ),
        autosize = False,
        height = 500,
        width=800,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_range = [ymin,ymax])
    fig.show()
    status_label.value = "Plot updated."

##### MARGINALISATION OPERATION ###########

def gpr_components_visualisation():
    fig = make_subplots(
        rows=2, cols=4, 
        column_widths=[0.45, 0.1, 0.1, 0.1],
        row_heights=[0.65, 0.2],  
        specs=[[{"rowspan": 2}, {}, {}, {}], 
               [None, {}, {}, {}]], 
        horizontal_spacing=0.05,
        subplot_titles=("", "Prior Cov", "Data Cov", "Post Cov",
                        "Prior Mean", "Data Mean", "Post Mean", "Post Mean"))
    
    # Add GP Data plot 
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(color='red'), name='Data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color="blue", width=3, dash='dash'), name='GP Mean'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], fill='toself', fillcolor='#1f77b4', line=dict(color='rgba(255,255,255,0)'), name='GP 95% CI', opacity=0.2), row=1, col=1)
    
    # Add Heatmaps for Covariances
    fig.add_trace(go.Heatmap(z=[], colorscale='RdYlBu',showscale = False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=[], colorscale='RdYlBu', showscale=False), row=1, col=3)
    fig.add_trace(go.Heatmap(z=[], colorscale='RdYlBu', showscale=False), row=1, col=4)
    
    # Add Heatmaps for Means
    fig.add_trace(go.Heatmap(z=[], colorscale='RdYlBu', showscale=False), row=2, col=2)
    fig.add_trace(go.Heatmap(z=[], colorscale='RdYlBu', showscale=False), row=2, col=3)
    fig.add_trace(go.Heatmap(z=[], colorscale='RdYlBu', showscale=False), row=2, col=4)
    fig.update_layout(
        title={
            'text': "Gaussian Process Components",
            'y':0.98,  # Positioned near the top of the plot
            'x':0.5,   # Centered horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}  # You can adjust the font size as needed
        },
        # autosize = True,
        height=450,  # Adjusted height 400
        width=1100,  # Adjusted width
        legend=dict(
            x=0.25, 
            y=1.2,
            orientation="h",
            xanchor="center",
            yanchor="top"),
        xaxis_rangeslider=dict(visible=True, thickness = 0.1, bgcolor = '#fff' ),  # Enable range slider
        xaxis_showgrid = False,
        yaxis_showgrid = False,
    plot_bgcolor='rgba(0,0,0,0)',
      #  margin=dict(l=80, r=80, t=80, b=80)
    )
    
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)

    fig.update_xaxes(range=[-8, 8], row=1, col=1)  # Restrict x-axis range for overall data
    output = widgets.Output()
    fig.update_xaxes(matches='x', row=1, col=1)  # Overall Data
    fig.update_xaxes(matches='x', row=1, col=2)  # Prior Cov
    fig.update_xaxes(matches='x', row=1, col=3)  # Data Cov
    fig.update_xaxes(matches='x', row=1, col=4)  # Posterior Cov
    
    fig.update_yaxes(matches='x', row=1, col=2)  # Prior Cov
    fig.update_yaxes(matches='x', row=1, col=3)  # Data Cov
    fig.update_yaxes(matches='x', row=1, col=4)  # Posterior Cov
    
    fig.update_xaxes(matches='x', row=2, col=2)  # Prior Mean
    fig.update_xaxes(matches='x', row=2, col=3)  # Data Mean
    fig.update_xaxes(matches='x', row=2, col=4)  # Posterior Mean
    

    for col in [2, 3, 4]:
        fig.update_yaxes(range=[-1, 1],showticklabels=False, row=2, col=col)
        
    for col in [2, 3, 4]:
        fig.update_yaxes(scaleratio=0.1, row=2, col=col)
        fig.update_yaxes( scaleratio=1, row=1, col=col) #scaleanchor='y',
    
    fig.add_annotation(
        x=0.88, y=0.85, 
        text="=",
        showarrow=False,
        font=dict(size=40),
        xref="paper", yref="paper")
    
    fig.add_annotation(
        x=0.71, y=0.85, 
        text="-",
        showarrow=False,
        font=dict(size=44),
        xref="paper", yref="paper")
    
    fig.add_annotation(
        x=0.88, y=0,  
        text="=",
        showarrow=False,
        font=dict(size=40),
        xref="paper", yref="paper")
    
    fig.add_annotation(
        x=0.72, y=0,  
        text="+",
        showarrow=False,
        font=dict(size=40),
        xref="paper", yref="paper")
    return fig

##### MARGINALISATION  ###########

# def create_figure(X_vals, Y_vals, Z_vals, X_label, Y_label, title, project_contour=True):
#     X_mesh, Y_mesh = np.meshgrid(X_vals, Y_vals)
#     diff = Z_vals.max() - Z_vals.min()
#     contour_offset = Z_vals.min() - diff

#     surface = go.Surface(
#         z=Z_vals.T, 
#         x=X_vals, 
#         y=Y_vals, 
#         colorscale='RdYlBu', 
#         opacity=1,
#         contours=dict(
#             z=dict(
#                 show=project_contour,
#                 start=Z_vals.min(),
#                 end=Z_vals.max(),
#                 size=(Z_vals.max() - Z_vals.min()) / 100,
#                 usecolormap=True,
#                 project=dict(z=project_contour)     
#             )
#         )
#     )

#     data = [surface]

#     layout = go.Layout(
#         title=title,
#         scene=dict(
#             xaxis=dict(title=X_label),
#             yaxis=dict(title=Y_label),
#             zaxis=dict(title='Log Marginal Likelihood', range=[contour_offset, Z_vals.max()]),
#             aspectmode='cube',  # Ensure the plot is square

#         ),
#         margin=dict(l=0, r=0, b=0, t=40)
#     )

#     fig = go.Figure(data=data, layout=layout)
#     return fig



#########   Log Marginal Likelihood   #########


def compute_log_marginal_likelihood(param1_values, param2_values, param1_name, param2_name, fixed_params, X, y, m, k):
    Z = np.zeros((len(param1_values), len(param2_values)))
    for i, val1 in enumerate(param1_values):
        for j, val2 in enumerate(param2_values):
            params = fixed_params.copy()
            params[param1_name] = val1
            params[param2_name] = val2
            param_keys = list(params.keys())
            flat_params = np.array([params[key] for key in param_keys])
            Z[i, j] = neg_log_marginal_likelihood(flat_params,param_keys, X, y, m, k)
    return Z

def create_figure(X_vals, Y_vals, Z_vals, X_label, Y_label, title, project_contour=True):
    X_mesh, Y_mesh = np.meshgrid(X_vals, Y_vals)
    diff = Z_vals.max() - Z_vals.min()
    contour_offset = Z_vals.min() - diff

    surface = go.Surface(
        z=Z_vals.T, 
        x=X_vals, 
        y=Y_vals, 
        colorscale='RdYlBu', 
        opacity=1,
        contours=dict(
            z=dict(
                show=project_contour,
                start=Z_vals.min(),
                end=Z_vals.max(),
                size=(Z_vals.max() - Z_vals.min()) / 100,
                usecolormap=True,
                project=dict(z=project_contour)        ) ) )
    data = [surface]

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=X_label),
            yaxis=dict(title=Y_label),
            zaxis=dict(title='Negative Log Marginal Likelihood', range=[contour_offset, Z_vals.max()]),
            aspectmode='cube',
        ),
        width=700,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig


############ MODEL COMPARISON #######################

def kernel_comparison_gpr(X, y, Xtest, ytest,kernels, kernel_names, mean_function, optimised=True, layout=(1, 4), showlegend = True):
    fig, ax = plt.subplots(*layout, figsize=(layout[1] * 6, layout[0] * 6))
    title_fontsize = 22  # Adjust as needed
    LLs = []
    params_all = []
    ax = ax.flatten()  # Flatten in case of 2D array of axes
    for i, k in enumerate(kernels):
        params = {'varSigma': 1, 'lengthscale': 1, 'noise': 1, 'period': 1}
        if optimised:
            mu_f, k_f, log_Z = gp_inference_optimised(mean_function, k, X, y, params)
        else:
            mu_f, k_f, log_Z = gp_inference(mean_function, k, params, X, y, False)
        title = f'{kernel_names[i]} Kernel'
        LLs.append(log_Z)  # Negative log marginal likelihood
        mu_pred = mu_f(Xtest)
        k2 = k_f(Xtest, Xtest)
        k_pred = np.sqrt(np.diag(k2))
    
        # Plot results for GPR
        ax[i].plot(X, y, 'ro', label="Observations")
        ax[i].plot(Xtest, ytest, 'g-', label="Ground truth")
        ax[i].fill_between(Xtest.flatten(), mu_pred - 2 * k_pred, mu_pred + 2 * k_pred, 
                           color='#1f77b4', alpha=0.2, label='Predictive variance')
        ax[i].plot(Xtest, mu_pred, 'b--', lw=2, label="Predictive Mean")
        ax[i].set_title(title, fontsize = 26)
        ax[i].set_xlabel('X')
        ax[i].set_ylabel('y')
        ax[i].set_ylim([-2.5, 2.5])
        ax[i].set_xticks([-4, 0, 4])
        ax[i].set_yticks([-2, 0, 2])

        params_all.append(params)
    
    for j in range(len(kernels), len(ax)):  # Hide any unused subplots
        fig.delaxes(ax[j])
        
    if showlegend:
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0),  prop={'size': 26})
    if optimised:
        fig.suptitle("Gaussian Process Regression after hyperparameter optimisation ")
    else:
        fig.suptitle("Gaussian Process Regression before hyperparameter optimisation ")
    plt.tight_layout()
    plt.show()
    return LLs, params_all



# def plot_kernel_and_samples(mu_f, K_f, params, x, n, k_name="", mu_name=""):
#     K = K_f(x, x, params) + params["noise"]**2 * np.eye(len(x)) 
#     m = mu_f(x)

#     #L = np.linalg.cholesky(K)

#     f = draw_samples(m, K, n)
    
#     # Generate samples from the Gaussian process
#     #U = np.random.normal(loc=0, scale=1, size=(len(x), n))
#     #f = m[:, None] + np.dot(L, U)  # Broadcasting the mean vector to match the shape of samples
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
#     im = ax1.imshow(K, aspect="auto", extent=[-6, 6, -6, 6], origin='lower')
#     ax1.set_title(f"{k_name} Kernel Matrix")
#     plt.colorbar(im, ax=ax1)
    
#     ax2.plot(x, f)
#     ax2.set_title("Samples from GP")

#     plt.show()
#     plt.tight_layout()



# def plot_posterior(K1, K2, K3, X, y, Xtest, ytest, k_pred, mu_pred, kernel_name = ""):
#     fig, ax = plt.subplots(1, 3, figsize=(24, 6))#6
#     im = ax[0].imshow(K1, aspect="auto", extent=[-6, 6, -6, 6], origin='lower')
#     ax[0].set_title("Prior Covariance Matrix$", fontsize=16)
#     plt.colorbar(im, ax=ax[0])
    
#     im = ax[1].imshow(K2, aspect="auto", extent=[-6, 6, -6, 6], origin='lower')
#     ax[1].set_title("Data Covariance Contribution Matrix", fontsize=16)
#     plt.colorbar(im, ax=ax[1])
    
#     im = ax[2].imshow(K3, aspect="auto", extent=[-6, 6, -6, 6], origin='lower')
#     ax[2].set_title("Posterior Covariance Matrix", fontsize=16)
#     plt.colorbar(im, ax=ax[2])

#     # Save the figure to a BytesIO object
#     buf = BytesIO()
#     fig.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
#     image_base64_cov = base64.b64encode(buf.read()).decode('ascii')
# #
#     # Plot predictive mean and variance
#     fig, ax = plt.subplots(figsize=(24, 6))
#     ax.plot(X, y, 'ro', label="Observations")
#     ax.plot(Xtest, ytest, 'g-', label="Ground truth")
#     ax.fill_between(Xtest.flatten(), mu_pred - 2 * k_pred, mu_pred + 2 * k_pred, color='blue', alpha=0.2, label='Predictive variance')
#     ax.plot(Xtest, mu_pred, 'b--', lw=2, label="Predictive Mean")
#     ax.set_title(f'Gaussian Process Regression {kernel_name}', fontsize=16)
#     ax.set_xlabel('x', fontsize=14)
#     ax.set_ylabel('y', fontsize=14)
#     ax.legend()

#     # Save the plot to a BytesIO object
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
    
#     image_base64_gp = base64.b64encode(buf.read()).decode('ascii')

#     return f"data:image/png;base64,{image_base64_cov}", f"data:image/png;base64,{image_base64_gp}"

# def data_1d(N, noise):
#     x = np.random.randn(N)
#     x = np.sort(x)
#     y = np.sin(3 * x).flatten()
#     y = y + noise * np.random.randn(N)
#     x = x.reshape(-1, 1)
#     return x, y

# def plot_data(X, y):
#     plt.figure(figsize = (24,6))
#     plt.scatter(X, y, color='red')
#     plt.title('Generated Data with Noise')
#     plt.xlabel('X')
#     plt.ylabel('y')
#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
#     plt.close()
#     return 'data:image/png;base64,{}'.format(img_base64)

# def plot_combined_data_and_gp(X, y, Xtest, mu_pred, k_pred):
#     plt.figure()
#     plt.scatter(X, y, color='red', label='Data')
#     plt.plot(Xtest, mu_pred, color='blue', label='GP Mean')
#     plt.fill_between(Xtest.flatten(), mu_pred - 1.96 * k_pred, mu_pred + 1.96 * k_pred, color='blue', alpha=0.2, label='GP 95% CI')
#     plt.title('Data and GP Fit')
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.legend()
#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
#     plt.close()
#     return 'data:image/png;base64,{}'.format(img_base64)



# def plot_data_and_gp(X, y, Xtest, mu_pred, k_pred, x_range):
#     trace_data = go.Scatter(x=X.flatten(), y=y, mode='markers', marker=dict(color='red'), name='Data')
#     trace_mean = go.Scatter(x=Xtest.flatten(), y=mu_pred, mode='lines', line=dict(color='blue', width=3), name='GP Mean')
#     trace_ci = go.Scatter(x=np.concatenate([Xtest.flatten(), Xtest.flatten()[::-1]]),
#                           y=np.concatenate([mu_pred - 1.96 * k_pred, (mu_pred + 1.96 * k_pred)[::-1]]),
#                           fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(255,255,255,0)'), name='GP 95% CI')
   
#     fig = go.Figure(data=[trace_data, trace_mean, trace_ci], layout=layout_data_and_gp)
#     return go.FigureWidget(fig)


# def plot_contribution(m1, m2, m3, title1, title2, title3, common_layout):
#     m1 = np.repeat(m1, 100, 0)
#     m2 = np.repeat(m2, 100, 0)
#     m3 = np.repeat(m3, 100, 0)
    
#     trace_m1 = go.Heatmap(z=m1, colorscale='RdYlBu', showscale=False)
#     trace_m2 = go.Heatmap(z=m2, colorscale='RdYlBu', showscale=False)
#     trace_m3 = go.Heatmap(z=m3, colorscale='RdYlBu', showscale=False)
    
#     layout_m1 = go.Layout(title_text=title1, **common_layout)
#     layout_m2 = go.Layout(title_text=title2, **common_layout)
#     layout_m3 = go.Layout(title_text=title3, **common_layout)
    
#     fig_m1 = go.FigureWidget(data=[trace_m1], layout=layout_m1)
#     fig_m2 = go.FigureWidget(data=[trace_m2], layout=layout_m2)
#     fig_m3 = go.FigureWidget(data=[trace_m3], layout=layout_m3)
#     return fig_m1, fig_m2, fig_m3


# def plot_colorbar():
#     trace = go.Heatmap(z=[[0, 1], [1, 0]], colorscale='RdYlBu', showscale=True, colorbar=dict(thickness=10, len=1.0, y=0.3, yanchor='middle'))
#     fig = go.Figure(data=[trace], layout=layout_colorbar)
#     return go.FigureWidget(fig)

# def plot_gp_figures(X, y, Xtest, ytest, kernel_f, mean_f, params, x_range):
#     mu_f, k_f, log_Z, cov_contribution_f, mean_contribution_f = gp_inference(mean_f, kernel_f, params, X, y, True)
#     mu_pred, k2 = GP_marginal(mu_f, k_f, Xtest)
#     k_pred = np.sqrt(np.diag(k2))

#     fig_data_gp = plot_data_and_gp(X, y, Xtest, mu_pred, k_pred, x_range)

#     K1 = kernel_f(Xtest, Xtest, params)  # marginalised prior covariance on test data points
#     K2 = cov_contribution_f(Xtest, Xtest)  # marginalised data
#     K3 = k_f(Xtest, Xtest, params)  # marginalised covariance 
    
#     fig_k1, fig_k2, fig_k3 = plot_contribution(K1, K2, K3, "Prior Covariance Matrix", "Data Contribution", "Posterior Covariance Matrix",common_layout_cov)

#     mu1 = mean_f(Xtest).reshape(1, -1)  # marginalised prior mean on test data points
#     mu2 = mean_contribution_f(Xtest).reshape(1, -1)  # marginalised data mean contribution
#     mu3 = mu_f(Xtest).reshape(1, -1)  # marginalised posterior mean
   
#     fig_m1, fig_m2, fig_m3 = plot_contribution( mu1, mu2, mu3, "Prior Mean", "Data Contribution Mean", "Posterior Mean",common_layout_mean)
#     fig_colorbar = plot_colorbar()

#     return fig_data_gp, fig_k1, fig_k2, fig_k3, fig_m1, fig_m2, fig_m3, fig_colorbar


# def plot_overall_data(X, y):
#     trace_data = go.Scatter(x=X.flatten(), y=y, mode='markers', marker=dict(color='red'), name='Data')
#     fig = go.Figure(data=[trace_data], layout=layout_data)
#     return go.FigureWidget(fig)


