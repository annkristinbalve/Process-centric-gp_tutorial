import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout
from means import *
from kernels import *

mean_dropdown = widgets.Dropdown(options=list(mean_dict.keys()),
                                 value='Zero', 
                                 description='Mean:',
                                 disabled=False,
                                 layout=widgets.Layout(width='auto'))
kernel_dropdown = widgets.Dropdown(options=list(kernel_dict.keys()), 
                                   value='RBF', 
                                   description='Kernel:', 
                                   disabled=False,
                                   layout=widgets.Layout(width='auto'))


def create_mean_dropdown():
    return   widgets.Dropdown(options=list(mean_dict.keys()),
                                 value='Zero', 
                                 description='Mean:',
                                 disabled=False,
                                 layout=widgets.Layout(width='auto'))
def create_cov_dropdown():
    return widgets.Dropdown(options=list(kernel_dict.keys()), 
                                   value='RBF', 
                                   description='Kernel:', 
                                   disabled=False,
                                   layout=widgets.Layout(width='auto'))


batch_size_dropdown = widgets.Dropdown(options=[1, 2, 10],
                                       value=1,
                                       disabled=False,
                                       description='Batch Size:',
                                       layout=widgets.Layout(width='auto'))


## marginalisation plot
range_slider = widgets.FloatRangeSlider(value=[-10, 10],
                                    min=-10,
                                    max=10,
                                    step=1,
                                    layout=Layout(width=f'{400}px', margin='0px 20px 20px 20px'),
                                    description='X Range:',
                                    continuous_update=False,
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='.1f',
                                )


# Creating sliders for hyperparameters
varSigma_slider = widgets.FloatSlider(value=1, min=0.01, max=1, step=0.1, description='varSigma:', layout=widgets.Layout(width='280',height='200'))
lengthscale_slider = widgets.FloatSlider(value=1, min=0.01, max=2, step=0.2, description='lengthscale:', layout=widgets.Layout(width=varSigma_slider.layout.width))
noise_slider = widgets.FloatSlider(value=0.001, min=0.001, max=0.2, step=0.02, description='noise:', layout=widgets.Layout(width=varSigma_slider.layout.width))
period_slider = widgets.FloatSlider(value=np.pi, min=0.001, max=np.pi, step=np.pi/10, description='period:', layout=widgets.Layout(width=varSigma_slider.layout.width))

def create_status_label():
    return widgets.Label(value="", layout = Layout(margin='0 0 0 20px'))


def toggle_sliders(change):
    if change['new'] == 'Periodic':
        period_slider.disabled = False
        lengthscale_slider.disabled = False
    elif change['new'] == 'White':
       lengthscale_slider.disabled = True
       period_slider.disabled = True
    else:
        period_slider.disabled = True