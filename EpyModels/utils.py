import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

def load_model(path):
    with open(path, 'rb') as p:
        return pickle.load(p)

def save_model(model, path):
    with open(path + '.pkl', 'wb') as p:
        pickle.dump(model, p)

def plot_states(model, type='line',run=1):
    colormap = {'Susceptible': 'blue', 'Exposed' : 'orange', 'Infected': 'red', 'Recovered': 'green', 'Dead': 'black', 'Not Susceptible': 'purple'}
    fig = go.Figure()
    for col in model.state_timeline.columns:
        
        if type == 'line':
            fig.add_trace(go.Scatter(x=model.state_timeline.query('Run == ' + str(run)).index.get_level_values(1), y=model.state_timeline[col], mode='lines', name=col, line=dict(color=colormap[col],width=2)))
            
        elif type == 'bar':
            fig.add_trace(go.Bar(x=model.state_timeline.query('Run == ' + str(run)).index.get_level_values(1), y=model.state_timeline[col],marker_color=colormap[col],name=col))
            fig.update_layout(barmode='stack')
        
    fig.update_layout( title="State progression over time for Model {}".format(model.name), xaxis_title="Time", yaxis_title="Population", font=dict( size=12,color="#7f7f7f"))
    fig.show()
    
def generate_sample(distribution, params = {}):
    
    sample = 0 
    if params != {}:
        distribution = distribution.lower()

        if distribution == 'exponential':
            sample = np.random.exponential(1.0/params['scale'])

        if distribution == 'gaussian' or distribution == 'normal':
            sample = np.random.normal(loc=params['loc'], scale=params['scale'])

        if distribution == 'uniform':
            sample = np.random.uniform(low=params['low'], high=params['high'])

        if distribution == 'triangle' or distribution == 'triangular':
              sample = np.random.triangular(left=params['left'], mode=params['mode'],right=params['right'])
        
    return sample