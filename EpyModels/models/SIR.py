import numpy as np
import pandas as pd
from ..utils import generate_sample
  
class SIR():
    '''
    Susceptible-Infected-Recovered (SIR) model
    '''
    
    def __init__(self, name, population=1000, init_s=0.999, init_i=0.001, init_r=0, beta=0.2, mortality_rate=0.01, recovery_duration=5, mortality_duration=7, dist_params={}):
        '''
        Initializes an SIR model
        -----
        INPUTS:
        name: Model name
        population: Number of individuals in the simulation
        init_s: Initial proportion of the population in state S (Susceptible)
        init_i: Initial proportion of the population in state I (Infected)
        init_r: Initial proportion of the population in state R (Recovered)
        beta: Average number of susceptible individuals infected by one infected individual per time unit
        recovery_duation: Average duration until an infected individuals recovers
        mortality_rate: Average number of infected individuals who die per time unit
        mortality_duration
        dist_params: Dictonary with parameter names, distribution names, and distribution parameters (using numpy.random.* distribution parameter names) (e.g. {param_name : {prob_dist_name : {dist_params: dist_param_values }}})
        -----
        '''
        
        self.name = name 
        self.params = {}
        self.params['population'] = population
        self.params['beta'] = beta 
        self.params['mortality_rate'] = mortality_rate 
        self.params['duration_recovery'] = recovery_duration 
        self.params['mortality_duration'] = mortality_duration 
        self.states = {}
        self.states['S'] = np.round(init_s*population)
        self.states['I'] = np.round(init_i*population)
        self.states['R'] = np.round(init_r*population)
        self.states['D'] = 0.0
        self.dist_params = dist_params
        
        # Dataframe with state values over time
        self.state_timeline = pd.DataFrame([[i for i in [1,0,self.states['S'], self.states['I'], self.states['R'], self.states['D']]]], columns=['Run','Time','Susceptible', 'Infected', 'Recovered', 'Dead'],  dtype='int') 
        self.state_timeline.set_index(['Run','Time'],inplace=True)
        self.time = 1 # time tracker
        self.run_num = 1  # run tracker
                
    def reset(self):
        '''
        Resets model states to initial conditions
        '''
        self.state_timeline = self.state_timeline.iloc[[0]]
        self.states['S'], self.states['I'], self.states['R'], self.states['D'] = self.state_timeline.iloc[0].tolist()
        self.time = 1
        self.run_num = 1
        
    def calculate_deltas(self):
        # calculate changes in state counts
        delta_S = - min(self.states['S'] * self.params['beta'] * self.states['I']/self.params['population'], self.states['S'])
        delta_I = min(self.states['S'] * self.params['beta'] * self.states['I']/self.params['population'],self.states['S']) - self.states['I']/self.params['duration_recovery']- self.states['I']/self.params['mortality_duration']  
        delta_R = (1-self.params['mortality_rate']) * self.states['I'] / self.params['duration_recovery']
        delta_D = self.params['mortality_rate'] * self.states['I'] / self.params['mortality_duration']
        
        return delta_S, delta_I, delta_R, delta_D
 
    def step(self):
        '''
        Updates model states by one time unit
        '''

        delta_S, delta_I, delta_R, delta_D = self.calculate_deltas()
        # update states
        self.states['S'] += delta_S
        self.states['I'] += delta_I
        self.states['R'] += delta_R
        self.states['D'] += delta_D

        # Update state dataframe
        self.state_timeline.loc[(self.run_num,self.time),:] = np.round([i for i in [self.states['S'], self.states['I'], self.states['R'], self.states['D']]],0)
        
        self.time += 1

    def run(self, T, runs=1):
        ''' 
        Run the simulation until time T
        '''
        for run in range(1,runs+1):
            if self.dist_params != {}:
                for key in self.dist_params.keys():
                    self.params[key] = generate_sample(self.dist_params[key]['distribution'], self.dist_params[key]['values'])
            self.time = 1
            for time in range(1,T+1):
                self.step()
            self.run_num += 1