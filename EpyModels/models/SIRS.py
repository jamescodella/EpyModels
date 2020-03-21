import numpy as np
import pandas as pd
from .SIR import SIR

class SIRS(SIR):
    '''
    Susceptible-Infected-Recovered-Susceptible (SIRS) model
    '''
    
    def __init__(self, name, population, init_s, init_i, init_r, beta, susceptible_rate, recovery_rate, mortality_rate,dist_params={}):
        '''
        Initializes an SIRS model
        -----
        INPUTS:
        name: Model name
        population: Number of individuals in the simulation
        init_s: Initial proportion of the population in state S (Susceptible)
        init_i: Initial proportion of the population in state I (Infected)
        init_r: Initial proportion of the population in state R (Recovered)
        beta: Average number of susceptible individuals infected by one infected individual per time unit
        susceptible_rate: Average number of recovered individuals becoming susceptible again per time unit
        recovery_rate: Average number of infected individuals recovered per time unit
        mortality_rate: Average number of infected individuals who die per time unit
        dist_params: Dictonary with parameter names, distribution names, and distribution parameters (using numpy.random.* distribution parameter names) (e.g. {param_name : {prob_dist_name : {dist_params: dist_param_values }}})
        -----
        '''
        super(SIRS, self).__init__(name, population, init_s, init_i, init_r, beta, recovery_rate, mortality_rate,dist_params={})

        self.params['susceptible_rate'] = susceptible_rate 
 
    def next(self):
        '''
        Updates model states by one time unit
        '''
        
        # calculate changes in state counts
        delta_S = -min(self.states['S'] * (self.params['beta']) * self.states['I']/self.params['population'], self.states['S']) + self.states['R'] * self.params['susceptible_rate']   
        delta_I = min(self.states['S'] * (self.params['beta']) * self.states['I']/self.params['population'], self.states['S']) - self.states['I'] * self.params['recovery_rate'] - self.states['I'] * self.params['mortality_rate']       
        delta_R = self.states['I'] * (self.params['recovery_rate']) - self.states['R'] * self.params['susceptible_rate']        
        delta_D = self.states['I'] * self.params['mortality_rate']
        
        # update states
        self.states['S'] += delta_S
        self.states['I'] += delta_I
        self.states['R'] += delta_R
        self.states['D'] += delta_D

        # Update state dataframe
        self.state_timeline.loc[(self.run_num,self.time),:] = np.round([i for i in [self.states['S'], self.states['I'], self.states['R'], self.states['D']]],0)
        self.time += 1