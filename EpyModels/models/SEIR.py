import numpy as np
import pandas as pd
from .SIR import SIR

class SEIR(SIR):
    '''
    Simple SEIR model
    '''
    
    def __init__(self, name, population, init_s, init_e, init_i, init_r, beta, incubation_duration, mortality_rate, recovery_duration, mortality_duration, dist_params={}):
        
        
        
        '''
        Initializes an SIR model
        -----
        INPUTS:
        name: Model name
        population: Number of individuals in the simulation
        init_s: Initial proportion of the population in state S (Susceptible)
        init_e: Initial proportion of the population in state E (Exposed)
        init_i: Initial proportion of the population in state I (Infected)
        init_r: Initial proportion of the population in state R (Recovered)
        beta: Average number of susceptible individuals infected by one infected individual per time unit
        incubation_rate: Average number of exposed individuals who become infected by one infected per time unit
        recovery_rate: Average number of infected individuals recovered per time unit
        mortality_rate: Average number of infected individuals who die per time unit
        dist_params: Dictonary with parameter names, distribution names, and distribution parameters (using numpy.random.* distribution parameter names) (e.g. {param_name : {prob_dist_name : {dist_params: dist_param_values }}})
        -----
        '''
        super(SEIR, self).__init__(name, population, init_s, init_i, init_r, beta, mortality_rate, recovery_duration, mortality_duration, dist_params={})
        
        self.states['E'] = np.round(init_e*population) 
        self.params['incubation_duration'] = incubation_duration
        
        # Dataframe with state values over time
        self.state_timeline = pd.DataFrame([[i for i in [1,0,self.states['S'], self.states['E'], self.states['I'], self.states['R'], self.states['D']]]], columns=['Run','Time','Susceptible', 'Exposed','Infected', 'Recovered' ,'Dead'], dtype='int')
        self.state_timeline.set_index(['Run','Time'],inplace=True)
        
    def reset(self):
        '''
        Resets model states to initial conditions
        '''
        self.state_timeline = self.state_timeline.iloc[[0]]
        self.states['S'], self.states['E'], self.states['I'], self.states['R'], self.states['D'] = self.state_timeline.iloc[0].tolist()
        self.time = 1
        self.run_num = 1
        
    def calculate_deltas(self):
        # calculate changes in state counts
        delta_S = - min(self.states['S'] * self.params['beta'] * self.states['I'] / self.params['population'], self.states['S'])
        delta_E = min(self.states['S'] * self.params['beta'] * self.states['I'] / self.params['population'], self.states['S']) -  self.states['E'] / self.params['incubation_duration']
        
        delta_I = self.states['E'] / self.params['incubation_duration'] - self.states['I']/self.params['duration_recovery'] - self.states['I']/self.params['mortality_duration']  
        delta_R = (1-self.params['mortality_rate']) * self.states['I'] / self.params['duration_recovery']
        delta_D = self.params['mortality_rate'] * self.states['I'] / self.params['mortality_duration']
        
        return delta_S, delta_E, delta_I, delta_R, delta_D
 
 
    def step(self):
        '''
        Updates model states by one time unit
        '''
        
        # calculate changes in state counts       
        delta_S, delta_E, delta_I, delta_R, delta_D = self.calculate_deltas()

        # update states
        self.states['S'] += delta_S
        self.states['E'] += delta_E
        self.states['I'] += delta_I
        self.states['R'] += delta_R
        self.states['D'] += delta_D

        # Update state dataframe
        self.state_timeline.loc[(self.run_num,self.time),:] = np.round([i for i in [self.states['S'], self.states['E'],self.states['I'], self.states['R'], self.states['D']]],0)

        self.time += 1

