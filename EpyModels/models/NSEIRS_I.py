import numpy as np
import pandas as pd
from .SEIR import SEIR

class NSEIRS_I(SEIR):
    '''
    Not_susceptible Susceptible-Infected-Recovered-Susceptible with Interventions (NSEIRS_I) model
    '''
    
    def __init__(self, name, population, init_s, init_e, init_i, init_r, init_n, beta, incubation_rate, recovery_rate, susceptible_rate, mortality_rate, interventions={}, dist_params={}):
        '''
        Initializes an SEIRS_I model
        -----
        INPUTS:
        name: Model name
        population: Number of individuals in the simulation
        init_s: Initial proportion of the population in state S (Susceptible)
        init_e: Initial proportion of the population in state E (Exposed)
        init_i: Initial proportion of the population in state I (Infected)
        init_r: Initial proportion of the population in state R (Recovered)
        init_n: Initial proportion of the population in state N (Not Susceptible)
        beta: Average number of susceptible individuals infected by one infected individual in one time step
        incubation_rate: Average number of exposed individuals who become infected by one infected per time unit
        susceptible_rate: Average number of recovered individuals becoming susceptible again in one time step
        recovery_rate: Average number of infected individuals recovered in one time step
        mortality_rate: Average number of infected individuals who die in one time step
        interventions: D
        dist_params: Dictonary with parameter names, distribution names, and distribution parameters (using numpy.random.* distribution parameter names) (e.g. {param_name : {prob_dist_name : {dist_params: dist_param_values }}})
        -----
        '''
        
        super(NSEIRS_I, self).__init__(name, population, init_s, init_e, init_i, init_r, beta, incubation_rate, recovery_rate, mortality_rate,dist_params={})
        self.states['N'] = np.round(init_n*population)
        self.interventions = interventions
        
        # Dataframe with state values over time
        self.state_timeline = pd.DataFrame([[i for i in [1,0,self.states['S'], self.states['E'], self.states['I'], self.states['R'], self.states['N'], self.states['D']]]], columns=['Run','Time','Susceptible', 'Exposed','Infected', 'Recovered', 'Not Susceptible','Dead'],
                                   dtype='int') 
        self.state_timeline.set_index(['Run','Time'],inplace=True)
        self.time = 1
        self.run_num = 1
                
    def reset(self):
        '''
        Resets model states to initial conditions
        '''
        self.state_timeline = self.state_timeline.iloc[[0]]
        self.states['S'], self.states['E'], self.states['I'], self.states['R'], self.states['N'], self.states['D'] = self.state_timeline.iloc[0].tolist()
        self.time = 1
        self.run_num = 1
 
    def next(self):
        '''
        Updates model states by one time unit
        '''
        
        intervention_time = self.interventions.get("intervention_time", None) 
        if intervention_time is not None:
            if self.time == intervention_time:
                effect = self.interventions.get("intervention_effect", 0)
                self.params['beta'] = self.params['beta']*(1.0-effect)
                
        im = 0.0
        vaccination_time = self.interventions.get("vaccination_time", None) 
        if vaccination_time is not None:
            if self.time >= vaccination_time:
                im = self.interventions.get("vaccination_rate", 0)
        
        # calculate changes in state counts
        delta_S = -min(self.states['S'] * self.params['beta'] * self.states['I']/self.params['population'], self.states['S']) - min(im,self.states['S'])
        delta_E = min(self.states['S'] * self.params['beta'] * self.states['I']/self.params['population'], self.states['S']) - self.params['incubation_rate'] * self.states['E']  
        delta_I = self.params['incubation_rate'] * self.states['E']  - self.states['I']*self.params['recovery_rate'] - self.states['I']*self.params['mortality_rate']   
        delta_R = self.states['I']*(self.params['recovery_rate'])      
        delta_D = self.states['I']*self.params['mortality_rate']
        delta_N = min(im,self.states['S']) 

        # update states
        self.states['S'] += delta_S
        self.states['E'] += delta_E
        self.states['I'] += delta_I
        self.states['R'] += delta_R
        self.states['D'] += delta_D
        self.states['N'] += delta_N

        # Update state dataframe
       # self.state_timeline.loc[len(self.state_timeline)] = np.round([i * self.params['population'] for i in [self.states['S'], self.states['I'], self.states['R'], self.states['D']]],0)
        
        self.state_timeline.loc[(self.run_num,self.time),:] = np.round([i for i in [self.states['S'], self.states['E'],self.states['I'], self.states['R'], self.states['N'], self.states['D']]],0)
        
        self.time += 1

    def run(self, T, runs=1):
        ''' 
        Run the simulation until time T
        '''
        for run in range(1,runs+1):
            self.time = 1
            for time in range(1,T+1):
                self.next()
            self.run_num += 1