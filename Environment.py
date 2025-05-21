import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli
from scipy import stats
import copy


class rotting_many_Env: #slow rotting
    def inverse_cdf(self, y, beta):
    # Computed analytically
        return 1-(1-y)**(1/beta)

    def sample_distribution(self,beta):
        uniform_random_sample = random.random()
        return self.inverse_cdf(uniform_random_sample,beta)
    
    # def sample_distribution(self,beta):
    #     u = np.random.uniform(0, 1, 1)
    
    # # Apply the inverse CDF transformation
    #     mu_sample = 1 - u**(1 / beta)
    #     # uniform_random_sample = random.random()
    #     return mu_sample
    
    def __init__(self,rho,seed,T,beta=1,L_bool=False):
        np.random.seed(seed)
        self.optimal=1
        self.exp_reward=[]
        self.rho=rho
        self.T=T
        self.beta=beta
        self.bool=L_bool
        # if beta==1:
        #     for k in range(self.T):
        #         self.exp_reward[k]=np.random.uniform(0,1)
        # else: 
        #     for k in range(self.T):
        #         self.exp_reward[k]=self.sample_distribution(beta)


    def sample_arm(self):
        if self.beta==1:
            self.exp_reward.append(np.random.uniform(0,1))
        else: 
            self.exp_reward.append(self.sample_distribution(self.beta))
        
    def observe(self,k,t):
        # S=round(np.sqrt(self.T))
        gap=1-self.exp_reward[k]
        reward=self.exp_reward[k]+np.random.uniform(-gap,gap)
        exp_reward=copy.deepcopy(self.exp_reward[k])
        # if t%S==0:
        #     self.exp_reward[k]= np.random.uniform(0,1)
        rho_t=self.rho*(100/(t+1))
        if self.bool==True:
            L=int(np.sqrt(self.T))
            if t<=L:
                rho_t=1
            else:
                rho_t=0
        # # rho_t=1/10
        self.exp_reward[k]=max(0,self.exp_reward[k]-rho_t)
        return exp_reward, reward
    

    