# infinite-armed bandit environments
import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from scipy.stats import bernoulli
from scipy import stats


class rotting_many_Env: 
    """
    slow rotting bandit environment
    """
    def inverse_cdf(self, y, beta):
        """
        analytically compute inverse cdf
        input:
            beta: reservoir regularity 
            y: cdf input
        """
        return 1-(1-y)**(1/beta)

    def sample_distribution(self,beta):
        """
        sample from distribution using inverse cdf
        input:
            beta: reservoir regularity 
        """
        uniform_random_sample = random.random()
        return self.inverse_cdf(uniform_random_sample,beta)
    
    def __init__(self,rho,seed,T,beta=1):
        """
        input:
            rho: rotting amount
            seed: random seed
            T: time horizon
            beta: reservoir regularity 
        """
        np.random.seed(seed)
        self.optimal=1
        self.exp_reward=[] # mean reward vector
        self.rho=rho
        self.T=T
        self.beta=beta


    def sample_arm(self):
        """
        sample a random arm from reservoir
        """
        if self.beta==1:
            self.exp_reward.append(np.random.uniform(0,1))
        else: 
            self.exp_reward.append(self.sample_distribution(self.beta))
        
    def observe(self,k,t):
        """
        observe round t reward of arm k and update via rotting
        return:
            exp_reward: vector of mean rewards
            reward: observed reward of arm k at this round
        """
        gap=1-self.exp_reward[k]
        reward=self.exp_reward[k]+np.random.uniform(-gap,gap) # observed random reward with variance gap
        exp_reward=self.exp_reward[k]
        rho_t=self.rho*(1/(t+1)) # set rotting for this round
        self.exp_reward[k]=max(0,self.exp_reward[k]-rho_t)
        return exp_reward, reward
    

    