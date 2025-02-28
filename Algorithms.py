# infinite-armed bandit algorithms
import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environment import *
from tqdm import tqdm

    
class Elimination: 
    """
    Elimination algorithm for tracking significant shifts (Alg 2 in paper) 
    """
    def __init__(self,T,seed,beta,Environment):
        """
        parameters:
            T: time horizon
            seed: random seed
            beta: reservoir regularity parameter
            Environment: bandit environment
        """
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float) # experienced observed rewards
        self.r_Exp=np.zeros(T,float) # experienced mean rewards
        K_sum=0
        K=0 # subsample size
        C_1=1 # elimination threshold
        t=0 # round
        t_ml=0
        if T==1:
            K=0
            self.Env.sample_arm()            
            self.r_Exp[0],self.r[0]=self.Env.observe(K,0)
        else:
            while t<T-1:
                detect_change=False
                for m in range(math.ceil(math.log(T))): # doubling epochs
                    m=m+1                        
                    K_sum+=K # running count of total number of subsampled arms
                    K=math.ceil(2**((m+1)*beta/(beta+1))) # set subsample size 
                    for k in range(K): # subsample arms
                        self.Env.sample_arm()
                    t_ml=t # start of epoch       
                    delta_sum=np.zeros(K)
                    arm_set=list(range(K)) # set of candidate arms
                    for t in range(t_ml+1,min(t_ml+2**m,T)):
                        k=random.choice(arm_set) # play an arm uniformly at random
                        self.r_Exp[t],self.r[t]=self.Env.observe(k+K_sum,t) # mean and observed reward of arm k
                        delta_sum[k]=delta_sum[k]+((1-self.r[t])*len(arm_set)) # importance-weighted reward estimate
                        if delta_sum[k]>=C_1*K*np.log(T): 
                            arm_set.remove(k) # eliminate arm
                        if len(arm_set)==0:
                            detect_change=True # restart episode
                            break 
                    
                    if detect_change or t==T-1:
                        break

    def rewards(self):
        """
        return mean rewards of played arms
        """
        return self.r_Exp  
        
class base_alg: 
    """
    UCB base algorithm to be used in blackbox
    """
    def __init__(self,T,K,seed,Environment):
        """
        parameters:
            T: time horizon
            K: number of arms
            seed: random seed
            beta: reservoir regularity parameter
            Environment: bandit environment
        """
        np.random.seed(seed)
        self.Env=Environment
        self.n=np.zeros(K)
        self.mu=np.zeros(K)
        self.ucb=np.zeros(K)
        self.K=K
        self.T=T

    def run(self,t):
        """
        select arm at round t based on UCB index
        """
        for k in range(self.K):
            self.ucb[k]=self.mu[k]+math.sqrt(2*math.log(self.T)/self.n[k])
        k=np.argmax(self.ucb)

        return k

    def update(self, t, k, r):
        """
        update arm k's estimated mean reward at round t based on observation r
        """
        self.mu[k] = (self.mu[k] * self.n[k] + r[t]) / (self.n[k] + 1)
        self.n[k] += 1
        self.ucb[k] = self.mu[k] + math.sqrt(2 * math.log(self.T) / self.n[k])
    
class black_box: 
    """
    Blackbox algorithm for nonstationary infinite-armed bandits (Alg 1 in paper)
    """
    def __init__(self,T,seed,beta,Environment):
        """
        parameters:
            T: time horizon
            seed: random seed
            beta: reservoir regularity parameter
            Environment: bandit environment
        """
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float) # experienced observed rewards
        self.r_Exp=np.zeros(T,float) # experienced mean rewards
        K_sum=0 # running count of total number of subsampled arms
        K=0 # subsample size
        C_1=1 # restart threshold
        t=0
        t_ml=0 # start round of epoch
        if T==1:
            k=0
            self.Env.sample_arm()            
            self.r_Exp[0],self.r[0]=self.Env.observe(k,0)
        else:
            while t<T-1:
                detect_change=False
                for m in range(math.ceil(math.log(T))): # doubling epochs
                    m=m+1                        
                    K_sum+=K
                    # set subsampling rate
                    if beta>=1:
                        K=min(math.ceil(2**(m*beta/(beta+1))*(np.log(T)**(1/(beta+1)))),2**m)
                    else:
                        K=min(math.ceil(2**(m*beta/2)*np.log(T)),2**m)
                        
                    base=base_alg(T,K,seed,self.Env) # initialize base algorithm
                    for k in range(K):
                        self.Env.sample_arm() # sample arm
                    t_ml=t # set start of epoch       
                    for t in tqdm(range(t_ml+1,min(t_ml+2**m,T))):
                        k=base.run(t) # choose arm according to base
                        self.r_Exp[t],self.r[t]=self.Env.observe(k+K_sum,t) # get mean and observed reward of this arm
                        base.update(t,k,self.r) # update base alg's information
                        if sum(1-self.r[t_ml:t+1])>=C_1*max(K,2**(m/2))*(math.log(T)**3): # check for large regret to restart
                            detect_change=True
                            break
                    if detect_change or t==T-1:
                        break

    def rewards(self):
        """
        return mean rewards of played arms
        """
        return self.r_Exp  
        


class AUCBT_AW: 
    """
    Adaptive UCB-Threshold with Adaptive Sliding Window (AUCBT-AW) algorithm from Kim et al., 2024.
    """
    def __init__(self,T,seed,Environment):    
        """
        parameters:
            T: time horizon
            seed: random seed
            beta: reservoir regularity parameter
            Environment: bandit environment
        """
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float) # observed experienced rewards
        self.r_Exp=np.zeros(T,float) # mean rewards of experienced arms
        if T==1:
            k=0
            self.Env.sample_arm()            
            self.r_Exp[0],self.r[0]=self.Env.observe(k,0)
        else:
            k=0
            H=math.ceil(math.sqrt(T)) # epoch size
            B=math.ceil(math.log2(H))
            alpha=min(1,math.sqrt(B*math.log(B)/((math.e-1)*math.ceil(T/H)))) # exploration parameter
            w=np.ones(B) # exponential weights
            p=np.zeros(B) # play probabilities based on exp weights
            for i in tqdm(range(math.ceil(T/H))):
                if k!=0: 
                    k=k+1
                self.Env.sample_arm()
                self.r_Exp[i*H],self.r[i*H]=self.Env.observe(k,0) # get mean and observed rewards
                
                p=(1-alpha)*w/w.sum()+alpha/B
                j=np.random.choice(B,1,p=p) # sample arms according to play probabilities
                delta=(1/2)**(j) # threshold on gap
                t_=i*H # starting round of new exploration phase

                for t in range(i*H+1,min(H*(i+1),T)):
                    for l in range(math.floor(math.log2(t-t_))+1):  
                        win=2**l
                        s=max(t-win,t_)
                        sum_r=np.sum(self.r[s:t])
                        n=t-s
                        mu=sum_r/n
                        ucb=mu+math.sqrt(12*math.log(H)/n) # compute ucb index

                        if ucb<1-delta:
                            k=k+1 # sample new arm
                            t_=t
                            self.Env.sample_arm()
                            break
                    self.r_Exp[t],self.r[t]=self.Env.observe(k,t)
                # update exponential weights
                w[j]=w[j]*math.exp(alpha/(B*p[j])*(1/2+self.r[i*H:H*(i+1)].sum()/(100*H*math.log(T)+4*math.sqrt(H*math.log(T)))))    
                
    def rewards(self):
        """
        return mean rewards of played arms
        """
        return self.r_Exp 
    


class SSUCB:
    """
    Subsampling UCB (SSUCB) algorithm of Bayati et al., (2020).
    """
    def __init__(self,K,T,seed,Environment):
        """
        parameters:
            T: time horizon
            seed: random seed
            beta: reservoir regularity parameter
            Environment: bandit environment
        """                   
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float) # observed rewards of played arms
        self.r_Exp=np.zeros(T,float) # mean rewards of played arms
        n=np.zeros(K) # pull count vector of arms
        mu=np.zeros(K) # mean reward vector
        ucb=np.zeros(K) # ucb index vector
        for k in range(K): # subsample arms
            self.Env.sample_arm()            

        for t in tqdm(range(T)):
            if t<K:
                k=t
            else:         
                k=np.argmax(ucb) # play arm maximizing ucb index
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k,n[k]) # get mean and observed rewards
            mu[k]=(mu[k]*(n[k])+self.r[t])/(n[k]+1)                
            n[k]=n[k]+1
            ucb[k]=mu[k]+math.sqrt(2*math.log(1+(t+1)*(math.log(t+1))**2)/n[k]) # update ucb index

    def rewards(self):
        """
            return mean rewards of played arms
        """
        return self.r_Exp  
