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
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)  # experienced observed rewards
        self.r_Exp=np.zeros(T,float)  # experienced mean rewards
        A=0
        K_sum=0
        K=0
        t_l=0
        C_1=1
        t=0
        t_ml=0
        if T==1:
            k=0
            self.Env.sample_arm()            
            self.r_Exp[0],self.r[0]=self.Env.observe(k,0)
        else:
            while t<T-1:
                detect_change=False
                for m in range(math.ceil(math.log(T))): # doubling epochs
                    m=m+1                        
                    K_sum+=K  # running count of total number of subsampled arms
                    K=min(math.ceil(2**((m+1)*beta/(beta+1))),2**m)
                    for k in range(K): # subsample arms
                        self.Env.sample_arm()
                    t_ml=t       
                    delta_sum=np.zeros(K)
                    arm_set=list(range(K))
                    for t in range(t_ml+1,min(t_ml+2**m,T)):
                        print(t)
                        k=random.choice(arm_set)    # play an arm uniformly at random
                        self.r_Exp[t],self.r[t]=self.Env.observe(k+K_sum,t)  # mean and observed reward of arm k
                        delta_sum[k]=delta_sum[k]+((1-self.r[t])*len(arm_set))
                        if delta_sum[k]>=C_1*K*np.log(T):
                            arm_set.remove(k) # eliminate arm
                        if len(arm_set)==0:
                            detect_change=True  # restart episode
                            break 
                            
                          
                    if detect_change or t==T-1:
                        break

    def rewards(self):
        """
        return mean rewards of played armsMore actions
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
        self.r=np.zeros(T,float)  # experienced observed rewards
        self.r_Exp=np.zeros(T,float)  # experienced mean rewards
        A=0
        K_sum=0 # running count of total number of subsampled arms
        K=0  # subsample size
        t_l=0
        C_1=1
        t=0
        t_ml=0 # start round of epoch
        if T==1:
            k=0
            self.Env.sample_arm()            
            self.r_Exp[0],self.r[0]=self.Env.observe(k,0)
        else:
            while t<T-1:
                detect_change=False
                for m in range(math.ceil(math.log(T))):  # doubling epochs
                    m=m+1                        
                    K_sum+=K
                    # set subsampling rate

                    if beta>=1:
                        K=math.ceil(2**(m*beta/(beta+1)))
                    else:
                        K=math.ceil(2**(m*beta/2))
                        
                    base=base_alg(T,K,seed,self.Env) # initialize base algorithm
                    for k in range(K):
                        self.Env.sample_arm() # sample arm
                    t_ml=t        # set start of epoch   
                    for t in tqdm(range(t_ml+1,min(t_ml+2**m,T))):
                        print(t)
                        k=base.run(t) # choose arm according to base
                        self.r_Exp[t],self.r[t]=self.Env.observe(k+K_sum,t) # get mean and observed reward of this arm
                        base.update(t,k,self.r)  # update base alg's information
                        if sum(1-self.r[t_ml:t+1])>=C_1*max(K,2**(m/2))*(math.log(T)**3):
                            # check for large regret to restart
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
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        if T==1:
            k=0
            self.Env.sample_arm()            

            self.r_Exp[0],self.r[0]=self.Env.observe(k,0)
        else:
            k=0
            H=math.ceil(math.sqrt(T))
            B=math.ceil(math.log2(H))
            alpha=min(1,math.sqrt(B*math.log(B)/((math.e-1)*math.ceil(T/H))))
            w=np.ones(B)
            p=np.zeros(B)
            for i in tqdm(range(math.ceil(T/H))):
                if k!=0: 
                    k=k+1
                self.Env.sample_arm()
                self.r_Exp[i*H],self.r[i*H]=self.Env.observe(k,0) # get mean and observed rewards
                
                p=(1-alpha)*w/w.sum()+alpha/B # sample arms according to play probabilities
                j=np.random.choice(B,1,p=p)
                delta=(1/2)**(j) # threshold on gap
                t_=i*H  # starting round of new exploration phase

                for t in range(i*H+1,min(H*(i+1),T)):


                    for l in range(math.floor(math.log2(t-t_))+1):  
                        win=2**l
                        s=max(t-win,t_)
                        sum_r=np.sum(self.r[s:t])
                        n=t-s
                        mu=sum_r/n
                        ucb=mu+math.sqrt(12*math.log(H)/n)  # compute ucb index

                        if ucb<1-delta:
                            k=k+1  # sample new arm
                            t_=t
                            self.Env.sample_arm()
                            break
                    self.r_Exp[t],self.r[t]=self.Env.observe(k,t)  # get mean and observed rewards
                w[j]=w[j]*math.exp(alpha/(B*p[j])*(1/2+self.r[i*H:H*(i+1)].sum()/(100*H*math.log(T)+4*math.sqrt(H*math.log(T)))))                    # update exponential weights

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

        self.r_Exp=np.zeros(T,float)  # mean rewards of played arms

        n=np.zeros(K) # pull count vector of arms
        mu=np.zeros(K)  # mean reward vector
        ucb=np.zeros(K) # ucb index vector
        for k in range(K):  # subsample arms
            self.Env.sample_arm()            

        for t in tqdm(range(T)):
            if t<K:
                k=t
            else:         
                k=np.argmax(ucb)      # play arm maximizing ucb index
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k,t)  # get mean and observed rewards
            mu[k]=(mu[k]*(n[k])+self.r[t])/(n[k]+1)                
            n[k]=n[k]+1
            ucb[k]=mu[k]+math.sqrt(2*math.log(1+(t+1)*(math.log(t+1))**2)/n[k])  # update ucb index

    def rewards(self):
        """
            return mean rewards of played arms
        """
        return self.r_Exp  

class SSUCB_SW:
    def __init__(self,K,T,seed,Environment):
        print('SSUCB-SW')
                    
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        w=int(np.sqrt(T))
        n_hist=np.zeros((K,T))
        r_hist=np.zeros((K,T))
        n=np.zeros(K)
        mu=np.zeros(K)
        ucb=np.zeros(K)
        for k in range(K):
            self.Env.sample_arm()            
        for t in tqdm(range(T)):
            if t<K:
                k=t
            else:         
                k=np.argmax(ucb)    
            n_hist[k,t]=1    
            self.r_Exp[t],self.r[t]=self.Env.observe(k,t)
            r_hist[k,t]=self.r[t]
            if t>1:
                mu[k]=np.sum(r_hist[k,max(t-w,0):t])/np.sum(n_hist[k,max(t-w,0):t])
            n[k]=n[k]+1
            if t>1:
                ucb[k]=mu[k]+math.sqrt(2*math.log(1+(t+1)*(math.log(t+1))**2)/np.sum(n_hist[k,max(t-w,0):t]))
    def rewards(self):
        return self.r_Exp  