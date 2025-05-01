import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environment import *
from tqdm import tqdm

    

    
class Elimination: ##Algorithm 1
    def __init__(self,T,seed,beta,Environment):
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
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
                for m in range(math.ceil(math.log(T))):
                    m=m+1                        
                    K_sum+=K
                    K=min(math.ceil(2**((m+1)*beta/(beta+1))),2**m)
                    # K=math.ceil(2**(m/2)*np.sqrt(np.log(T)))
                    # base=base_alg(T,K,seed,self.Env)
                    for k in range(K):
                        self.Env.sample_arm()
                    # t_ml=t_ml+2**(m-1)            
                    t_ml=t       
                    # print(t_ml)
                    delta_sum=np.zeros(K)
                    arm_set=list(range(K))
                    for t in range(t_ml+1,min(t_ml+2**m,T)):
                        # print(t_ml,min(t_ml+2**m,T)-1)
                        # if t&100==0:
                        print(t)
                        k=random.choice(arm_set)   
                        # base.run(t)
                        self.r_Exp[t],self.r[t]=self.Env.observe(k+K_sum,t)
                        delta_sum[k]=delta_sum[k]+((1-self.r[t])*len(arm_set))
                        # base.update(t,k,self.r)
                        # print(delta_sum[k])
                        if delta_sum[k]>=C_1*K*np.log(T):
                            arm_set.remove(k)
                            # print('remove******************************')
                        if len(arm_set)==0:
                            # t_ml=t
                            detect_change=True
                            break 
                            
                        # if sum(1-self.r[t_ml:t+1])>=C_1*(math.sqrt((t-t_ml+1)*math.log(T))+K*math.log(T)):
                          
                    if detect_change or t==T-1:
                        break

    def rewards(self):
        return self.r_Exp  
        



class base_alg: 
    def __init__(self,T,K,seed,Environment):
 
        np.random.seed(seed)
        self.Env=Environment
        self.n=np.zeros(K)
        self.mu=np.zeros(K)
        self.ucb=np.zeros(K)
        self.K=K
        self.T=T

    def run(self,t):
        for k in range(self.K):
            self.ucb[k]=self.mu[k]+math.sqrt(2*math.log(self.T)/self.n[k])
        k=np.argmax(self.ucb)

        return k
    def update(self, t, k, r):
        self.mu[k] = (self.mu[k] * self.n[k] + r[t]) / (self.n[k] + 1)
        self.n[k] += 1
        self.ucb[k] = self.mu[k] + math.sqrt(2 * math.log(self.T) / self.n[k])
    
class black_box: ##Algorithm 2
    def __init__(self,T,seed,beta,Environment):
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
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
                for m in range(math.ceil(math.log(T))):
                    m=m+1                        
                    K_sum+=K
                    if beta>=1:
                        K=math.ceil(2**(m*beta/(beta+1)))
                    # K=math.ceil(2**(m/2)*np.sqrt(np.log(T)))
                    else:
                        K=math.ceil(2**(m*beta/2))
                        
                    base=base_alg(T,K,seed,self.Env)
                    for k in range(K):
                        self.Env.sample_arm()
                    # t_ml=t_ml+2**(m-1)     
                    t_ml=t       
                    # print(t_ml)
                    for t in tqdm(range(t_ml+1,min(t_ml+2**m,T))):
                        # print(t_ml,min(t_ml+2**m,T)-1)
                        # if t&100==0:
                        print(t)
                        k=base.run(t)
                        self.r_Exp[t],self.r[t]=self.Env.observe(k+K_sum,t)
                        base.update(t,k,self.r)
                        if sum(1-self.r[t_ml:t+1])>=C_1*max(K,2**(m/2))*(math.log(T)**3):
                            # t_ml=t
                            detect_change=True
                            break
                    if detect_change or t==T-1:
                        break

    def rewards(self):
        return self.r_Exp  
        


class AUCBT_AW: 
    def __init__(self,T,seed,Environment):
        print('AUCBT-ASW')
        
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
                self.r_Exp[i*H],self.r[i*H]=self.Env.observe(k,0)
                
                p=(1-alpha)*w/w.sum()+alpha/B
                j=np.random.choice(B,1,p=p)
                delta=(1/2)**(j)
                t_=i*H

                for t in range(i*H+1,min(H*(i+1),T)):


                    for l in range(math.floor(math.log2(t-t_))+1):  
                        win=2**l
                        s=max(t-win,t_)
                        sum_r=np.sum(self.r[s:t])
                        n=t-s
                        mu=sum_r/n
                        ucb=mu+math.sqrt(12*math.log(H)/n)

                        if ucb<1-delta:
                            k=k+1
                            t_=t
                            self.Env.sample_arm()
                            break
                    self.r_Exp[t],self.r[t]=self.Env.observe(k,t)
                w[j]=w[j]*math.exp(alpha/(B*p[j])*(1/2+self.r[i*H:H*(i+1)].sum()/(100*H*math.log(T)+4*math.sqrt(H*math.log(T)))))    
    def rewards(self):
        return self.r_Exp 
    


class SSUCB:
    def __init__(self,K,T,seed,Environment):
        print('SSUCB')
                    
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
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
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k,t)
            mu[k]=(mu[k]*(n[k])+self.r[t])/(n[k]+1)                
            n[k]=n[k]+1
            ucb[k]=mu[k]+math.sqrt(2*math.log(1+(t+1)*(math.log(t+1))**2)/n[k])

    def rewards(self):
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
                print(t)
            else:         
                k=np.argmax(ucb)    
            n_hist[k,t]=1    
            self.r_Exp[t],self.r[t]=self.Env.observe(k,t)
            r_hist[k,t]=self.r[t]
            print('window',max(t-w,0))
            mu[k]=sum(r_hist[k,max(t-w,0):t])/sum(n_hist[k,max(t-w,0):t])
            # mu[k]=(mu[k]*(n[k])+self.r[t])/(n[k]+1)                
            n[k]=n[k]+1
            ucb[k]=mu[k]+math.sqrt(2*math.log(1+(t+1)*(math.log(t+1))**2)/sum(n_hist[k,max(t-w,0):t]))

    def rewards(self):
        return self.r_Exp  