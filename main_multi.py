# script to run experiments
from Environment import *
from Algorithms import *
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
import multiprocessing

def run(T,num,repeat,beta, i, j):
    """
    Run bandit experiments
    parameters:
        T: horizon
        num: number of investigated horizon times over maximum time horizon
        repeat: number of running algorithms using different seeds.
        beta: reservoir regularity parameter
    """
    T_1=int(T/num)
    num=num+1
    T_list=np.zeros(num)
    avg=dict()
    sd=dict()
    regret_sum_list=dict()
    avg_regret_sum=dict()

    alg_list=['Blackbox', 'AUCBT_AW', 'SSUCB','Elimination']

    print('num:', i)
    if i==0:
        T=1
    else:
        T=T_1*i
    if T==1:
        rho=1
    else:
        rho=1
    for name in alg_list:
        avg[name]=np.zeros(T,float)
        sd[name]=np.zeros(T,float)
        regret_sum_list[name]=np.zeros((repeat,T),float)
        avg_regret_sum[name]=np.zeros(T,float)
    K=int(max(np.sqrt(T),T**(beta/(beta+1))))
    T_list[i]=T
    regret=np.zeros(T,float) # per-round regrets
    regret_sum=np.zeros(T,float) # cumulative regret

    ## Run model

    print('repeat: ',j)
    for name in alg_list:
        print(name)
        seed=j # random seed
        Env=rotting_many_Env(rho,seed,T,beta) # make environment
        if name=='Blackbox':
            algorithm=black_box(T,seed,beta,Env)
        if name=='AUCBT_AW':
            algorithm=AUCBT_AW(T,seed,Env)
        if name=='SSUCB':
            algorithm=SSUCB(K,T,seed,Env)
        if name=='Elimination':
            algorithm=Elimination(T,seed,beta,Env)
    
        opti_rewards=Env.optimal # get optimal rewards
            
        regret=opti_rewards-algorithm.rewards() # compute regret
        regret_sum=np.cumsum(regret)  
   
        filename_1=name+'T'+str(T)+'num'+str(i)+'beta'+str(beta)+'repeat'+str(j)+'regret.txt'
        with open('./result/'+filename_1, 'wb') as f:
            pickle.dump(regret_sum, f)
            f.close()
 

def run_multiprocessing(T,num,repeat, beta):
    """
    Run bandit experiments via multi-processing.
    parameters:
        T: horizon
        num: number of investigated horizon times over maximum time horizon
        repeat: number of running algorithms using different seeds
        beta: reservoir regularity parameter
    """
    Path("./result").mkdir(parents=True, exist_ok=True)

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(run, [( T,num,repeat,beta, i, j) for j in range(repeat) for i in range(num+1)])

    pool.close()
    pool.join()
    
def plot(T,num,repeat, beta):
    """
    Plot results of experiments
    parameters:
        T: horizon
        num: number of investigated horizon times over maximum time horizon
        repeat: number of running algorithms using different seeds
        beta: reservoir regularity parameter
    """
    T_1=int(T/num)
    num=num+1

    print('load data without running')
    regret_list=dict()
    std_list=dict()
    regret_list_T=dict()
    std_list_T=dict()
    T_list=np.zeros(num)
    avg=dict()
    sd=dict()
    regret_sum_list=dict()
    avg_regret_sum=dict()
    alg_list=['Blackbox', 'AUCBT_AW', 'SSUCB','Elimination']
    

    for name in alg_list:
        regret_list[name]=np.zeros(num,float)
        std_list[name]=np.zeros(num,float)

    for i in range(num):
        print('num:', i)
        if i==0:
            T=1
        else:
            T=T_1*i
        T_list[i]=T
        for name in alg_list:
            avg[name]=np.zeros(T,float)
            sd[name]=np.zeros(T,float)
            regret_sum_list[name]=np.zeros((repeat,T),float)
            avg_regret_sum[name]=np.zeros(T,float)
            
            for j in range(repeat):
                filename_1=name+'T'+str(T)+'num'+str(i)+'beta'+str(beta)+'repeat'+str(j)+'regret.txt'
 
                pickle_file1 = open('./result/'+filename_1, "rb")
                objects = []
                while True:
                    try:
                        objects.append(pickle.load(pickle_file1))
                    except EOFError:
                        break
                pickle_file1.close()
                regret=objects[0]

                
                regret_sum_list[name][j,:]=regret
                avg_regret_sum[name]+=regret
                
            avg[name]=avg_regret_sum[name]/repeat
            sd[name]=np.std(regret_sum_list[name],axis=0)
            
            regret_list[name][i]=avg[name][T-1]
            std_list[name][i]=sd[name][T-1]
            if i==num-1:
                regret_list_T[name]=avg[name]
                std_list_T[name]=sd[name]
    fig,(ax)=plt.subplots(1,1)

    ax.errorbar(x=T_list, y=regret_list['Blackbox'], yerr=1.96*std_list['Blackbox']/np.sqrt(repeat), color="royalblue", capsize=7,capthick=2, elinewidth=2,linewidth=2,
                 marker="^", markersize=0,label='Blackbox (Algorithm 1)',zorder=3,ls='-.') 
    ax.errorbar(x=T_list, y=regret_list['AUCBT_AW'], yerr=1.96*std_list['AUCBT_AW']/np.sqrt(repeat), color="lightseagreen", capsize=7,capthick=2,elinewidth=2,linewidth=2,
                 marker="o", markersize=0,label='AUCBT-AW',zorder=2,ls='--')
    ax.errorbar(x=T_list, y=regret_list['SSUCB'], yerr=1.96*std_list['SSUCB']/np.sqrt(repeat), color="gray", capsize=7,capthick=2,elinewidth=2,linewidth=2,
                 marker="s", markersize=0,label='SSUCB',zorder=1,ls=':')
    ax.errorbar(x=T_list, y=regret_list['Elimination'], yerr=1.96*std_list['Elimination']/np.sqrt(repeat), color="salmon", capsize=7,capthick=2,elinewidth=2,linewidth=2,
                 marker="s", markersize=0,label='Elimination (Algorithm 2)',zorder=4,ls='-')
    Path("./plot").mkdir(parents=True, exist_ok=True)


    #font size
    ax.tick_params(labelsize=18)
    plt.rc('legend',fontsize=15)
    ax.yaxis.get_offset_text().set_fontsize(18)
    ax.xaxis.get_offset_text().set_fontsize(18)
    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    labels = [ labels[2], labels[1],labels[0], labels[3]]
    handles=[handles[2],handles[1],handles[0], handles[3]]

    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.xlabel(r'$T$',fontsize=18)
    plt.ylabel(r'$R_T$',fontsize=18)
    plt.savefig('./plot/T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.clf()    

            
if __name__=='__main__':

    
    run_bool=True # True: run model and save data with plot, False: load data with plot.
    
    T=10**5  # Maximum Time horizon
    num=5 # number of investigated horizon times over maximum time horizon
    repeat=5    # number of running algorithms using different seeds.
    beta=1
    if run_bool==True:
        run_multiprocessing(T,num,repeat, beta)
    plot(T,num,repeat, beta)    
    beta=1.2
    if run_bool==True:
        run_multiprocessing(T,num,repeat, beta)
    plot(T,num,repeat, beta) 
    beta=0.8
    if run_bool==True:
        run_multiprocessing(T,num,repeat, beta)
    plot(T,num,repeat, beta) 

