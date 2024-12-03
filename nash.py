#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize


# In[2]:


N_agents = 4
N_actions = 3

payoff_matrix_shape = tuple([N_actions for i in range(N_agents)] + [N_agents])
payoff_matrix = np.random.normal(1, 4.5, size=payoff_matrix_shape)

payoff_matrix = payoff_matrix/(np.abs(np.sum(payoff_matrix))+0.00001)

#payoff_matrix = np.reshape(np.array([ 3,  8,  4,  0,  5,  3,  1, -3,  9,  4,  8,  5,  4,  7,  2,  6,  2, 9]),(3,3,2))
#payoff_matrix = payoff_matrix.transpose(1, 0, 2)

def zero_sum(x):
    sp = np.sum(x, axis=-1).shape
    return x - np.sum(x, axis=-1).reshape(tuple(list(sp) + [1]))/N_agents

w = zero_sum(payoff_matrix)
print(w.shape)
www = np.sum(w, axis=-1)
print(www)
if np.any(np.nonzero(www)):
    print("WWWWW")

def relu(x):
    return np.maximum(0, x)
# set payoff_matrix:


# In[3]:



# In[4]:



# In[5]:


def normalize_per_row(matrix):
    N, M = matrix.shape
    matrix[:] = relu(matrix)
    for i in range(N):
        total = np.sum(matrix[i])
        if total== 0:
            matirx[i][0] = 1
        else:
            matrix[i] = matrix[i]/total


# In[6]:


global strategies_matrix 
#strategies_matrix = np.ones((N_agents, N_actions))/N_actions
strategies_matrix = np.random.random_sample((N_agents, N_actions)) 
normalize_per_row(strategies_matrix)
# In[7]:


def project_grad(g):
  return g - g.sum() / g.size


# In[8]:


histories_matrix = np.zeros_like(strategies_matrix)

grad_histories =  np.zeros_like(histories_matrix)

grad_strategies = np.zeros_like(strategies_matrix)

other_player_fx = np.zeros_like(strategies_matrix)

reg_exp = np.zeros(N_agents)

exp_thresh = 0.01

lr_strategy = 0.0001

lr_history = 0.01

anneal_steps = 0


from scipy import special


# In[15]:


# take an agent's action: np.random.choice(N_actions, p=strategies_matrix[agent])


temp = 1
def single_iter(t,strategies_matrix,histories_matrix,grad_histories,grad_strategies,other_player_fx,reg_exp,lr_history,lr_strategy,exp_thresh):
    if t%1000 == 0:
        print(t)
    global anneal_steps
    global temp
    actions = np.array([np.random.choice(N_actions, p=strategies_matrix[agent]) for agent in range(N_agents)])
    for i in range(N_agents):
        nabla_i = np.zeros_like(strategies_matrix[i])
        for j in range(N_agents):
            if i == j:
                continue
                
            actions_with_holes = list(actions.copy()) + [i]
            actions_with_holes[i] = slice(None)
            actions_with_holes[j] = slice(None)
            actions_with_holes = tuple(actions_with_holes)
            hess_i_ij = payoff_matrix[actions_with_holes]
            nabla_ij = hess_i_ij.dot(strategies_matrix[i])
            nabla_i += nabla_ij/ float(N_agents - 1)
        grad_histories[i][:] = histories_matrix[i][:] - nabla_i
        
        if temp >= 1e-3:
            br_i = special.softmax(histories_matrix[i] / temp)
            br_i_mat = (np.diag(br_i) - np.outer(br_i, br_i)) / temp
            br_i_strategies_gradient = nabla_i - temp * (np.log(br_i + 0.00001) + 1)
        else:
            s_i = np.max(histories_matrix[i])
            br_i = np.zeros_like(strategies_matrix[i])
            maxima_i = (histories_matrix[i] == s_i)
            br_i[maxima_i] = 1. / maxima_i.sum()
            br_i_mat = np.zeros((br_i.size, br_i.size))
            br_i_strategies_gradient = np.zeros_like(br_i)

        strategies_gradient_i = nabla_i - temp * (np.log(strategies_matrix[i] + 0.00001) + 1)

        other_player_fx[i][:] = (br_i - strategies_matrix[i]) + br_i_mat.dot(br_i_strategies_gradient)

        entr_br_i = temp * special.entr(br_i).sum()
        entr_strategies_i = temp* special.entr(strategies_matrix[i]).sum()

        reg_exp[i] = histories_matrix[i].dot(br_i - strategies_matrix[i]) + entr_br_i - entr_strategies_i

        grad_strategies[i][:] -= strategies_gradient_i
        for j in range(N_agents):
            if i == j:
                continue
                
            actions_with_holes = list(actions.copy()) + [j]
            actions_with_holes[i] = slice(None)
            actions_with_holes[j] = slice(None)
            actions_with_holes = tuple(actions_with_holes)
            hess_j_ij = payoff_matrix[actions_with_holes]
            action_u = np.random.choice(strategies_matrix[j].size)
            other_player_fx_j = (strategies_matrix[j].size) * other_player_fx[j][action_u]
            grad_strategies[i][:] += hess_j_ij[:,action_u]*other_player_fx_j
            
        grad_strategies[i][:] = project_grad(grad_strategies[i][:])
        if np.isnan(grad_strategies).any():
            print("nanananan!")
    reg_exp_mean = np.mean(reg_exp)
    if (reg_exp_mean < exp_thresh) and (anneal_steps >= 1/lr_history):
        temp = np.clip((temp/(2.)), 0, np.inf)
        print("AHHHHHHHHHHHHHHHHHh")
        d_anneal_steps = -anneal_steps
    else:
        d_anneal_steps = 1

    #update
    lr_h = np.clip(1 / float(t + 1), lr_history, np.inf)
    anneal_steps += d_anneal_steps 
    histories_matrix[:] = histories_matrix[:] - lr_h*grad_histories
    strategies_matrix[:] = np.clip(strategies_matrix, 0, np.inf) - lr_strategy * grad_strategies
    normalize_per_row(strategies_matrix)
    


# In[16]:
N = 45000
W = N_agents * N_actions

a = np.zeros((N, W))

temps = np.zeros(N)
steps = np.zeros(N)

for i in range(N):
    single_iter(i,strategies_matrix,histories_matrix,grad_histories,grad_strategies,other_player_fx,reg_exp,lr_history,lr_strategy,exp_thresh)
    a[i] = strategies_matrix.flatten()
    temps[i] = np.log(temp)
    steps[i] = anneal_steps

print(strategies_matrix)
lines = a.T

x = np.arange(N)
for i in range(W):
    a, b = np.unravel_index(i,(N_agents,N_actions))
    plt.plot(x, lines[i], label = "agent {}, strategy {}".format(a,b)) 

plt.legend() 
plt.show()

plt.plot(x, temps, label = "temps") 
plt.show()
plt.plot(x, steps, label = "steps") 
plt.show()

# In[ ]:





# In[ ]:





# In[ ]:




