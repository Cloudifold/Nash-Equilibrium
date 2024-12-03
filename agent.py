import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from scipy.special import softmax

#parameters

N_agents = 2
N_actions = 3
payoff_matrix_shape = tuple([N_actions for i in range(N_agents)] + [N_agents])
print(payoff_matrix_shape)

def relu(x):
    return np.maximum(0, x)


# set payoff_matrix:
payoff_matrix = np.random.normal(1, 4.5, size=payoff_matrix_shape)
payoff_matrix = np.reshape(np.array([ 3,  8,  4,  0,  5,  3,  1, -3,  9,  4,  8,  5,  4,  7,  2,  6,  2, 9]),(3,3,2))
payoff_matrix = payoff_matrix.transpose(1, 0, 2)


def utility_of_action(agent, actions):
    return payoff_matrix[tuple(actions)][agent]


def entropy(temp, agent, strategies):
    return -temp*np.dot(strategies[agent],np.log(strategies[agent]))

print(payoff_matrix)
learning_rate = 0.07

batch_size = 10

# Draw the heat map of the payoff of the first playerin the case N_agents = 2
'''
fig, (ax1, ax2) = plt.subplots(1,2)
sns.heatmap(payoff_matrix[:,:,0], linewidth=0.5, ax=ax1, square=True)
sns.heatmap(payoff_matrix[:,:,1], linewidth=0.5, ax=ax2, square=True)
ax1.tick_params(labeltop=True, top=True, bottom=False, labelbottom=False)
ax2.tick_params(labeltop=True, top=True, bottom=False, labelbottom=False)
plt.show()
'''


def normalize_per_row(matrix):
    N, M = matrix.shape
    matrix[:] = relu(matrix)
    for i in range(N):
        total = np.sum(matrix[i])
        if total== 0:
            matirx[i][0] = 1
        else:
            matrix[i] = matrix[i]/total

global mixed_strategies_matrix
# set the mixed strategies matrix: rows: agents, colums: strategies, each entry: agent i's probability to use strategy j
mixed_strategies_matrix = np.random.random_sample((N_agents, N_actions)) 
normalize_per_row(mixed_strategies_matrix)
print(mixed_strategies_matrix)



def single_batch(payoff_matrix,batch_size = 10):
    choices = np.zeros((batch_size, N_agents), dtype=int)
    temp_choices = np.zeros((N_actions, N_agents), dtype=int)
    strategy_update = np.zeros((N_agents, N_actions))
    payoff_gained_per_agent = np.zeros(N_agents)

    for b in range(batch_size):
        for i in range(N_agents):
            choices[b][i] = np.random.choice(N_actions, p=mixed_strategies_matrix[i])
        # evaluate:
        payoff_gained_per_agent += payoff_matrix[tuple(choices[b])]

        # record:
        for i in range(N_agents):
            temp_choices = np.tile(choices[b],(N_actions, 1))
            for j in range(N_actions):
                temp_choices[j][i] = j
            # eval strategies

            jop = np.argmax(np.array([payoff_matrix[tuple(temp_choices[j])][i] for j in range(N_actions)]))


            strategy_update[i][jop] += 1

    # update:
    mixed_strategies_matrix[:] = mixed_strategies_matrix + learning_rate * strategy_update
    normalize_per_row(mixed_strategies_matrix)

    return payoff_gained_per_agent 




def iter_game(payoff_matrix, batch_size, iteration = 1000):
    payoff_gained_each_iter = np.zeros((iteration, N_agents), dtype=int)

    for i in range(iteration):
        payoff_gained_each_iter[i] = single_batch(payoff_matrix, batch_size)
        

    lines = np.transpose(payoff_gained_each_iter)
    x = np.arange(iteration)
    for i in range(N_agents):
        plt.plot(x, lines[i], label = "agent {}".format(i)) 

    plt.legend() 
    plt.show()

iter_game(payoff_matrix, batch_size, iteration=400)
print(mixed_strategies_matrix)


iter_game(payoff_matrix, batch_size, iteration=70000)
print(mixed_strategies_matrix)







