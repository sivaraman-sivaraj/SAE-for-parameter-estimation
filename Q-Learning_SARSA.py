import numpy as np
import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
     if 'puddle' in env:
          print('Remove {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]
import puddle_world
import matplotlib.pyplot as plt 

method = 1 # 0:Q-Learning, 1:SARSA, 2:SARSA-Lambda
e = 0.1 # epsilon
g = 0.9 # gamma
episodes = 500
alpha = 0.2
experiments = 1
limit = 300000
lamda = 0.3

def pick_action(possibles, epsilon):
    equals = all(ele == possibles[0] for ele in possibles)
    if(equals):
        act = np.random.randint(0,4)
    else:
        act = np.argmax(possibles)
    if(np.random.rand() >= (1-epsilon)):
        act = np.random.randint(0,4)
    return act

# Goal A, Wind is on
env = gym.make('puddle-v0')

steps = np.zeros((experiments, episodes))
rewards = np.zeros((experiments, episodes))
Q = np.zeros((12,12,4))

for run in range(experiments):
    q = np.zeros((12,12,4))
    if(method==2):
        el = np.zeros((12,12,4))
    for ep in range(episodes):
        env.reset()
        c = env.current_state
        an = pick_action(q[c[0]][c[1]], e) # next action initialization
        while(not(env.done)):
            c = env.current_state # current state
            #a = pick_action(q[c[0]][c[1]], e)
            a = an
            
            n, r = env.step(a) # next state, reward
            steps[run][ep] += 1
            rewards[run][ep] += r
            
            if(method==0):
                q[c[0]][c[1]][a] += alpha*(r + g*np.max(q[n[0]][n[1]]) - q[c[0]][c[1]][a])
                an = pick_action(q[n[0]][n[1]], e)
            
            elif(method==1):
                an = pick_action(q[n[0]][n[1]], e)
                q[c[0]][c[1]][a] += alpha*(r + g*q[n[0]][n[1]][an] - q[c[0]][c[1]][a])
            
            elif(method==2):
                an = pick_action(q[n[0]][n[1]], e)
                delta = r + g*q[n[0]][n[1]][an] - q[c[0]][c[1]][a]
                for k in range(4):
                    if(not(a==k)):
                        el[0:12, 0:12, k] = np.zeros((12,12))
                el[0:12, 0:12, a] *= g*lamda
                el[c[0]][c[1]][a] += 1
                q += alpha*delta*el
            
            if(steps[run][ep] >= limit):
                break
        
        rewards[run][ep] /= steps[run][ep]
    print(run)
    Q += q

avg_steps1 = np.mean(steps, axis=0)
avg_rewards1 = np.mean(rewards, axis=0)
Q /= experiments

actions1 = np.array(np.zeros((12,12)), dtype=str)
for i in range(12):
    for j in range(12):
        a = np.argmax(q[i][j])
        if(a==0):
            actions1[i][j] = '\u2191'
        elif(a==1):
            actions1[i][j] = '\u2193'
        elif(a==2):
            actions1[i][j] = '\u2190'
        elif(a==3):
            actions1[i][j] = '\u2192'
            
# Goal B, Wind is on
env = gym.make('puddle-v0', wind=True, end=1)

steps = np.zeros((experiments, episodes))
rewards = np.zeros((experiments, episodes))
Q = np.zeros((12,12,4))

for run in range(experiments):
    q = np.zeros((12,12,4))
    if(method==2):
        el = np.zeros((12,12,4))
    for ep in range(episodes):
        env.reset()
        c = env.current_state
        an = pick_action(q[c[0]][c[1]], e) # next action initialization
        while(not(env.done)):
            c = env.current_state # current state
            #a = pick_action(q[c[0]][c[1]], e)
            a = an
            
            n, r = env.step(a) # next state, reward
            steps[run][ep] += 1
            rewards[run][ep] += r
            
            if(method==0):
                q[c[0]][c[1]][a] += alpha*(r + g*np.max(q[n[0]][n[1]]) - q[c[0]][c[1]][a])
                an = pick_action(q[n[0]][n[1]], e)
            
            elif(method==1):
                an = pick_action(q[n[0]][n[1]], e)
                q[c[0]][c[1]][a] += alpha*(r + g*q[n[0]][n[1]][an] - q[c[0]][c[1]][a])
            
            elif(method==2):
                an = pick_action(q[n[0]][n[1]], e)
                delta = r + g*q[n[0]][n[1]][an] - q[c[0]][c[1]][a]
                for k in range(4):
                    if(not(a==k)):
                        el[0:12, 0:12, k] = np.zeros((12,12))
                el[0:12, 0:12, a] *= g*lamda
                el[c[0]][c[1]][a] += 1
                q += alpha*delta*el
            
            if(steps[run][ep] >= limit):
                break
            
        rewards[run][ep] /= steps[run][ep]
    print(run)
    Q += q

avg_steps2 = np.mean(steps, axis=0)
avg_rewards2 = np.mean(rewards, axis=0)
Q /= experiments

actions2 = np.array(np.zeros((12,12)), dtype=str)
for i in range(12):
    for j in range(12):
        a = np.argmax(q[i][j])
        if(a==0):
            actions2[i][j] = '\u2191'
        elif(a==1):
            actions2[i][j] = '\u2193'
        elif(a==2):
            actions2[i][j] = '\u2190'
        elif(a==3):
            actions2[i][j] = '\u2192'

# Goal C, Wind is off
env = gym.make('puddle-v0', wind=False, end=2)

steps = np.zeros((experiments, episodes))
rewards = np.zeros((experiments, episodes))
Q = np.zeros((12,12,4))

for run in range(experiments):
    q = np.zeros((12,12,4))
    if(method==2):
        el = np.zeros((12,12,4))
    for ep in range(episodes):
        env.reset()
        c = env.current_state
        an = pick_action(q[c[0]][c[1]], e) # next action initialization
        while(not(env.done)):
            c = env.current_state # current state
            #a = pick_action(q[c[0]][c[1]], e)
            a = an
            
            n, r = env.step(a) # next state, reward
            steps[run][ep] += 1
            rewards[run][ep] += r
            
            if(method==0):
                q[c[0]][c[1]][a] += alpha*(r + g*np.max(q[n[0]][n[1]]) - q[c[0]][c[1]][a])
                an = pick_action(q[n[0]][n[1]], e)
            
            elif(method==1):
                an = pick_action(q[n[0]][n[1]], e)
                q[c[0]][c[1]][a] += alpha*(r + g*q[n[0]][n[1]][an] - q[c[0]][c[1]][a])
            
            elif(method==2):
                an = pick_action(q[n[0]][n[1]], e)
                delta = r + g*q[n[0]][n[1]][an] - q[c[0]][c[1]][a]
                for k in range(4):
                    if(not(a==k)):
                        el[0:12, 0:12, k] = np.zeros((12,12))
                el[0:12, 0:12, a] *= g*lamda
                el[c[0]][c[1]][a] += 1
                q += alpha*delta*el
            
            if(steps[run][ep] >= limit):
                break
            
        rewards[run][ep] /= steps[run][ep]
    print(run)
    Q += q

avg_steps3 = np.mean(steps, axis=0)
avg_rewards3 = np.mean(rewards, axis=0)
Q /= experiments

actions3 = np.array(np.zeros((12,12)), dtype=str)
for i in range(12):
    for j in range(12):
        a = np.argmax(q[i][j])
        if(a==0):
            actions3[i][j] = '\u2191'
        elif(a==1):
            actions3[i][j] = '\u2193'
        elif(a==2):
            actions3[i][j] = '\u2190'
        elif(a==3):
            actions3[i][j] = '\u2192'

# PLotting
x = np.arange(1, episodes+1)

plt.plot(x, avg_rewards1, label = 'Goal A')
plt.plot(x, avg_rewards2, label = 'Goal B')
plt.plot(x, avg_rewards3, label = 'Goal C')
plt.axes()
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Averge Reward')
plt.title('The plot of Average Rewards vs Episodes averaged over 5 runs')
plt.tight_layout()
if(method==0):
    plt.savefig('QLrng_rewards_2.pdf', dpi=1200)
elif(method==1):
    plt.savefig('SARSA_rewards_2.pdf', dpi=1200)
elif(method==2):
    plt.savefig('SARSA-Lambda_rewards_2.pdf', dpi=1200)
plt.close()

plt.plot(x, avg_steps1, label = 'Goal A')
plt.plot(x, avg_steps2, label = 'Goal B')
plt.plot(x, avg_steps3, label = 'Goal C')
plt.axes()
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Steps')
plt.title('The plot of Number of Steps to Goal vs Episodes averaged over 5 runs')
plt.tight_layout()
if(method==0):
    plt.savefig('QLrng_steps_2.pdf', dpi=1200)
elif(method==1):
    plt.savefig('SARSA_steps_2.pdf', dpi=1200)
elif(method==2):
    plt.savefig('SARSA-Lambda_steps_2.pdf', dpi=1200)
plt.close()
