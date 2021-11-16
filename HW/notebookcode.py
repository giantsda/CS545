#!/usr/bin/env python
# coding: utf-8

# # A5.1 Reinforcement Learning for Marble with Variable Goal

# For this assignment, start with the `19 Reinforcement Learning Modular Framework` notebook.  Recall that this code used reinforcement learning to learn to push a marble towards the goal position of 5 and keep it there.
# 
# The objective of the following required modification is an agent that has been trained to directly move the marble to a specified goal without any further training. 
# 
# <font color="red">Modify the code</font> to allow any goal position from 1 to 9.  First, rename the `Marble` class to `Marble_Variable_Goal`.  Then, modify the `Marble_Variable_Goal` class so that it includes the goal in the state, allowing the agent to learn to push the marble to any given goal.  Modify the `intial_state` function to set the goal to a random integer from 1 to 9.
# 
# <font color='red'>Do not modify</font> the `Qnet` class. It should run correctly when applied to your new `Marble_Variable_Goal` class.
# 
# <font color='red'>Discuss</font> what you modified in the code for this assignment.

# <font color='red'>Add some code</font> at the end of the notebook that applies the trained agent to the marble at goals from 1 to 9.  For each goal, start the marble at positions 0 through 10 with zero velocity and the specified goal and applies the trained agent to control the marble for 200 steps.  Calculate the distance of the final state from the goal.  Average this distance over all starting positions for the specified goal and store in a numpy array with one row for each goal and each row containing the goal and the average of distances to goal over all starting positions. Call this numpy array, `distances_to_goal`.  Plot the results of these average distances versus the goal.
# 
# <font color='red'>Explore different parameter values</font>, including the network hidden layer structure, number of trials, number of steps per trial, learning rate, number of epochs, and final epsilon value to try to get the best results for `distances_to_goal`. Try just three or four different values for each parameter, varying one parameter value at a time. After you have found some parameter values that often work well, set the parameters to these values and run again to produce the graphs from `plot_status` showing the results with these parameters. But, first <font color='red'>modify `plot_status` code</font> for subplots 6 and 9 so that the vertical pink goal region correctly shows the current goal.  Add the current goal to the title of the subplot 9.
# 
# <font color='red'>Discuss</font> the results, and discuss which parameter values seem to perform well.

# Here is some code and parameter values that I have found to be successful...usually.  As you know, results vary quite a bit from one run to another. Understand that you will not find parameter values that work perfectly every time.  You are welcome to start with these parameter values and experiment with variations of these.

# In[53]:



import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time  # for sleep
import IPython.display as ipd  # for display and clear_output
from IPython.display import display, clear_output  # for the following animation
import os
import copy
import signal
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import optimizers as opt
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import neuralnetworks_A4 as nn   # from A4

from IPython.display import display, clear_output

from abc import ABC, abstractmethod
   
class Environment(ABC):
   
   def __init__(self, valid_actions):
       self.valid_actions = valid_actions

   @abstractmethod
   def initial_state(self):
       return state  # the initial state
   
   @abstractmethod
   def next_state(self, state, action):
       return next_state  
   
   @abstractmethod
   def reinforcement(self, state):
       return r # scalar reinforcement
  
   def terminal_state(self, state):
       return False  # True if state is terminal state




class Agent(ABC):
   
   def __init__(self, environment):
       self.environment = environment

   @abstractmethod
   def make_samples(self, n_samples, epsilon):
       return X, R, Qn, terminal_state

   def update_Qn(self, X, Qn, terminal_state):
       n_samples = X.shape[0]
       for i in range(n_samples - 1):
           if not terminal_state[i+1]:
               Qn[i] = self.use(X[i+1])
       return Qn

   def epsilon_greedy(self, state, epsilon):
       valid_actions = self.environment.valid_actions

       if np.random.uniform() < epsilon:
           # Random Move
           action = np.random.choice(valid_actions)
       else:
           # Greedy Move
           Qs = [self.use(np.hstack((state, a)).reshape((1, -1))) for a in valid_actions]
           ai = np.argmax(Qs)
           action = valid_actions[ai]

       return action

   @abstractmethod
   def train(self):
       return

   @abstractmethod
   def use(self, X):
       return # Q values for each row of X, each consisting of state and action
class Qnet(Agent):

   def __init__(self, environment, hidden_layers, X_means=None, X_stds=None, Q_means=None, Q_stds=None):
       self.environment = environment
       state_size = environment.initial_state().size  # assumes state is an np.array
       valid_actions = environment.valid_actions
       action_size = 1 if valid_actions.ndim == 1 else valid_actions.shape[1]

       self.Qnet = nn.NeuralNetwork(state_size + action_size, hidden_layers, 1)
       if X_means:
           self.Qnet.X_means = np.array(X_means)
           self.Qnet.X_stds = np.array(X_stds)
           self.Qnet.T_means = np.array(Q_means)
           self.Qnet.T_stds = np.array(Q_stds)

   def make_samples(self, n_samples, epsilon):

       state_size = self.environment.initial_state().size  # assumes state is an np.array
       valid_actions = self.environment.valid_actions
       action_size = 1 if valid_actions.ndim == 1 else valid_actions.shape[1]

       X = np.zeros((n_samples, state_size + action_size))
       R = np.zeros((n_samples, 1))
       Qn = np.zeros((n_samples, 1))
       terminal_state = np.zeros((n_samples, 1), dtype=bool)  # All False values

       state = self.environment.initial_state()
       state = self.environment.next_state(state, 0)        # Update state, sn from s and a
       action = self.epsilon_greedy(state, epsilon)

       # Collect data from n_samples steps
       for step in range(n_samples):

           next_state = self.environment.next_state(state, action)        # Update state, sn from s and a
           r = self.environment.reinforcement(state)   # Calculate resulting reinforcement
           next_action = self.epsilon_greedy(next_state, epsilon)
           X[step, :] = np.hstack((state, action))
           R[step, 0] = r
           if self.environment.terminal_state(state):
               terminal_state[step, 0] = True
               Qn[step, 0] = 0
           else:
               Qn[step, 0] = self.use(np.hstack((next_state, next_action)))
           # Advance one time step
           state, action = next_state, next_action

       return X, R, Qn, terminal_state

   def update_Qn(self, X, Qn, terminal_state):
       n_samples = X.shape[0]
       for i in range(n_samples - 1):
           if not terminal_state[i+1]:
               Qn[i] = self.use(X[i+1])
       return Qn

   def train(self, n_trials, n_steps_per_trial, n_epochs, method, learning_rate, 
             gamma, epsilon, final_epsilon,
             trial_callback=None):

       if trial_callback:
           fig = plt.figure(figsize=(10, 10))
           
       epsilon_decay =  np.exp(np.log(final_epsilon) / n_trials) # to produce this final value
       print('epsilon_decay is', epsilon_decay)
       epsilon_trace = np.zeros(n_trials)
       r_trace = np.zeros(n_trials)

       for trial in range(n_trials):

           X, R, Qn, terminal_state = self.make_samples(n_steps_per_trial, epsilon)

           for epoch in range(n_epochs):
               self.Qnet.train(X, R + gamma * Qn, 1,  method=method, learning_rate=learning_rate, batch_size=-1, verbose=False)
               self.update_Qn(X, Qn, terminal_state)

           epsilon *= epsilon_decay

           # Rest is for plotting
           epsilon_trace[trial] = epsilon
           r_trace[trial] = np.mean(R)

           if trial_callback and (trial + 1 == n_trials or trial % (n_trials / 10) == 0):
               print('runing %2.2f' % (trial / (n_trials / 10))*100)
               fig.clf()
               trial_callback(agent, trial, n_trials, X, epsilon_trace, r_trace)
               clear_output(wait=True)
               display(fig)

       if trial_callback:
           clear_output(wait=True)

       return epsilon_trace, r_trace


   def use(self, X):
       return self.Qnet.use(X)
from matplotlib import cm

def plot_status(agent, trial, n_trials, X, epsilon_trace, r_trace):
   a=1

   plt.subplot(3, 3, 6)
   plt.plot(X[:, 0], X[: ,1])
   plt.plot(X[-1, 0], X[-1, 1], 'ro')
   plt.xlabel('$x$')
   plt.ylabel('$\dot{x}$')
   plt.fill_between([4, 6], [-5, -5], [5, 5], color='red', alpha=0.3)  # CHECK OUT THIS FUNCTION!
   plt.xlim(-1, 11)
   plt.ylim(-5, 5)
   plt.title('Last Trial')



   test_it(agent, 10, 500)

   plt.tight_layout()


def test_it(agent, n_trials, n_steps_per_trial):
   xs = np.linspace(0, 10, n_trials)
   plt.subplot(3, 3, 9) 
   
   # For a number (n_trials) of starting positions, run marble sim for n_steps_per_trial
   for x in xs:
       
       s = [x, 0,goal] # 0 velocity
       x_trace = np.zeros((n_steps_per_trial, 3))
       for step in range(n_steps_per_trial):
           a = agent.epsilon_greedy(s, 0.0) # epsilon = 0
           s = agent.environment.next_state(s, a)
           x_trace[step, :] = s
           
       plt.plot(x_trace[:, 0], x_trace[:, 1])
       plt.plot(x_trace[-1, 0], x_trace[-1, 1], 'ro')
       plt.fill_between([4, 6], [-5, -5], [5, 5], color='pink', alpha=0.3)
       plt.xlim(-1, 11)
       plt.ylim(-5, 5)
       plt.ylabel('$\dot{x}$')
       plt.xlabel('$x$')
       plt.title('State Trajectories for $\epsilon=0$')
       
       


class Marble_Variable_Goal(Environment):

   def __init__(self, valid_actions):
       super().__init__(valid_actions)
       self.goal = None
       
       
   def initial_state(self):
       goal=np.random.randint(1,10)
       self.goal =goal
       return np.array([10 * np.random.uniform(), 0.0, goal])

   def next_state(self, state, action):
       '''[0] is position, s[1] is velocity. a is -1, 0 or 1'''    
       next_state = state.copy()
       deltaT = 0.1                           # Euler integration time step
       next_state[0] += deltaT * state[1]                  # Update position
       force = action
       mass = 0.5
       next_state[1] += deltaT * (force / mass - 0.2 * state[1])  # Update velocity. Includes friction
       
       next_state[2]=self.goal 
       # Bound next position. If at limits, set velocity to 0.
       if next_state[0] < 0:        
           next_state = [0., 0.,self.goal ]    # these constants as ints were causing the errors we discussed in class. I DON'T KNOW WHY!!
       elif next_state[0] > 10:
           next_state = [10., 0.,self.goal ]

       return next_state

   def reinforcement(self, state):
       goal = self.goal
       return 0 if abs(state[0]- goal) < 1 else -1

   def terminal_state(self, state):
       return False
   def changeGoal(self,newgoal):
       self.goal=newgoal
       
       
       
       
def distances_to_goal(agent,goal):
   n_steps_per_trial=200
   xs = np.linspace(0, 10, 25)
   d=[]
   for x in xs:
       s = [x, 0, goal]  # 0 velocity
       x_trace = np.zeros((n_steps_per_trial, 3))
       for step in range(n_steps_per_trial):
           a = agent.epsilon_greedy(s, 0.0) # epsilon = 0
           s = agent.environment.next_state(s, a)
           x_trace[step, :] = s
       lastPostion=x_trace[-1,0]
       d.append(abs(lastPostion-goal))
       print(f'distance={abs(lastPostion-goal)}\n')
   return sum(d)/len(d)  # return average distance


# In[3]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))

agent = Qnet(marble, hidden_layers=[10, 10],
             X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
             Q_means=[-2], Q_stds=[1])

epsilon_trace, r_trace =  agent.train(n_trials=50, n_steps_per_trial=20, n_epochs=10,
                                      method='sgd', learning_rate=0.01, gamma=0.9,
                                      epsilon=1, final_epsilon=0.01,
                                      trial_callback=plot_status)


# In[55]:


# running way too long

marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
 
goals=[]
distances=[]
for goal in range(1,10):
    marble.changeGoal(goal)
    agent = Qnet(marble, hidden_layers=[10, 10],
                 X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
                 Q_means=[-2], Q_stds=[1])
 
    epsilon_trace, r_trace =  agent.train(n_trials=300, n_steps_per_trial=200, n_epochs=100,
                                          method='sgd', learning_rate=0.01, gamma=0.9,
                                          epsilon=1, final_epsilon=0.01,
                                          trial_callback=plot_status)
    d=distances_to_goal(agent,goal)
    goals.append(goal)
    distances.append(d)

    print(f'goal={goal}; distance={d}\n')
plt.plot(goals,distances,'o-')
plt.show()


# In[41]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
goal=5
n_trialsS=(30, 100, 150)
ds=[]
for n_trials in n_trialsS:  
    marble.changeGoal(goal)
    agent = Qnet(marble, hidden_layers=[10, 10],
                 X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
                 Q_means=[-2], Q_stds=[1])

    epsilon_trace, r_trace =  agent.train(n_trials=n_trials, n_steps_per_trial=100, n_epochs=150,
                                          method='sgd', learning_rate=0.01, gamma=0.9,
                                          epsilon=1, final_epsilon=0.01,
                                          trial_callback=plot_status)
    d=distances_to_goal(agent,goal)
    ds.append(d)

for i in range(0,3):
    print(f'n_trials={n_trialsS[i]}; distance={ds[i]}\n')


# In[44]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
goal=5
hidden_layersS=([0],[3,3],[7,7])
ds=[]
for hidden_layers in hidden_layersS:  
    marble.changeGoal(goal)
    agent = Qnet(marble, hidden_layers=hidden_layers,
                 X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
                 Q_means=[-2], Q_stds=[1])

    epsilon_trace, r_trace =  agent.train(n_trials=150, n_steps_per_trial=200, n_epochs=150,
                                          method='sgd', learning_rate=0.01, gamma=0.9,
                                          epsilon=1, final_epsilon=0.01,
                                          trial_callback=plot_status)
    d=distances_to_goal(agent,goal)
    ds.append(d)

for i in range(0,3):
    print(f'hidden_layers={hidden_layersS[i]}; distance={ds[i]}\n')


# In[45]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
goal=5
n_steps_per_trialS=(10, 50, 100)
ds=[]
for n_steps_per_trial in n_steps_per_trialS:  
    marble.changeGoal(goal)
    agent = Qnet(marble, hidden_layers=[3, 3],
                 X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
                 Q_means=[-2], Q_stds=[1])

    epsilon_trace, r_trace =  agent.train(n_trials=150, n_steps_per_trial=n_steps_per_trial, n_epochs=150,
                                          method='sgd', learning_rate=0.01, gamma=0.9,
                                          epsilon=1, final_epsilon=0.01,
                                          trial_callback=plot_status)
    d=distances_to_goal(agent,goal)
    ds.append(d)

for i in range(0,3):
    print(f'n_steps_per_trial={n_steps_per_trialS[i]}; distance={ds[i]}\n')


# In[48]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
goal=5
learning_rateS=(0.01, 0.02, 0.03)
ds=[]
for learning_rate in learning_rateS:  
    marble.changeGoal(goal)
    agent = Qnet(marble, hidden_layers=[3, 3],
                 X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
                 Q_means=[-2], Q_stds=[1])

    epsilon_trace, r_trace =  agent.train(n_trials=150, n_steps_per_trial=50, n_epochs=50,
                                          method='sgd', learning_rate=learning_rate, gamma=0.9,
                                          epsilon=1, final_epsilon=0.01,
                                          trial_callback=plot_status)
    d=distances_to_goal(agent,goal)
    ds.append(d)

for i in range(0,3):
    print(f'learning_rate={learning_rateS[i]}; distance={ds[i]}\n')


# In[47]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
goal=5
n_epochsS=(50,100,150)
ds=[]
for n_epochs in n_epochsS:  
    marble.changeGoal(goal)
    agent = Qnet(marble, hidden_layers=[3, 3],
                 X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
                 Q_means=[-2], Q_stds=[1])

    epsilon_trace, r_trace =  agent.train(n_trials=150, n_steps_per_trial=100, n_epochs=n_epochs,
                                          method='sgd', learning_rate=learning_rate, gamma=0.9,
                                          epsilon=1, final_epsilon=0.01,
                                          trial_callback=plot_status)
    d=distances_to_goal(agent,goal)
    ds.append(d)

for i in range(0,3):
    print(f'n_epochs={epsilonS[i]}; distance={ds[i]}\n')


# In[50]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
goal=5
final_epsilonS=(0.001,0.01,0.02)
ds=[]
for final_epsilon in final_epsilonS:  
    marble.changeGoal(goal)
    agent = Qnet(marble, hidden_layers=[3, 3],
                 X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
                 Q_means=[-2], Q_stds=[1])

    epsilon_trace, r_trace =  agent.train(n_trials=150, n_steps_per_trial=100, n_epochs=150,
                                          method='sgd', learning_rate=learning_rate, gamma=0.9,
                                          epsilon=1, final_epsilon=final_epsilon,
                                          trial_callback=plot_status)
    d=distances_to_goal(agent,goal)
    ds.append(d)

for i in range(0,3):
    print(f'epsilon={epsilonS[i]}; distance={ds[i]}\n')


# In[54]:


marble = Marble_Variable_Goal(valid_actions=np.array([-1, 0, 1]))
goal=5
 
ds=[]
 
marble.changeGoal(goal)
agent = Qnet(marble, hidden_layers=[3, 3],
             X_means=[5, 0, 5, 0], X_stds=[2, 2, 2, 0.8],
             Q_means=[-2], Q_stds=[1])

epsilon_trace, r_trace =  agent.train(n_trials=150, n_steps_per_trial=100, n_epochs=200,
                                      method='sgd', learning_rate=learning_rate, gamma=0.9,
                                      epsilon=1, final_epsilon=0.01,
                                      trial_callback=plot_status)
d=distances_to_goal(agent,goal)
ds.append(d)


# The optimal network hidden layer structure appears to be: [3,3]. In this problem, it looks like the linear model does not generate a good R and a good play strategy
# The optimal number of n_trialsS is 150. In this problem, the larger the n_trialsS, the better result we can get.
# The optimal number of n_steps_per_trial is 50. The conclusion is that the best n_steps_per_trial does not to be the largest. However, it must be noteted that the initial weights are randomly generated. Therefore, running the same code multiple times may give different results.
# The optimal n_epochs is 150. This tells us that at iteration 150, the error is still desending. However, the code ran for too long so I choosed 150. increase n_epochs may give a better result.
# The optimal learning rate is 0.02. When the learning rate  is too larger, the system is nnstable. when the learning rate  is too small, the converge rate can be imporved when a larger learning rate is given.
# The optimal final epsilon value is 0.01. Too small is not good, neither as too large.
#  

# # Grading
# 
# Download [A5grader.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/A5grader.tar) and extract `A5grader.py` before running the following cell

# In[ ]:


get_ipython().run_line_magic('run', '-i A5grader.py')


# ## Extra Credit
# 
# Receive 1 point of extra credit for each of these:
# 
#    * Modify your solution to this assignment by creating and using a `Marble2D` class that simulates the marble moving in two-dimensions, on a plane.  Some of the current plots will not work for this case. Just show the ones that are still appropriate.
#    * Experiment with seven valid actions rather than three.  How does this change the behavior of the controlled marble?
