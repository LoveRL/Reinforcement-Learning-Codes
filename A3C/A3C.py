import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import *
from shared_adam import SharedAdam

import gym, math, os


'''
Hyper parameters setting
'''
os.environ["OMP_NUM_THREADS"] = '1'

t_max = 10          # Steps for local A2C (update term)
T_max = 3000        # Possible number of Max episodes
discount_factor = 0.9
max_epi_step = 200  # Max steps per episode

env = gym.make('Pendulum-v0')
StateSize, ActionSize = env.observation_space.shape[0], env.action_space.shape[0]


'''
Advantage Actor-Critic agent
inheriting torch.nn.Module
'''
# A2C inheritent
class A2C(nn.Module) :

    def __init__(self, state_dim, action_dim) :
        super(A2C, self).__init__() # means initialization of super class of A2C, 'torch.nn.Module'
        
        self.state_dim, self.action_dim = state_dim, action_dim

        # A arg. of 'nn.Linear()' means dim of one-sample in mini-batch
        
        # Actor Layer
        self.Actor_Hidden = nn.Linear(self.state_dim, 256)
        self.mu = nn.Linear(256, self.action_dim)
        self.sigma = nn.Linear(256, self.action_dim)
                
        self.policy = torch.distributions.Normal
        
        # Critic Layer
        self.Critic_Hidden = nn.Linear(self.state_dim, 128)
        self.state_value = nn.Linear(128, 1)

        # Initialize weights of all layers above
        Init_Layers([self.Actor_Hidden, self.mu, self.sigma, self.Critic_Hidden, self.state_value])

        return

    
    '''
    Method overriding for feed-forward
    '''
    def forward(self, state) :

        # For continuous action-space, mu and sigma for Normal Dist. are need
        actor_hidden = F.relu(self.Actor_Hidden(state))
        mu = 2 * torch.tanh(self.mu(actor_hidden))
        sigma = F.softplus(self.sigma(actor_hidden)) + 0.001

        critic_hidden = F.relu(self.Critic_Hidden(state))
        state_value = self.state_value(critic_hidden)

        return mu, sigma, state_value
    

    '''
    Policy
    '''
    def action_selection(self, state) :
        
        # Continuous Action space
        self.training = False
        mu, sigma, _ = self.forward(state)

        pi_theta = self.policy(mu.view(1, ).data, sigma.view(1, ).data)
        action = pi_theta.sample().numpy()

        return action


    '''
    Calculate actor-loss + critic-loss = total-loss
    '''
    def loss_function(self, state, action, td_target) :

        self.train()
        
        mu, sigma, state_value = self.forward(state)

        # [Critic Loss] : square of TD-Error 
        advantage_func = td_target - state_value # n_step TD-Error(state-value)
        critic_loss = advantage_func.pow(2)

        # [Actor Loss] : performance function J(theta)
        # in which a gradient of it is 'dtheta(advantage * log_pi_theta(a|s))'
        pi_theta = self.policy(mu, sigma)
        log_pi_theta = pi_theta.log_prob(action)
        
        entropy = 0.5 + 0.5 * math.log(2*math.pi) + torch.log(pi_theta.scale) # Exploration
        actor_loss = -(log_pi_theta * advantage_func.detach() + 0.005 * entropy)

        total_loss = (actor_loss + critic_loss).mean()

        return total_loss


'''
Each process working in one thread i.e. 'Local_model'
'''
class Each_local_worker(mp.Process) :

    def __init__(self, Global_A2C, Optimizer, Global_epi_cnt, Global_epi_reward, Result_queue, name) :
        super(Each_local_worker, self).__init__()
        
        self.name = 'Thread-' + str(name)
        self.Local_A2C = A2C(StateSize, ActionSize)
               
        self.Global_A2C = Global_A2C
        self.Optimizer = Optimizer
        
        self.Global_epi_cnt = Global_epi_cnt
        self.Global_epi_reward = Global_epi_reward
        self.Result_queue = Result_queue

        self.env = gym.make('Pendulum-v0').unwrapped

        return
    

    '''
    Method overriding
    by calling .start(), this method starts automatically
    '''
    def run(self) :

        total_step = 1
        
        while self.Global_epi_cnt.value < T_max :
            
            state = self.env.reset()
            state_buffer, action_buffer, reward_buffer = [], [], []
            Local_epi_reward = 0.

            for step in range(1, max_epi_step+1) :

                # input data-type must be an Tensor with 2-dim by [None, :]
                # because 'torch.nn' only provide mini-batch type
                action = self.Local_A2C.action_selection(To_tensor(state[None, :]))
                next_state, reward, done, _ = self.env.step(action.clip(-2, 2)) # 확인해보기
                                
                state_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append((reward+8.1)/8.1) # reward normalization(0~1) to reduce variance
                Local_epi_reward += reward
                                
                if step == max_epi_step :
                    done = True

                # Communication between Local_A2C and Global_A2C
                if total_step % t_max == 0 or done :
          
                    Push_and_Pull(self.Optimizer, self.Local_A2C, self.Global_A2C, done, next_state, \
                                  state_buffer, action_buffer, reward_buffer, discount_factor)
                    
                    state_buffer, action_buffer, reward_buffer = [], [], []

                    if done :
                        record(self.Global_epi_cnt, self.Global_epi_reward, Local_epi_reward, self.Result_queue, self.name)
                        break

                state = next_state
                total_step += 1
               
        self.Result_queue.put(None)
        
        return


'''
Main part
'''
if __name__ == '__main__' :
    
    Global_A2C = A2C(StateSize, ActionSize) # Create Global A2C agent
    Global_A2C.share_memory()               # Load the Global A2C agent to shared memory
    
    Optimizer = SharedAdam(Global_A2C.parameters(), lr=0.0002)

    Global_epi_cnt, Global_epi_reward, Result_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue() # Variables used in shared memory

    # Create Local A2C agents (processes) 
    Workers = [Each_local_worker(Global_A2C, Optimizer, Global_epi_cnt, Global_epi_reward, Result_queue, i) for i in range(mp.cpu_count())]

    # Let them begin in each assigned thread
    [thread.start() for thread in Workers]

    # Pop training rewards
    res = []
    while True :
        r = Result_queue.get()
        if r is not None :
            res.append(r)
        else :
            break
        
    # Let them wait until the others end to prevent zombie process
    [thread.join() for thread in Workers]

    # Plotting train result
    Show_Result(res)

    # Test the model
    test_model(Global_A2C, env)
