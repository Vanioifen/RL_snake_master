from mujoco_py import load_model_from_path , MjSim , MjViewer
import math
import os
import numpy as np
from numpy import matrix
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import copy

import gym




gym.envs.register(
    id='MujocoEnv13-v0',
    entry_point='mujoco_env13:MujocoEnv13',  # 模块名.类名
)



def system(u,a,b,uu0,uu1,wij,TT,tt):


    du = np.zeros(20)
    g = lambda x: np.maximum(0, x)


    for i in range(5):
        xx1 = 0
        xx2 = 0
        for j in range(5):
            xx1 += wij[i][j] * u[j*4 + 2]
            xx2 += wij[i][j] * u[j*4 + 3]


        du[i*4 + 0] = (uu0[i] - u[i*4 + 0] - b * u[i*4 + 2] - a * g(u[i*4 + 1]) - xx1) * TT
        du[i*4 + 1] = (uu1[i] - u[i*4 + 1] - b * u[i*4 + 3] - a * g(u[i*4 + 0]) - xx2) * TT
        du[i*4 + 2] = (- u[i*4 + 2] +  g(u[i*4 + 0]) ) * TT * tt
        du[i*4 + 3] = (- u[i*4 + 3] +  g(u[i*4 + 1]) ) * TT * tt

    return du





# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        #self.dropout = torch.nn.Dropout(p=0.3)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 256)
        self.mean_layer = torch.nn.Linear(256, action_dim)
        self.log_std = torch.nn.Parameter(torch.ones(1,action_dim) * - 1.0)
        #self.log_std = torch.nn.Parameter(torch.zeros(1, action_dim)) 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                init.constant_(m.bias, 0)
          
    def forward(self, x):

        x = torch.tanh(self.fc1(x))      
        
        x = torch.tanh(self.fc2(x)) 
        #x = self.dropout(x)
        x = torch.tanh(self.fc3(x)) 
        
        #x = F.relu(self.fc4(x))
        
        mean = torch.tanh(self.mean_layer(x))

        return mean
    
    def get_dist(self, x):
        mean = self.forward(x)
        log_std = self.log_std#.expand_as(mean)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std) 
        return dist,mean


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim , 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        #self.fc4 = torch.nn.Linear(256, 256)
        
        self.fc_out = torch.nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias, 0)
        

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return self.fc_out(x)


# 实现DDPG算法
class PPO(object):
    def __init__(self, state_dim, action_dim, action_max, action_min, expand_max,expand_min,a_lr,c_lr,lmbda, epochs, epss, gammmma, device,entropy_coef):
        
        self.actor = Actor(state_dim, action_dim).to(device)
    
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=a_lr, weight_decay = 1e-4, eps = 1e-5) #$$$$

        self.critic = Critic(state_dim, action_dim).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=c_lr,weight_decay = 1e-4, eps = 1e-5) #$$$$
        
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma = 0.5)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma = 0.5)
        
        self.action_dim = action_dim
        self.action_max = action_max
        self.action_min = action_min
        self.action_dim = action_dim
        self.expand_max = expand_max
        self.expand_min = expand_min
        
        self.gamma = gammmma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = epss
        
        self.steps = 0

        self.device = device
        self.train_times = 0
        
        self.noise_factor = 0.0
        
        self.epsilon = 0.9
        
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)

        #actionn = self.actor(state).cpu().data.numpy().flatten()
        with torch.no_grad():
            dist,mean = self.actor.get_dist(state)
            action = dist.sample()  # Sample the action according to the probability distribution
            action = torch.clamp(action, expand_min[0], expand_max[0])  # [-max,max]
            a_logprob = dist.log_prob(action)  # The log probability density of the action
        

        return action.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten(),mean
        
    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)
        
        
        
    def update(self, replay_buffer, iterations, batch_size):  # discount = gamma
        re_av = 0
        self.train_times += 1
        self.steps += 1
        acl = 0
        crl = 0

            
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_a_logprobs, batch_dws = replay_buffer.sample(batch_size)
        #re_av +=  batch_rewards.item()

        state = torch.FloatTensor(batch_states).to(device) # n* 1
        next_state = torch.FloatTensor(batch_next_states).to(device)
        action = torch.FloatTensor(batch_actions).to(device)
        reward = torch.FloatTensor(batch_rewards).view(-1, 1).to(device)
        done = torch.FloatTensor(batch_dones).view(-1, 1).to(device)
        a_log_prob = torch.FloatTensor(batch_a_logprobs).to(device)
        dw = torch.FloatTensor(batch_dws).view(-1, 1).to(device)
        
        '''
        size1 = state.shape
        size2 = next_state.shape
        size3 = action.shape
        size4 = reward.shape
        size5 = done.shape
        size6 = a_log_prob.shape
        size7 = dw.shape
        
        print("######")
        print("state: ", size1)

        print("next_state: ", size2)

        print("action: ", size3)

        print("reward: ", size4)

        print("done: ", size5)

        print("a_log_prob: ", size6)

        print("dw: ", size7)

        '''
        
        adv = []
        gae = 0
        reward =  (reward + 0) / 8
        with torch.no_grad():
            vs = self.critic(state)
            vs_ = self.critic(next_state)
            deltas = reward + self.gamma * (1.0 - dw) * vs_ - vs
            '''
            size8 = vs.shape
            size9 = vs_.shape
            size10 = deltas.shape
            
            print("vs: ",size8)
            print("vs_: ",size9)
            print("deltas: ",size10)
            '''
            
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lmbda * gae * (1.0 - d)
                adv.insert(0,gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(device)
            
            
            v_target = adv + vs
            
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5)) # 归一化
        

        for _ in range(self.epochs):
            
            '''
            
            dist_now,mmmean = self.actor.get_dist(state)

            dist_entropy = dist_now.entropy()#.unsqueeze(1)#sum(1,keepdim=True)

            a_log_prob_now = dist_now.log_prob(action)
            #ratio = torch.exp(log_probs - old_log_probs)

            log_diff = a_log_prob_now - a_log_prob

            log_diff_clipped = torch.clamp(log_diff, max=80)  # 假设20为上限阈值
            ratio = torch.exp(log_diff_clipped)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
            #print(-torch.min(surr1, surr2))
            #print(dist_entropy)
            #print("#######$$$$$$$$$$$$$")
            actor_loss = (-torch.min(surr1, surr2)).mean() - self.entropy_coef * dist_entropy.mean()
            #print(dist_entropy)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            #print(self.critic(state[i]))
            #print(v_target[i])
            v_s = self.critic(state)
            #print(xc)
            #print(yc)

            critic_loss = F.mse_loss(v_target, v_s)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()

            acl = actor_loss + acl
            crl = critic_loss + crl
            
            
            '''
            for i in range(len(batch_states)):
                
                
                dist_now,mmmean = self.actor.get_dist(state[i])
                
                
                dist_entropy = dist_now.entropy()#.unsqueeze(1)#sum(1,keepdim=True)
    
                a_log_prob_now = dist_now.log_prob(action[i])
                #ratio = torch.exp(log_probs - old_log_probs)


                
                log_diff = a_log_prob_now - a_log_prob[i]
                

                
                log_diff_clipped = torch.clamp(log_diff, max=80)  # 假设20为上限阈值
                ratio = torch.exp(log_diff_clipped)
                


                surr1 = ratio * adv[i]
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv[i]
                

                #print(dist_entropy)
                #print("#######$$$$$$$$$$$$$")
                actor_loss = (-torch.min(surr1, surr2)).mean() - self.entropy_coef * dist_entropy.mean()
                #print(dist_entropy)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                
                #print(self.critic(state[i]))
                #print(v_target[i])
                v_s = self.critic(state[i])
                #print(xc)
                #print(yc)
                
                critic_loss = F.mse_loss(v_target[i], v_s)
            
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
                
                
                
                acl = actor_loss + acl
                crl = critic_loss + crl


        

        return acl/self.epochs ,crl/self.epochs

        #print(f"actor_loss: {acl/iterations:.3f}, critic_loss: {crl/iterations:.3f}")     
                
    def save(self, timestep): 
        
        torch.save(self.actor.state_dict(), "./model14/ddpg_actor{}.pth".format(timestep))
        torch.save(self.critic.state_dict(), "./model14/ddpg_critic{}.pth".format(timestep))

    def load(self, timestep):
        
        self.actor.load_state_dict(torch.load("./model14/ddpg_actor{}.pth".format(timestep)))
        self.critic.load_state_dict(torch.load("./model14/ddpg_critic{}.pth".format(timestep)))



# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建环境

#path = '/home/wyf/rl_snake/t14_onlystate/snake14.xml'

#path = '/home/wyf/rl_snake/t18_vel/snake14.xml'
path = '/home/wyf/rl_snake/t28/snake15_test.xml'
#gym.envs.registration.unregister(id='MujocoEnv-v0')
#gym.register(id = 'MujocoEnv1-v0', entry_point= 'mujoco_env1:MujocoEnv1')
env = gym.make('MujocoEnv13-v0',model_path = path ,seed = None)


state_dim = env.observation_space.shape[0] * 10
action_dim = env.action_space.shape[0] # 1

#azz = 0.3
#action_max =  np.array([ azz, azz, azz])
#action_min =  np.array([-azz,-azz,-azz]) 

azz = 1.0
aazz = -1.0
action_max =  np.array([azz,azz,azz,azz,azz,azz])
action_min =  np.array([aazz,aazz,aazz,aazz,aazz,aazz])

action_cha = (action_max - action_min) 

expand_max  = action_max + action_cha * 0.0
expand_min = action_min  - action_cha * 0.0

a_lr = 2e-4
c_lr = 2e-4
gamma = 0.99
lmbda = 0.95
epochs = 10
eps = 0.2
entropy_coef = 0.001

ppo = PPO(state_dim, action_dim, action_max, action_min, expand_max,expand_min,a_lr,c_lr,lmbda, epochs, eps, gamma, device,entropy_coef)
ppo.load(75)

def normalize(state):
    
    n_state = np.zeros(10)
    
    for i in range(10):
        if i==0 or i==1:
            n_state[i] = state[i] * 5
        elif  i == 2:
            n_state[i] = state[i] / 3
            
        else:
            n_state[i] = state[i] / 2
            
    return n_state

#env.change_target(target_pos)
old_state = env.get_now_state()
next_state = old_state
old_pos = env.get_now_pos()
next_pos = env.get_now_pos()
old_all_state = np.zeros(100)
next_all_state = np.zeros(100)

action = np.zeros(5)
sim = env.get_sim()

tt = 0

#m1 = sim.model.geom_name2id("target_marker")
#sim.model.geom_pos[m1] = m_tp

viewer= MjViewer(sim)

env.reset()

all_turns = 0.0

vic_turns = 0.0

all_target = [[1.0,-1.0],[1.5,1.0],[3.0,1.0],[4.0,2.0],[5.0,0.0]]

arrive = False

yy = []

for iiii in range(len(all_target)):
    
    all_turns += 1
    tt = 0
    
    this_target = all_target[iiii]
    
    #target_r = np.random.uniform(1.5, 2.5, 1)[0]
    #target_ug1 = np.random.uniform(-60, 60, 1)[0]

    #target_ug2 = np.radians(target_ug1)

    #target_x = target_r * math.cos(target_ug2)
    #target_y = target_r * math.sin(target_ug2)


    #target_pos = [target_x, target_y]
    #target_pos = np.array(target_pos)
    env.change_target(np.array(this_target))
    print("$$$$$$$$$$$$$$$$$$")
    print(f"Target:{this_target}  all_turns: {iiii+1}")
    
    while True:
        
        if arrive == True and tt % 100 == 0:
            arrive = False
            break


        if tt >= 500 and tt % 100 == 0 :
            old_all_state = next_all_state
            action,a_log_prob,mean = ppo.select_action(old_all_state) # Output : u1,u2,u3,f,ww,psi,r 7 
            action = mean
            #print(f"T = {int(tt/ 100)}, action= {action}")

        if tt % 10 == 0:
            tr = int( (tt % 100) / 10 )
            next_state = env.get_now_state()
            next_all_state[tr:tr+10] = normalize(next_state)
            old_state = next_state
            old_pos = next_pos

        next_state, done, next_pos, dw, _ = env.step(action)

        #env.change_target(target_pos)

        viewer.render()


        tt += 1
        
        
        if next_state[2] < 0.3:
            vic_turns += 1
            arrive = True
            print(f"Target:{this_target} Vic: {True} ")
        


