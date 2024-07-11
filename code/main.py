import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import gym
import copy
import math
from torch.distributions import MultivariateNormal


import mujoco_env13
from PPO import Actor, Critic, PPO
from ReplayBuffer import ReplayBuffer




gym.envs.register(
    id='MujocoEnv13-v0',
    entry_point='mujoco_env13:MujocoEnv13',  # 模块名.类名
)






 
    
def get_reward(old_state,next_state,old_pos,next_pos,target_pos,tt):
    
    ddo = False

    anti = False
    
    reward = 0
    
    v_d = next_state[0]
    v_k = next_state[1]
    
    ugol = next_state[3]

    d2 = next_state[2]
    d1 = old_state[2]



    reward +=  10 * max(v_d,0) -  5 * abs(v_k)
    
    reward += 2 *((v_d ** 2 + v_k ** 2) ** 0.5) 
    
    reward *= math.exp(-d2/ 2) 
    
    
    if abs(ugol) > math.pi / 3  :
        anti = True
    
    return reward,anti,ddo





# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建环境

#path = '/home/wyf/rl_snake/t14_onlystate/snake14.xml'

#path = '/home/wyf/rl_snake/t18_vel/snake14.xml'
path = '/home/wyf/rl_snake/t25_direct/snake15.xml'
#gym.envs.registration.unregister(id='MujocoEnv-v0')
#gym.register(id = 'MujocoEnv1-v0', entry_point= 'mujoco_env1:MujocoEnv1')
env = gym.make('MujocoEnv13-v0',model_path = path ,seed = None)


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



gamma = 0.96
lmbda = 0.95
epochs = 10
eps = 0.2
entropy_coef = 0.005

ppo = PPO(state_dim, action_dim, action_max, action_min, expand_max,expand_min,a_lr,c_lr,lmbda, epochs, eps, gamma, device,entropy_coef)
#ppo.load(7)
# 初始化经验回放池
replay_buffer = ReplayBuffer()

# 训练参数
episodes = 20000  # 总训练回合数
min_size = 1000
batch_size = 256

iterations = 1

target_vel = 0.08
#exploration_noise = 0.1

tc = 100

reward_ave = []

suc_num = []
suc_s = 0

target_pos = np.array([3.0,3.0])

update_turns = 0

ch_t = 0

for episode in range(episodes):
    state = env.reset()
    
    
    if episode % 1 == 0:
        ch_t = 0
        target_r = np.random.uniform(1.5, 2.5, 1)[0]
        target_ug1 = np.random.uniform(-60, 60, 1)[0]
        
        target_ug2 = np.radians(target_ug1)
        
        target_x = target_r * math.cos(target_ug2)
        target_y = target_r * math.sin(target_ug2)
        
        
        target_pos = [target_x, target_y]
        target_pos = np.array(target_pos)
    
    
    env.change_target(target_pos)
    old_state = env.get_now_state()
    next_state = old_state
    old_pos = env.get_now_pos()
    next_pos = env.get_now_pos()
    
    old_all_state = np.zeros(100)
    next_all_state = np.zeros(100)
    
    episode_reward = 0
    reward = 0
    
    action = np.zeros(5) 
    action0 = np.zeros(5) 
    
    done = False
    tt = 0
    
    mmu = 0
    ssigma = 0
    
    fail = 0
    
    ddo = False
    
    done = False
    
    dw = False
    
    dstart = np.array([1e-4,1e-4])
    vend = np.array([1e-4,1e-4])
    
    while not done:
        anti = False
        
        inside = False
        inside_ugol = 0
        
        dw = False
        
        if tt == 5:
            dstart = env.get_real_vvdd()[2:4]
        
        #if min_size <= replay_buffer.get_size():
        if tt >= 5:
            action,a_log_prob,mean = ppo.select_action(old_all_state) # Output : u1,u2,u3,f,ww,psi,r 7 

        for i in range(tc):
            next_state, done, next_pos, dw, _ = env.step(action)
            
            if i % 10 == 0:
                ii = int (i/10)
                normalized_state = normalize(next_state)
                next_all_state[ii:ii+10] = normalized_state
            
            if next_state[2]< 0.2 and old_state[2] >= 0.2:
                inside = True
                inside_ugol = next_state[3]

        
        #print(max(old_state[9:]))
        tarx = 0

        if tt >= 5:
            reward,anti,ddo = get_reward(old_state,next_state,old_pos,next_pos,target_pos,tt)
            
            if anti == False:
                fail = 0
            else:
                fail += 1
                
            if fail >= 5 and ddo == False:
                #reward -= 200 / (tt+1)
                dw = True
                
            if inside:
                dw = True
                
                vend = env.get_real_vvdd()[0:2]
                
                dotvvdd = np.dot(vend,dstart)
                norm_vvv = np.linalg.norm(vend)
                norm_ddd = np.linalg.norm(dstart)
                
                seugol = math.acos( dotvvdd / (norm_vvv * norm_ddd) )

                
                reward = reward + 300 *max( math.cos(seugol), 0.0) / tt
                ddo = True
                
            if dw:
                done = True
                

            
            if episode % 20 == 0 or ddo:
                fff = open('./epi_info2/{}.txt'.format(episode),'a')
                fff.write("###############################\n")
                fff.write(f"Steps: {tt}\n")
                fff.write(f"mean: {mean}\n")
                fff.write(f"a_log_prob: {a_log_prob}\n")
                #ff.write(f"old_state: {old_state}\n")
                fff.write(f"action: {action}\n")
                fff.write(f"next_state: {next_state}\n")
                #fff.write(f"old_pos: {old_pos}\n")
                #fff.write(f"average_action: {ave_action}\n")
                fff.write(f"next_pos: {next_pos}\n")
                fff.write(f"target_pos: {target_pos}\n")
                fff.write(f"reward: {reward}\n")
                fff.close()
            

        #done = ddo or done
            replay_buffer.add((old_all_state, next_all_state, action, reward, done, a_log_prob, dw))
            if replay_buffer.get_size() == 256:
                update_turns += 1
                acl,crl = ppo.update(replay_buffer, iterations, batch_size=batch_size, update_turns = update_turns)
                print(f"Update: {update_turns}  actor_loss: {acl.item()/iterations:.3f}, critic_loss: {crl.item()/iterations:.3f} ") 
                
                fff1 = open('result14.txt', 'a')
                fff1.write(f"Update: {update_turns}  actor_loss: {acl.item()/iterations:.3f}, critic_loss: {crl.item()/iterations:.3f} \n")
                fff1.close()
                
                if update_turns % 1 == 0 :
                    ppo.save(episode)  
    
    
    
        old_state = next_state
        old_pos = next_pos
        old_all_state = next_all_state
        
        next_all_state = np.zeros(100)
        
        episode_reward += reward
        
        tt += 1
    
    reward_ave.append(episode_reward/tt)
    
    if episode_reward >= 60:
        ch_t += 0
    
    if ddo:
        suc_s += 1

    #if len(replay_buffer.storage) > min_size:
    print(f"Epi: {episode}  Last action: { np.around(np.array(action),decimals=3)  } Win:{ddo} Ts: {suc_s}") 
    
    fff1 = open('result14.txt', 'a')
    fff1.write(f"Epi: {episode}  Last action: { np.around(np.array(action),decimals=3)  } Win:{ddo} Ts: {suc_s}\n")
    fff1.close()
    #print("finished")


    
    if tt > 0:
        ave_reward = episode_reward / tt
        fff = open('print_out14.txt','a')
        fff.write("###############################\n")
        fff.write(f"Episode: {episode},Steps: {tt} Reward: {episode_reward:.3f} Ave_reward: {ave_reward:.3f}\n")
        fff.write(f"Target: {target_pos[0]:.3f}, {target_pos[1]:.3f}\n")
        #fff.write(f"average_action: {ave_action}\n")
        fff.write(f"final_position: {next_pos[0]:.3f}, {next_pos[1]:.3f}\n")
        fff.write(f"final_velocity: {next_state[0]:.3f}, {next_state[1]:.3f}\n")
        fff.close()
    
    suc_num.append(ddo)
    if len(suc_num) > 100:
        removed_value = suc_num.pop(0)
        if removed_value:
            suc_s -= 1

    if suc_s >= 80:
        print(f"Finish！{episode}")
        ppo.save(episode)
        #break;
