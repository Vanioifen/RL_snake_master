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
        #self.log_std = torch.nn.Parameter(torch.ones(1,action_dim) * - 1.5)
        self.log_std = torch.nn.Parameter(torch.zeros(1, action_dim)) 
        
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
            #action = torch.clamp(action, expand_min[0], expand_max[0])  # [-max,max]
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
        
        
        
    def update(self, replay_buffer, iterations, batch_size, update_turns):  # discount = gamma
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
        

        
        adv = []
        gae = 0
        reward =  (reward + 1) / 20
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

        
        
        a_lr_now = max ( a_lr * (1 - self.steps / 2000), 5e-6)
        c_lr_now = max ( c_lr * (1 - self.steps / 2000), 5e-6)
        
        for p in self.actor_optimizer.param_groups:
            p['lr'] = a_lr_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = c_lr_now
        
        self.steps += 1
        
        # learning rate
        

        return acl/self.epochs ,crl/self.epochs

        #print(f"actor_loss: {acl/iterations:.3f}, critic_loss: {crl/iterations:.3f}")     
                
    def save(self, timestep): 
        
        torch.save(self.actor.state_dict(), "./model14/ddpg_actor{}.pth".format(update_turns))
        torch.save(self.critic.state_dict(), "./model14/ddpg_critic{}.pth".format(update_turns))

    def load(self, timestep):
        
        self.actor.load_state_dict(torch.load("./model14/ddpg_actor{}.pth".format(update_turns)))
        self.critic.load_state_dict(torch.load("./model14/ddpg_critic{}.pth".format(update_turns)))