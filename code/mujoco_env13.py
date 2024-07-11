import gym
from gym import spaces
import mujoco_py
import numpy as np
import math
import torch
import math
from scipy.signal import butter,filtfilt


class MujocoEnv13(gym.Env):
    def __init__(self, model_path, seed = 0):
        super(MujocoEnv13, self).__init__()
        self.model = mujoco_py.load_model_from_path(model_path)        
        self.sim = mujoco_py.MjSim(self.model)

        self.target_vel = 0.08

        self.num_bodys = self.sim.model.nbody  # model中的body数量
        self.t = 0
        

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.seed = seed
        self._set_seed()
        
        self.dw = False
        self.tr = False

        self.now_state = np.zeros(10)   # 表示当前状态 
        self.now_pos = np.array([-0.3,0])
        
        self.u0  = np.array([0.1, 0, 0, 0,    0, 0.1, 0, 0,   0.1,0, 0, 0,    0, 0.1,0, 0,   0.1, 0, 0, 0])
        self.u00 = np.array([0.1, 0, 0, 0,    0, 0.1, 0, 0,   0.1,0, 0, 0,    0, 0.1,0, 0,   0.1, 0, 0, 0])
        
        self.dt = 0.01

        

        
        
        self.total_vx = np.zeros(49)
        self.total_vy = np.zeros(49)
     

            

        
        self.target_point = np.array([3.0,0.0])
    
    def get_sim(self):
        
        simm = self.sim
        
        return simm
    
    def change_target(self, target_point):
        self.target_point = target_point

        
        
    def _get_position(self):
        now_pos = np.zeros(2)
        
        total_px = 0.0
        total_py = 0.0
        
        for i in range(int(self.num_bodys)):
            if i != 0:
                total_px += self.sim.data.body_xpos[i][0]
                total_py += self.sim.data.body_xpos[i][1]
        
        now_pos[0] = total_px / (self.num_bodys - 1)
        now_pos[1] = total_py / (self.num_bodys - 1)
        
        return now_pos
    
    def _get_real_vvdd(self):
        now_state = np.zeros(4)
        
        w_size = 50
        
        
        now_vx = np.convolve(self.total_vx[-50:], np.ones(w_size)/w_size, mode='valid')
        now_vy = np.convolve(self.total_vy[-50:], np.ones(w_size)/w_size, mode='valid')
        
        now_poss =  self._get_position()
        
        now_state[0] = now_vx[0]
        now_state[1] = now_vy[0]
        now_state[2] = self.target_point[0] - now_poss[0]
        now_state[3] = self.target_point[1] - now_poss[1]
        
        return now_state
        

    def _get_observation(self):
        
        now_state = np.zeros(10)
        state1 = np.zeros(10)
        
        w_size = 50
        
        
        now_vx = np.convolve(self.total_vx[-50:], np.ones(w_size)/w_size, mode='valid')
        now_vy = np.convolve(self.total_vy[-50:], np.ones(w_size)/w_size, mode='valid')
        
        #total_velx = self.sim.data.body_xvelp[1][0]
        #total_vely = self.sim.data.body_xvelp[1][1]
        
        #now_state[2] = math.sqrt(total_velx**2 + total_vely**2)
        
        #now_state[0] = total_velx / 5
        #now_state[1] = total_vely / 5
        now_poss =  self._get_position()
        
        now_state[0] = now_vx[0]
        now_state[1] = now_vy[0]
        now_state[2] = self.target_point[0] - now_poss[0]
        now_state[3] = self.target_point[1] - now_poss[1]

        #########################################
        # 头部角 与 角速度

        now_state[4] = math.asin(self.sim.data.body_xmat[1][1])
        #now_state[10] = self.sim.data.body_xvelr[1][2]

        #print(self.sim.data.qpos)

        #########################################

        for i in range (5):
            now_state[i+5] = self.sim.data.qpos[i+3]
            #now_state[i+11] = self.sim.data.qvel[i+3] 
           
        #print("jdsaklfjkl")
        #print(now_state)
        
        #now_state[14] = self.target_vel
        
        
        

        
        #return now_state#now_state
        
        
        #now_state[10] = np.mean(now_state[2:8]) - np.arctan2(now_state[8],now_state[9] )
        
        #print(now_state)
        
        vx = now_state[0]
        vy = now_state[1]
        
        vv = np.array([vx,vy])
        
        
        dx = now_state[2]
        dy = now_state[3]
        
        dd = np.array([dx,dy])
        
        dotvd = np.dot(vv,dd)
        norm_vv = np.linalg.norm(vv)
        norm_dd = np.linalg.norm(dd)

        cross_p = vx * dy - vy * dx
        
        
        
        
        norm_vv = max(1e-4,norm_vv)
        norm_dd = max(1e-4,norm_dd)
        
        ugol = math.acos( dotvd / (norm_vv * norm_dd) )
        
        if cross_p < 0:
            ugol = -ugol
        
        

        state1[0] = norm_vv * math.cos(ugol)
        state1[1] = norm_vv * math.sin(ugol)
        
        state1[2] = norm_dd
        
        state1[3] = ugol
        
        
        #for i in range(4,10):
        state1[4]  = now_state[i] - ugol
        
        for i in range(5,10):
            state1[i] = now_state[i]
        
        
        #for i in range(10,16):
            #state1[i] = now_state[i]

        return state1
        
        
    '''
    def _get_reward(self, next_state, now_pos):
        dx = v_change[0]
        dy = v_change[1]
        velx = next_state[0]
        vely = next_state[1] 
        return (dx-dy)
    '''
    
    def _is_done(self):
        if self.t == 20000:
            self.t = 0
            return True
        else:
            return False
        
    def _is_dead(self):
        #if self.now_pos[0] > 5 or self.now_pos[0]< -2:
        #    self.t = 0
        #    return True
        #if abs(self.now_pos[1]) > 5:
        #    self.t = 0
        #    return True

        
        #if abs(self.now_pos[1])   >= max(abs(self.target_point[1]) + 1 , 1.2):
        #    self.t = 0
        #    return True
        
        #if ( (self.now_pos[0]-self.target_point[0]) ** 2 + (self.now_pos[1]-self.target_point[1]) ** 2 ) ** 0.5 < 0.25:
        #    self.t = 0
        #    return True
        
        if ( (self.now_pos[0]-self.target_point[0]) ** 2 + (self.now_pos[1]-self.target_point[1]) ** 2 ) ** 0.5  > 6:
            self.t = 0
            return True
        
        elif max(abs(self.now_state[0:2])) > 50:
            self.t = 0
            return True
        elif max(abs(self.now_state[8:14])) > 1000:
            self.t = 0
            return True
        else:
            return False
    
    def osi_system(self,a,b,uu0,uu1,wij,TT,tt):


        du = np.zeros(20)
        g = lambda x: np.maximum(0, x)


        for i in range(5):
            xx1 = 0
            xx2 = 0
            for j in range(5):
                #xx1 += wij[i][j] * g(u[j*4 + 0])
                #xx2 += wij[i][j] * g(u[j*4 + 1])
                xx1 += wij[i][j] * self.u0[j*4 + 2]
                xx2 += wij[i][j] * self.u0[j*4 + 3]


            du[i*4 + 0] = (uu0[i] - self.u0[i*4 + 0] - b * self.u0[i*4 + 2] - a * g(self.u0[i*4 + 1]) - xx1) * TT
            du[i*4 + 1] = (uu1[i] - self.u0[i*4 + 1] - b * self.u0[i*4 + 3] - a * g(self.u0[i*4 + 0]) - xx2) * TT
            du[i*4 + 2] = (- self.u0[i*4 + 2] +  g(self.u0[i*4 + 0]) ) * TT * tt
            du[i*4 + 3] = (- self.u0[i*4 + 3] +  g(self.u0[i*4 + 1]) ) * TT * tt

    #du11_dt = 1 - 2*g(u21) - u11 - v11
    #du21_dt = 1 - 2*g(u11) - u21 - v21
    #dv11_dt = -v11 + 3 * g(u11)
    #dv21_dt = -v21 + 3 * g(u21)

        return du
        

    def get_self_target_vel(self,target_vel):
        self.target_vel = target_vel


    def step(self, actionn):
        # 先取状态值
        #now_state = self._get_observation()
        info = {}
        #for i in range(self.sim.data.ctrl.size)
        #    self.sim.data.ctrl[i] = action[i]
        #self.X = np.zeros(5)
        #self.X = action
        dt = self.dt

        
        #print(action)
        tt = self.t
        
        ww = 2
        beta = 10
        TT = 3.2 
        ttt = 1
        uu0 = np.ones(5) 
        uu1 = np.ones(5)
        

        
        uxx = 0.5

        action = np.zeros(5)
        
        for i in range(5):
            action[i] = actionn[i] * 0.1 + 1.0
            
        action = np.clip(action,0.9,1.1)
        

        uu0[0] = action[0] * uxx #* upand
        uu0[1] = action[1] * uxx * 3 #* upand
        uu0[2] = action[2] * uxx * 4 #* upand
        uu0[3] = action[3] * uxx * 3 #* upand
        uu0[4] = action[4] * uxx #* upand


        uu1[0] = (2 - action[0]) * uxx #* upand
        uu1[1] = (2 - action[1]) * uxx * 3 #* upand
        uu1[2] = (2 - action[2]) * uxx * 4 #* upand   
        uu1[3] = (2 - action[3]) * uxx * 3 #* upand
        uu1[4] = (2 - action[4]) * uxx #* upand


        wij = np.zeros(5*5)
        wij = wij.reshape(5,5)
        for i in range(5-1):
            wij[i][i+1] = math.pi/2
        


            
        u_max = np.array([ 1, 1, 1, 1, 1])
        u_min = -u_max    
 

        self.u0 = self.u0 + dt * self.osi_system(ww,beta,uu0,uu1,wij,TT,ttt) 

                                                      #(u,a,b,uu0,uu1,wij,TT,tt):
        

        y = np.zeros(5)
        
        for i in range(5):
            y[i] = max(self.u0[i*4],0) - max(self.u0[i*4+1],0)

            
            
        total_velx = 0.0
        total_vely = 0.0


        for i in range(1,12):
            total_velx += self.sim.data.body_xvelp[i][0]
            total_vely += self.sim.data.body_xvelp[i][1]

        total_velx /=11
        total_vely /=11

        self.total_vx=np.append(self.total_vx, total_velx)
        self.total_vy=np.append(self.total_vy, total_vely)
            
            
            

            
            #self.sim.data.ctrl[0] = u_fil[-1][0] + action
        self.sim.data.ctrl[:] = y[:]
        

        
        
        self.sim.step()
        

       
        next_pos = self._get_position()
        
        next_state = self._get_observation()
        
        
        has_nan = np.isnan(np.array(next_state)).any()
        if has_nan:
            print("have_nan")
            self.t = 0
            return np.zeros(30),True,np.array([-3,0]),info
        
        self.now_pos = next_pos
        
        self.now_state = next_state
        
        

        #reward = self._get_reward(next_state)
        dw = self._is_dead()
        done = self._is_done() 
        if done or dw:
            self.reset()

        self.t += 1
        
        
        return next_state, done, next_pos, dw, info
        #return now_state, next_state, reward, done

    def reset(self):
        
        self.sim.reset()
        self.u0 = self.u00
        self.t = 0
       
        self.now_pos = self._get_position()#np.array([-0.3,0])
        self.now_state = self._get_observation()
        #print(self.now_state)
        
        self.X = np.zeros(5)
        


 
        statee = self.now_state
        return statee
    
    def get_now_state(self):


        statee = self._get_observation()
        
        return statee
    
    def get_now_pos(self):
        
        poss = self.now_pos
        
        return poss
    
    def get_real_vvdd(self):
        real_state = self._get_real_vvdd()
        return real_state
    
    def _set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)    
    #def render(self):
    #    mujoco_py.MjViewer(self.sim).render()
    
# 创建环境实例
#env = MujocoEnv("/home/wyf/rl_snake/t1_move/snake7.xml")

# 可以像其他 Gym 环境一样使用它

