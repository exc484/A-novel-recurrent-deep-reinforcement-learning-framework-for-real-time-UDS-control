
"""
@author: cep
"""
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.optimizers import Adam

from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


class DQN:
    
    def __init__(self,params,env):
        tf.compat.v1.disable_eager_execution()
        self.params=params
        self.memory_buffer = deque(maxlen=2000)
        self.env=env
        self.action_table=pd.read_csv('./DQN_action_table.csv').values[:,1:]
        print('table shape: ',self.action_table.shape)
        
        self.model=self._build_net()
        self.target_model=self._build_net()
        
    def _build_net(self):#在 Keras 中，输入特征的数量是根据输入数据的形状自动确定的，因此不需要显式
        #DQN
        #eval net
        self.s = layers.Input(shape=self.params['state_dim'],name='s_input')#用来保存输入层的对象，输入层的名称被指定为's_input'
        V_prev = self.s#为了初始化 V_prev，将其赋值为输入层的输出，以便在循环中连接后续的隐藏层
        for i in np.arange(self.params['evalnet_layer_V']):
            V_prev=layers.Dense(self.params['evalnet_V'][i]['num'], activation='relu', name='evalnet_V'+str(i))(V_prev)
        self.eval_out=layers.Dense(self.params['action_dim'], activation='linear', name='evalnet_out_V')(V_prev)
        
        model=models.Model(inputs=[self.s],outputs=self.eval_out)#models.Model是一个模型类，用于实例化一个神经网络模型
        return model
    
    def choose_action(self,state,train_log):
        #input state, output action
        if train_log:#train_log 是一个布尔型变量
            #epsilon greedy
            pa = np.random.uniform()#浮点数的取值范围在[0.0, 1.0)之间
            if pa < self.params['epsilon']:
                action = np.random.randint(self.params['action_dim'])
            else:
                action_value = self.model.predict(np.array([state]))
                action = np.argmax(action_value)
        else:
            action_value = self.model.predict(np.array([state]))
            action = np.argmax(action_value)
        return action

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def process_batch(self, batch):
         # 从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, batch)
        # 生成Q_target;Q现实也就是y
        states = np.array([d[0] for d in data])#此矩阵维度是batch和state数量
        next_states = np.array([d[3] for d in data])
        y = self.model.predict(states)
        q = self.target_model.predict(next_states)#Q现实的计算要用下一个state

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.params['gamma'] * np.amax(q[i])
            y[i][action] = target
        return states, y
    
    
    def train(self,RainData):
        #sampling and upgrading
        history = {'episode': [], 'Batch_reward': [], 'Step_reward': [], 'Loss': []}
        self.model.compile(loss='mse', optimizer=Adam(1e-2))#optimizer = optim.Adam(model.parameters(), lr=1e-3)；criterion = nn.MSELoss()
        for j in range(self.params['training_step']):
            reward_sum = 0
            count = 0
            for i in range(self.params['num_rain']):
                print('training step:',j,' sampling num:',i)
                #Sampling: each rainfall represents one round of sampling
                batch=0
                reward_batch=0
                s = self.env.reset(RainData[i])
                done, batch = False, 0
                while not done:
                    a = self.choose_action(s,True)#train_log=True
                    action = self.action_table[a,:].tolist()
                    snext,reward,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,done = self.env.step(action)#states,rewards,flooding,CSO,done
                    self.remember(s, a, reward, snext, done)#buffer第一个数据不要
                    s = snext
                    batch+=1
                    reward_sum+=reward
                    reward_batch+=reward
                #Upgrading: each rainfall for one round of upgrading
                X, y = self.process_batch(batch)
                loss = self.model.train_on_batch(X, y)#该方法主要用于训练神经网络模型，通过将一批数据传递给模型，模型会根据输入数据进行前向传播计算，并与目标标签计算损失，然后使用反向传播算法更新模型的参数以降低损失。
                count += 1#在这个training step中有几场雨
                # 减小egreedy的epsilon参数。
                if self.params['epsilon'] >= self.params['ep_min']:
                    self.params['epsilon'] *= self.params['ep_decay']
                # 更新target_model
                if count % 10 == 0: #每一场雨结束后更新target model
                    self.target_model.set_weights(self.model.get_weights())
                if i % 5 == 0:
                    history['episode'].append(i)
                    history['Batch_reward'].append(reward_batch)
                    history['Loss'].append(loss)
                    print('Episode: {} | reward_batch: {} | loss: {:.3f} | e:{:.2f} | reward_sum:{}'.format(i, reward_batch, loss, self.params['epsilon'], reward_sum))
                if i % (self.params['num_rain']-1) == 0 and i != 0:                    
                    history['Step_reward'].append(reward_sum)
            #if reward_sum > 5250:
            self.model.save_weights('./model/dqn.h5')
            np.save('history.npy',np.array(history))
        return history
    
    def load_model(self):
        self.model.load_weights('./model/dqn.h5')
    
    def test(self,rain):
        # simulation on given rainfall
        test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[], 'O1':[], 'O2':[], 'P1':[], 'P2':[], 'P3':[], 'P4':[], 'P5':[], 'P6':[], 'P7':[], 'f1':[], 'f2':[], 'f3':[], 'f4':[]}
        s = self.env.reset(rain)
        done, t= False, 0
        test_history['time'].append(t)
        test_history['state'].append(s)
        while not done:
            a = self.choose_action(s,False)
            action = self.action_table[a,:].tolist()
            snext,reward,F,C,O1,O2,P1,P2,P3,P4,P5,P6,P7,f1,f2,f3,f4,done = self.env.step(action)#return states,rewards,flooding,CSO,done
            s = snext
            t += 1
            
            test_history['time'].append(t)
            test_history['state'].append(s)
            test_history['action'].append(action)
            test_history['reward'].append(reward)
            test_history['F'].append(F)
            test_history['C'].append(C)
            test_history['O1'].append(O1)
            test_history['O2'].append(O2)
            test_history['P1'].append(P1)
            test_history['P2'].append(P2)
            test_history['P3'].append(P3)
            test_history['P4'].append(P4)
            test_history['P5'].append(P5)
            test_history['P6'].append(P6)
            test_history['P7'].append(P7)
            test_history['f1'].append(f1)
            test_history['f2'].append(f2)
            test_history['f3'].append(f3)
            test_history['f4'].append(f4)
        return test_history
    
# data=np.load('C:/Users/Enpei Chen/DQN/Results/101.npy', allow_pickle=True)
    
    
        