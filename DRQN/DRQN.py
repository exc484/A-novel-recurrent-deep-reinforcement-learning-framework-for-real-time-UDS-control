
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
import tensorflow as tf
#import matplotlib.pyplot as plt

class DQN:
    
    def __init__(self,params,env):
        #tf.compat.v1.disable_eager_execution()
        #tf.compat.v1.experimental.output_all_intermediates(True)
        self.params=params
        self.memory_buffer = deque(maxlen=2000)
        self.env=env
        self.action_table=pd.read_csv('./DQN_action_table.csv').values[:,1:]
        print('table shape: ',self.action_table.shape)
        self.state_queue = deque(maxlen=self.params['sequence_dim'])
        self.model=self._build_net()
        self.target_model=self._build_net()
        
# =============================================================================
#     def df_to_xy(self,state,window_size):
#         x=[]
#         for i in range(len(state[:,0])-window_size+1):
#             row_x=[a for a in state[i:i+window_size]]
#             x.append(row_x)
#         return np.array(x)
# =============================================================================
    
    def _get_sequence(self, buffer, idx, seq_len):
        seq = []
        start = idx - seq_len + 1
        for i in range(start, idx + 1):
            seq.append(buffer[i][0])  # state
        return np.array(seq)
    
    def _get_next_sequence(self, buffer, idx, seq_len):
        seq = []
        start = idx - seq_len + 2
        for i in range(start, idx + 2):
            seq.append(buffer[i][3])  # next_state
        return np.array(seq)
        
    def _build_net(self):#在 Keras 中，输入特征的数量是根据输入数据的形状自动确定的，因此不需要显式
        #DQN
        #eval net
        self.s = layers.Input(shape=(None,self.params['state_dim']),name='s_input')#用来保存输入层的对象，输入层的名称被指定为's_input';None任意切片数量
        V_prev = self.s#为了初始化 V_prev，将其赋值为输入层的输出，以便在循环中连接后续的隐藏层
        V_prev = layers.LSTM(units=self.params['lstm_units'], return_sequences=False)(V_prev)
        for i in np.arange(self.params['evalnet_layer_V']-1):#2层或者3层
            V_prev=layers.Dense(self.params['evalnet_V'][i]['num'], activation='relu', name='evalnet_V'+str(i))(V_prev)
        self.eval_out=layers.Dense(self.params['action_dim'], activation='linear', name='evalnet_out_V')(V_prev)
        
        model=models.Model(inputs=[self.s],outputs=self.eval_out)#models.Model是一个模型类，用于实例化一个神经网络模型
        return model
    
    def choose_action(self,state_seq,train_log):
        #input state, output action
        state_seq = np.expand_dims(state_seq, axis=0)# shape -> [1, seq_len, state_dim]
        if train_log:#train_log 是一个布尔型变量
            #epsilon greedy
            #state = np.reshape(np.array([state]), [1, 1, self.params['state_dim']])
            pa = np.random.uniform()#浮点数的取值范围在[0.0, 1.0)之间
            if pa < self.params['epsilon']:
                action = np.random.randint(self.params['action_dim'])
            else:
                action_value = self.model.predict(state_seq)
                action = np.argmax(action_value)
        else:
            #state = np.reshape(np.array([state]), [1, 1, self.params['state_dim']])
            action_value = self.model.predict(state_seq)
            action = np.argmax(action_value)
        return action
#state = np.reshape(state, [1, args.time_steps, self.state_dim])

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)#为了确保在整个类中都可以访问和修改记忆缓冲区，需要使用 self.memory_buffer

# =============================================================================
#     def process_batch(self, batch):
#          # 从经验池中随机采样一个batch
#         data = random.sample(self.memory_buffer, batch)
#         #start_index = random.randint(0, len(self.memory_buffer)-batch)
#         #data = np.array(self.memory_buffer)[start_index:start_index + batch]
#         # 生成Q_target;Q现实也就是y
#         states = np.array([d[0] for d in data])#此矩阵维度是batch*state数量
#         next_states = np.array([d[3] for d in data])
#         states = self.df_to_xy(states,self.params['sequence_dim'])
#         next_states = self.df_to_xy(next_states,self.params['sequence_dim'])
#         y = self.model.predict(states)
#         q = self.target_model.predict(next_states)#Q现实的计算要用下一个state
# 
#         for i, (_, action, reward, _, done) in enumerate(data[:-(self.params['sequence_dim']-1)]):
#             target = reward
#             if not done:
#                 target += self.params['gamma'] * np.amax(q[i])
#             y[i][action] = target
#         return states, y
# =============================================================================
    
    def process_batch(self, batch):
        seq_len = self.params['sequence_dim']
        buffer = list(self.memory_buffer)
    
        # 关键：禁止 idx 太小
        max_idx = len(buffer) - 2
        min_idx = seq_len - 1
    
        #assert max_idx >= min_idx, "Replay buffer too small for sequence sampling"
    
        indices = np.random.randint(min_idx, max_idx + 1, size=batch)
    
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
    
        for idx in indices:
            states.append(self._get_sequence(buffer, idx, seq_len))
            next_states.append(self._get_next_sequence(buffer, idx, seq_len))
            actions.append(buffer[idx][1])
            rewards.append(buffer[idx][2])
            dones.append(buffer[idx][4])
    
        states = np.array(states)        # [batch, seq_len, state_dim]
        next_states = np.array(next_states)
        y = self.model.predict(states)
        q = self.target_model.predict(next_states)#Q现实的计算要用下一个state

        for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
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
                self.state_queue.clear()
                for _ in range(self.params['sequence_dim']):
                    self.state_queue.append(s)
                done, batch = False, 0
                while not done:
                    state_seq = np.array(self.state_queue)
                    a = self.choose_action(state_seq,True)#train_log=True
                    action = self.action_table[a,:].tolist()
                    snext,reward,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,done = self.env.step(action)#states,rewards,flooding,CSO,done
                    self.remember(s, a, reward, snext, done)#buffer第一个数据不要
                    self.state_queue.append(snext)
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
            #if reward_sum > 5280:
            self.model.save_weights('./model/dqn.h5')
            np.save('history.npy',np.array(history))
        return history
    
    def load_model(self):
        self.model.load_weights('./model/dqn.h5')
    
    def test(self,rain):
        # simulation on given rainfall
        test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[], 'O1':[], 'O2':[], 'P1':[], 'P2':[], 'P3':[], 'P4':[], 'P5':[], 'P6':[], 'P7':[], 'f1':[], 'f2':[], 'f3':[], 'f4':[]}
        s = self.env.reset(rain)
        self.state_queue.clear()
        for _ in range(self.params['sequence_dim']):
            self.state_queue.append(s)
        done, t= False, 0
        test_history['time'].append(t)
        test_history['state'].append(s)
        while not done:
            state_seq = np.array(self.state_queue)
            a = self.choose_action(state_seq,False)
            action = self.action_table[a,:].tolist()
            snext,reward,F,C,O1,O2,P1,P2,P3,P4,P5,P6,P7,f1,f2,f3,f4,done = self.env.step(action)#return states,rewards,flooding,CSO,done
            self.state_queue.append(snext)
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
    
    
        