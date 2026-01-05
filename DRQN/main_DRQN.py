
"""
@author: cep
"""
import numpy as np
import SWMM_ENV
import DQN
#import Rainfall_data as RD
import datetime
import tensorflow as tf
#import matplotlib.pyplot as plt
tf.compat.v1.experimental.output_all_intermediates(True)
env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV.SWMM_ENV(env_params)

#prepare rainfall
raindata1 = np.load('training_raindata.npy').tolist()

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':2**len(env.config['action_assets']),
    'sequence_dim':5,
    'lstm_units': 64,
    'evalnet_layer_A':3,
    'evalnet_A':[{'num':64},{'num':64},{'num':64}],
    'evalnet_layer_V':3,
    'evalnet_V':[{'num':64},{'num':64},{'num':64}],
    'targetnet_layer_A':3,
    'targetnet_A':[{'num':64},{'num':64},{'num':64}],
    'targetnet_layer_V':3,
    'targetnet_V':[{'num':64},{'num':64},{'num':64}],
    
    'num_rain':100,
    'training_step':100,
    'gamma':0.3,
    'epsilon':1,
    'ep_min':0.01,
    'ep_decay':0.995
}
agent = DQN.DQN(agent_params,env)
print('Model done')
history = agent.train(raindata1)
print('Training done')

raindata2 = np.load('test_raindata.npy').tolist()
agent.load_model()
#for i in range(len(raindata)):
for i in range(6):
    print('test:',i)
    test_his = agent.test(raindata2[i])
    np.save('./Results/'+str(i+100)+'.npy',test_his)

# data=np.load('C:/Users/Enpei Chen/DQN/Results/101.npy', allow_pickle=True)
raindata2 = np.load('test_raindata_RCP.npy').tolist()
agent.load_model()
#for i in range(len(raindata)):
for i in range(6):
    print('test:',i)
    test_his = agent.test(raindata2[i])
    np.save('./Results/'+str(i+1000)+'.npy',test_his)