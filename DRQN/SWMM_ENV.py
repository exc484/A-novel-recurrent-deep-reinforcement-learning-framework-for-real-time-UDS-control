
"""
@author: cep
"""
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np
#import pyswmm.toolkitapi as tkai

from swmm_api.input_file import read_inp_file
from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES
import tensorflow as tf
#import matplotlib.pyplot as plt
tf.compat.v1.experimental.output_all_intermediates(True)
import matplotlib.pyplot as plt
import datetime
import yaml

class SWMM_ENV:
    #can be used for every SWMM inp
    def __init__(self,params):
        '''
        params: a dictionary with input
        orf: original file of swmm inp
        control_asset: list of contorl objective, pumps' name
        advance_seconds: simulation time interval
        flood_nodes: selected node for flooding checking
        '''
        self.params = params
        self.config = yaml.load(open(self.params['orf']+".yaml"), yaml.FullLoader)
        #self.t=[] ;params='chaohu'
    
    def reset(self,rain):
        inp = read_inp_file(self.params['orf']+'.inp')
        inp[TIMESERIES]['rainfall']=TimeseriesData('rainfall',data=rain)
        inp.write_file(self.params['orf']+'_rain.inp')#形成有降雨的inp文件
        self.sim=Simulation(self.params['orf']+'_rain.inp')
        self.sim.start()
        
        #模拟一步
        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params['advance_seconds'])
        
        #obtain states and reward term by yaml (config)
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depth':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'total_inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)
        return states
        
    def step(self,action):
        #初始化
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        sys = SystemStats(self.sim)
        #obtain states and reward term by yaml (config)
        states = []
        
        #设置控制
        for item,a in zip(self.config['action_assets'],action):#zip后的结果是譬如(1, 'a')的元组；控制pump0,1开关
            links[item].target_setting = a
        
        
        #模拟一步
        if self.params['advance_seconds'] is None:
            time = self.sim._model.swmm_step()
        else:
            time = self.sim._model.swmm_stride(self.params['advance_seconds'])
        #self.t.append(self.sim._model.getCurrentSimulationTime())
        done = False if time > 0 else True
        
        #获取reward
        #flooding,CSO,inflow=0,0,0
        
        #for _temp in self.config['reward_targets']:
            #if _temp[1] == 'flooding':
                #if _temp[0] == 'system':
                #cum_flooding = sys.routing_stats[_temp[1]]           
                #flooding += (cum_flooding - self.data_log[_temp[1]][_temp[0]][-1]) *_temp[2]
        flooding = sys.routing_stats['flooding']    
                
                # log the cumulative value
                #self.data_log[_temp[1]][_temp[0]].append(cum_flooding)
            #else:
                
                #cum_cso = sys.routing_stats['outflow']
                #cum_cso = nodes[_temp[0]].volume
                #CSO += (cum_cso - self.data_log[_temp[1]][_temp[0]][-1]) * _temp[2]
        #CSO = nodes["CC-1"].cumulative_inflow + nodes["CC-2"].cumulative_inflow + nodes["JK-1"].cumulative_inflow + nodes["JK-2"].cumulative_inflow
        #CSO1 = nodes["CC-1"].total_inflow + nodes["CC-2"].total_inflow + nodes["JK-1"].total_inflow + nodes["JK-2"].total_inflow
        CSO2 = nodes["CC-1"].cumulative_inflow + nodes["CC-2"].cumulative_inflow + nodes["JK-1"].cumulative_inflow + nodes["JK-2"].cumulative_inflow
                # log the cumulative value
                #self.data_log[_temp[1]][_temp[0]].append(cum_cso)
                
        inflow = sys.routing_stats['dry_weather_inflow']+sys.routing_stats['wet_weather_inflow']+sys.routing_stats['groundwater_inflow']+sys.routing_stats['II_inflow']
            
            #nodes[_temp[0]].total_inflow
        objective1 = nodes["CC-storage"].depth
        objective2 = nodes["JK-storage"].depth
        pump1_flow = links["CC-R1"].flow
        pump2_flow = links["CC-R2"].flow
        pump3_flow = links["CC-S1"].flow
        pump4_flow = links["CC-S2"].flow
        pump5_flow = links["JK-R1"].flow
        pump6_flow = links["JK-R2"].flow
        pump7_flow = links["JK-S"].flow
        outfall1_flow = nodes["CC-1"].total_inflow
        outfall2_flow = nodes["CC-2"].total_inflow
        outfall3_flow = nodes["JK-1"].total_inflow
        outfall4_flow = nodes["JK-2"].total_inflow
        penalty = objective1 + objective2 - 0.9*(2.3+5.8) 
        rewards = (inflow-(flooding+CSO2))/inflow - 0.1*penalty        
        
        #获取模拟结果
        # nodes = Nodes(self.sim)
        # links = Links(self.sim)
        # rgs = RainGages(self.sim)
        # sys = SystemStats(self.sim)
        # #obtain states and reward term by yaml (config)
        # states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depth':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'total_inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)
        
        #降雨结束检测
        if done:
            self.sim._model.swmm_end()
            self.sim._model.swmm_close()
        return states,rewards,flooding,CSO2,objective1,objective2,pump1_flow,pump2_flow,pump3_flow,pump4_flow,pump5_flow,pump6_flow,pump7_flow,outfall1_flow,outfall2_flow,outfall3_flow,outfall4_flow,done
        

