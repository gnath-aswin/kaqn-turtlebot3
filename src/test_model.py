#!/usr/bin/env python

import torch as T
import rospy
from dqn import DeepQNetwork
from environment import Env


if __name__ == '__main__':
    done = False
    rospy.init_node("test",anonymous=True)
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        

    actor = DeepQNetwork(lr=0.001,n_actions=5,input_dims=[22],fc1_dims=256,fc2_dims=256)
    checkpoint = T.load('model_200epoch_noobstacle.pth')
    actor.load_state_dict(checkpoint)

    env = Env()
    state = env.reset()
    actions = actor(T.from_numpy(state).to(device))
    while not done:
        action = T.argmax(actions).item()
        print(action)
        observation_,reward,done,info = env.step(action)
        print(reward)
        if info:
            done = True
        actions = actor(T.from_numpy(observation_).to(device))
        print(actions,done)
        

