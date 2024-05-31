#!/usr/bin/env python

from dqn import DeepQNetwork
from environment import Env
from evaluation import Evaluation
import numpy as np
import rospy
import matplotlib.pyplot as plt
import torch as T
import json 
from gazebo_msgs.srv import DeleteModel
import os

def validate(actor, env, eval, epochs, steps):
    scores, epoch_time, pos_epoch, goal_list = [], [], [], []  
    collision_count, goal_count, timeout_count = 0, 0, 0
    reward_history = []

    for i in range(epochs):
        score = 0
        pos_step = []
        done, info = False, False
        observation = env.reset()
        start_time = eval.get_time()
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        goal_list.append(env.initial_goal_distance)
        for step in range(steps):
            if step == steps - 1:
                timeout_count += 1

            if done:
                collision_count += 1
                break

            if info:
                goal_count += 1
                break

            action = T.argmax(actor(T.from_numpy(observation).to(device))).item()
            observation_, reward, done, info = env.step(action)
            reward_history.append(reward)   

            pos_x, pos_y = eval.get_pose()
            pos_step.append([pos_x, pos_y])

            score += reward
            observation = observation_

        end_time = eval.get_time()
        epoch_time.append([start_time, end_time])
        pos_epoch.append(pos_step)
        scores.append(score)
        print(f"episode:{i}, score:{round(score, 2)}")
    print(f"length: {len(reward_history)}\n max reward:{max(reward_history)}\n min reward:{min(reward_history)}")   
    return scores, epoch_time, pos_epoch, collision_count, goal_count, timeout_count, goal_list


def plot_and_save_results(scores, window_size, save_path):
    x = np.arange(len(scores))
    print(len(x))
    average_score = np.asanyarray([np.mean(scores[i:i+window_size]) for i in range(len(scores))])
    print(len(average_score))
    plt.plot(x,average_score, label="mean reward")
    plt.fill_between(x,average_score - scores, average_score + scores, color='blue', alpha=0.05, label='rewards')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title('Reward per Episode')
    plt.legend()
    plt.savefig(save_path)

def save_evaluation_results(eval, pos_epoch, scores, goal_list, save_path):
    distance_in_epoch, mean_distance = eval.mean_distance(pos_epoch)
    mean_time = eval.mean_time(epoch_time)
    cov = eval.coefficient_of_variation(scores)
    ideal_distance = np.mean(goal_list)
    results = {
        'goal_count': goal_count,
        'collision count': collision_count,
        'timeout_count': timeout_count,
        'mean_distance': mean_distance,
        'mean_time': mean_time,
        'coefficient of variation': cov,
        'ideal_distance': ideal_distance,
        'ideal_time': ideal_distance/0.22, # 0.22 is the max linear velocity
        'reward_list': scores
    }
    print(results)
    json.dump(results, open(save_path, 'w'))

if __name__ == '__main__':
    rospy.init_node('env', anonymous=True)
    # Get the directory of the currently executing script
    script_directory = os.path.dirname(os.path.realpath(__file__))
    # Go up one level (parent directory)
    parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir)) 
    with open (script_directory+'/settings.json', 'r') as f:
        settings = json.load(f)

    actor = DeepQNetwork(n_actions=settings["NO_OF_ACTIONS"], input_dims=settings["INPUT_DIMENSION"],
                        lr=settings["LEARNING_RATE"],fc1_dims=settings["HIDDEN_UNITS"], fc2_dims=settings["HIDDEN_UNITS"])
    checkpoint = T.load("../models/epoch6000_ 2.pth")
    actor.load_state_dict(checkpoint['model_state_dict'])
    env = Env()
    eval = Evaluation()
    epochs = settings["EPOCHS"] 
    steps = settings["STEPS"]
    window_size = 25

    results = validate(actor, env, eval, epochs, steps)
    scores, epoch_time, pos_epoch, collision_count, goal_count, timeout_count, goal_list = results
    save_evaluation_results(eval, pos_epoch, scores, goal_list, parent_directory+settings["DIRECTORY_TO_SAVE_RESULTS"] + settings["NAME_RUN"] + ".json")
    plot_and_save_results(scores, window_size, parent_directory+settings["DIRECTORY_TO_SAVE_GRAPHS"] + settings["NAME_RUN"] + ".png")


    #Delete goal model
    rospy.wait_for_service('gazebo/delete_model')
    del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    del_model_prox("goal")
