#!/home/void/kan/pykan/bin/python3

from kanqn import Agent
from environment import Env
from evaluation import Evaluation
import numpy as np
import rospy
import matplotlib.pyplot as plt
import torch as T
import json 
from gazebo_msgs.srv import DeleteModel
import random
import os

# Load settings from settings
settings_path = os.path.dirname(os.path.realpath(__file__)) 
with open (settings_path+'/settings.json', 'r') as f:
    settings = json.load(f)

# Get the directory of the currently executing script
script_directory = os.path.dirname(os.path.realpath(__file__))
# Go up one level (parent directory)
parent_directory = os.path.abspath(os.path.join(script_directory, os.pardir))

def train_dqn(agent, env, eval, epochs, steps):
    scores, eps_history, epoch_time, pos_epoch, goal_list = [], [], [], [], []
    collision_count, goal_count, timeout_count = 0, 0, 0
    reward_history = []

    for epoch in range(epochs):
        score = 0
        pos_step = []
        done, info = False, False
        observation = env.reset()
        start_time = eval.get_time()
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

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            reward_history.append(reward)   

            pos_x, pos_y = eval.get_pose()
            pos_step.append([pos_x, pos_y])

            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        end_time = eval.get_time()
        epoch_time.append([start_time, end_time])
        pos_epoch.append(pos_step)
        scores.append(score)
        eps_history.append(agent.epsilon)
        save_best_model(agent, epoch, score, max_epochs=epochs, last_epochs=4000)
        print(f"Episode {epoch+1}/{epochs} | Score = {round(score,2)} | Epsilon = {agent.epsilon    :.2f}")
    return scores, eps_history, epoch_time, pos_epoch, collision_count, goal_count, timeout_count, goal_list

def save_best_model(agent, epoch, reward, max_epochs, last_epochs=100):
    """
    Save the model with the highest reward from the last `last_epochs` out of `max_epochs`.

    Parameters:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): Current epoch number.
        reward (float): Current reward.
        max_epochs (int): Total number of epochs.
        last_epochs (int): Number of final epochs to consider for saving.
        saved_model_path (str): Path to save the best model.

    Returns:
        None
    """

    if not os.path.exists(parent_directory+settings["DIRECTORY_TO_SAVE_MODELS"]):
        os.makedirs(parent_directory+settings["DIRECTORY_TO_SAVE_MODELS"])
    saved_model_path = parent_directory+settings["DIRECTORY_TO_SAVE_MODELS"] + settings["NAME_RUN"]  + ".pth"
    
    if epoch > max_epochs - last_epochs:
        if os.path.exists(saved_model_path):
            checkpoint = T.load(saved_model_path)
            best_reward = checkpoint['reward']
            if reward > best_reward:
                # If the current reward is better, save the model
                T.save({
                    'epoch': epoch,
                    'model_state_dict': agent.Q_eval.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': reward
                }, saved_model_path)
        else:
            # Save the model for the first time
            T.save({
                'epoch': epoch,
                'model_state_dict': agent.Q_eval.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': reward
            }, saved_model_path)

def plot_and_save_results(scores, window_size, save_path):
    x = np.arange(len(scores)- window_size+1)
    standard_deviation = np.array([np.std(scores[i:i+window_size]) for i in range(len(scores)- window_size + 1)])
    average_score = np.array([np.mean(scores[i:i+window_size]) for i in range(len(scores))])
    rolling_mean = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.figure(dpi=300)
    plt.plot(x, rolling_mean, label="Mean Reward")
    plt.fill_between(x, rolling_mean - standard_deviation, rolling_mean + standard_deviation, color='blue', alpha=0.05, label='Rewards')
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Rewards", fontsize=14)
    plt.title('Reward per Episode',fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(save_path)

def save_evaluation_results(eval, pos_epoch, scores, goal_list, save_path):
    distance_in_epoch, mean_distance = eval.mean_distance(pos_epoch)
    mean_time = eval.mean_time(epoch_time)
    cov = eval.coefficient_of_variation(scores)
    ideal_distance = np.mean(goal_list)
    mean_reward = np.mean(scores)
    results = {
        'goal_count': goal_count,
        'collision count': collision_count,
        'timeout_count': timeout_count,
        'mean_distance': mean_distance,
        'mean_time': mean_time,
        'ideal_distance': ideal_distance,
        'ideal_time': ideal_distance/0.22, # 0.22 is the max linear velocity
        'coefficient of variation': cov,
        'mean_reward': mean_reward,
        'reward_list': scores
    }
    print(results)
    json.dump(results, open(save_path, 'w'))

def set_seed(seed_value):
    # Set random seed for pytorch
    T.random.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


if __name__ == '__main__':
    rospy.init_node('env', anonymous=True)

    # Initialize agent, environment, and evaluation
    agent = Agent(gamma=settings["GAMMA"], epsilon=settings["EPSILON"], batch_size=settings["BATCH_SIZE"], n_actions=settings["NO_OF_ACTIONS"],  
                eps_end=settings["EPSILON_MIN"], input_dims=settings["INPUT_DIMENSION"], lr=settings["LEARNING_RATE"])
    
    agent.load_models()
    agent.initialize_models()
    
    env = Env()
    eval = Evaluation()

    epochs = settings["EPOCHS"] 
    steps = settings["STEPS"]
    window_size = 300
    results = train_dqn(agent, env, eval, epochs, steps)
    scores, eps_history, epoch_time, pos_epoch, collision_count, goal_count, timeout_count,goal_list = results
    
    if not os.path.exists(parent_directory+settings["DIRECTORY_TO_SAVE_RESULTS"]):
        os.makedirs(parent_directory+settings["DIRECTORY_TO_SAVE_RESULTS"])
    save_evaluation_results(eval, pos_epoch, scores, goal_list, parent_directory+settings["DIRECTORY_TO_SAVE_RESULTS"] + settings["NAME_RUN"] + ".json")
    if not os.path.exists(parent_directory+settings["DIRECTORY_TO_SAVE_GRAPHS"]):
        os.makedirs(parent_directory+settings["DIRECTORY_TO_SAVE_GRAPHS"])
    plot_and_save_results(scores, window_size, parent_directory+settings["DIRECTORY_TO_SAVE_GRAPHS"] + settings["NAME_RUN"] + ".png")

    # Plot policy
    plt.figure(dpi=300)
    agent.Q_eval.prune()
    agent.Q_eval.plot(mask=True)
    plt.savefig(parent_directory+'/policy/kaqn_policy_obstacle.png', dpi=1000)

    #Delete goal model
    rospy.wait_for_service('gazebo/delete_model')
    del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
    del_model_prox("goal")


