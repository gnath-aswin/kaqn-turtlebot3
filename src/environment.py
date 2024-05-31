#!/usr/bin/env python

import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
from respawn import Respawn
from json import load
import os
from typing import List, Tuple

# Load settings from settings.json.
settings_path = os.path.dirname(os.path.realpath(__file__)) 
with open (settings_path+'/settings.json', 'r') as f:
    settings = load(f)


class Env():
    """
    Represents the environment for robot navigation.

    This class manages the state of the environment, including robot position,
    goal position, and action space. It also provides methods for resetting the
    environment and obtaining sensor data.

    Attributes:
        posex (float): The x-coordinate of the robot's position.
        posey (float): The y-coordinate of the robot's position.
        goal_x (float): The x-coordinate of the goal position.
        goal_y (float): The y-coordinate of the goal position.
        initial_goal_distance (float): The initial distance to the goal.
        prev_distance (float): The previous distance to the goal.
        heading (float): The angle between robot and goal.
        get_goalbox (bool): Indicates if the goal has been reached.
        init_goal (bool): Indicates if the goal has been initialized.
        action_space (list): The available actions for the robot.
        reset_robot: A service proxy for resetting the robot's position.
        odom: A subscriber for obtaining odometry data.
        move_pub: A publisher for sending movement commands.
        respawn_goal: An instance of the Respawn class for managing goal respawn.
    """
    def __init__(self):
        self.posex = 0
        self.posey = 0
        self.goal_x = 0
        self.goal_y = 0
        self.initial_goal_distance = 0
        self.prev_distance = 0
        self.prev_heading = 0
        self.get_goalbox = False
        self.init_goal = True
        self.action_space = [-1.5, -0.75, 0, 0.75, 1.5]
        self.reset_robot = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.odom = rospy.Subscriber('/odom', Odometry, self.get_pose)
        self.move_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.respawn_goal = Respawn()


    def get_state(self, data: List[float]) -> Tuple[List[float], bool]:
        """
        Calculate the current state of the environment based on sensor data.

        This method computes the current state of the environment using sensor data 
        and internal state variables. It calculates the heading of the robot, checks
        for collisions with obstacles, and determines if the goal has been reached.

        Parameters:
            data: Sensor data containing information about the environment.

        Returns:
            tuple: A tuple containing the current state of the environment and a flag
            indicating whether the episode is done.

        """
        
        heading = self.heading / (2 * math.pi) + 0.5  # normalised
        done = False

        laser_state = [1.0 if data.ranges[i] > 4 else data.ranges[i] / 4 for i in range(0, 360, 18)]

        if min(laser_state) < settings["COLLISION_DISTANCE"]:
            done = True

        current_distance = self.get_goal_distance()
        if current_distance < settings["TARGET_DISTANCE"]:
            self.get_goalbox = True

        return (laser_state[0:5] + laser_state[-5:] + [heading, current_distance], done)


    def get_pose(self, data_pos: Odometry) -> None:
        """
        Update the robot's position and calculate its heading based on the given
        odometry data.

        Parameters:
            data_pos: Odometry data containing information about the
            robot's position and orientation.
        """
        self.posex = data_pos.pose.pose.position.x
        self.posey = data_pos.pose.pose.position.y

        orientation = data_pos.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.posey, self.goal_x - self.posex)
        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = heading

    def get_goal_distance(self) -> float:
        """
        Calculate the distance between the robot's current position and the goal.

        Returns:
            float: The distance between the robot's current position and the goal.
        """
        return math.hypot(self.goal_x - self.posex, self.goal_y - self.posey)

    def set_reward(self, state: List[float], done: bool, action: int) -> float:
        """
        Calculate the reward based on the current state, action, and episode status.

        Parameters:
            state (List[float]): The current state of the environment.
            done (bool): A flag indicating whether the episode is done.
            action (int): The action taken by the agent.

        Returns:
            float: The reward for the current step.
        """
        current_distance = state[-1]
        current_heading = state[-2]

        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.move_pub.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.set_goal(True, delete=True) # Use for validation only
        elif self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 250
            self.move_pub.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.set_goal(True, delete=True)# Log new goal position in terminal.
        else:
            # ## Reward function with only distance reward
            # reward = 30 * (self.prev_distance - current_distance) if current_distance < self.prev_distance else \
            #     60 * (self.prev_distance - current_distance) if current_distance > self.prev_distance else -1
            # self.prev_distance = current_distance
            
            # Reward function with angle ratio + distance ratio + angle penalty
            angle_ratio = current_heading / math.pi
            angle_reward = 2*(0.5-(angle_ratio**2))-2 

            distance_ratio = current_distance / self.initial_goal_distance
            distance_reward = 2*(1 - math.sqrt(2 * distance_ratio))-2 if distance_ratio>=0 else (-1 - math.sqrt(2 * distance_ratio))

            angle_penalty = 1* math.pi * abs(self.prev_heading - current_heading)
            reward = angle_reward + distance_reward - angle_penalty

            #print(f"angle penalty = {angle_penalty}\n {self.prev_heading, current_heading}")
            self.prev_heading = current_heading 
    
        return reward
    

    def step(self, action: int) -> Tuple[np.array, float, bool, bool]:
        """
        Perform one step in the environment.

        This method executes one step in the environment based on the given action.
        It publishes the action to control the robot's movement, waits for sensor
        data, calculates the current state, computes the reward, and normalizes
        the state before returning it along with the reward, episode completion flag,
        and the status of the goal.

        Parameters:
            action: An integer representing the action to take.

        Returns:
            Tuple: A tuple containing the current state, reward, episode completion flag,
            and the status of the goal.
        """
        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        move = Twist()
        move.linear.x = 0.10 #self.linear_velocity(min(data.ranges))
        move.angular.z = self.action_space[action]
        self.move_pub.publish(move)
        state, done = self.get_state(data)
        reward = self.set_reward(state, done, action)


        state[-1] /= 4  # normalising distance
        return np.array(state, dtype=np.float32), reward, done, self.get_goalbox

    def reset(self) -> np.array:
        """
        Reset the environment to its initial state.

        This method resets the environment to its initial state by resetting
        the robot's position, obtaining initial sensor data, setting the initial
        goal position, calculating the initial goal distance, and normalizing
        the distance in the state before returning it.        

        Returns:
            numpy.ndarray: The initial state of the environment.
        """
        self.get_goalbox = False
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_robot()
        except rospy.ServiceException:
            rospy.loginfo("gazebo/reset_world service call failed")
        rospy.loginfo("reset")
        data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

        if self.init_goal:
            self.goal_x, self.goal_y = self.respawn_goal.set_goal(position_check=True)
            self.init_goal = False

        state, done = self.get_state(data)
        self.initial_goal_distance = self.get_goal_distance()
        #self.prev_distance = self.goal_distance
        self.prev_heading = state[-2]
     


        state[-1] /= 4  # normalising distance
        return np.array(state, dtype=np.float32)
    
    def linear_velocity(self, distance_to_obstacle: float) -> float:
        max_velocity = 0.22  # Maximum velocity when far from obstacles
        min_velocity = 0.05  # Minimum velocity when near obstacles
        max_distance = 1.0  # Distance at which velocity is max_velocity

        velocity = np.sqrt(distance_to_obstacle) * (max_velocity - min_velocity) + min_velocity
        # Clip velocity to ensure it's within the specified range
        velocity = np.clip(velocity, min_velocity, max_velocity)

        return velocity

        
# action = [0.15,0]
# #testing environment
# if __name__ == '__main__':
#     rospy.init_node('env',anonymous=True)
#     for i in range(10):
#         total_reward = 0
#         a= Env()
#         done = False
        
#         print("reset")
#         while not done:
#             state, reward ,done , goal_reached= a.step(action)
#             print(len(state))
#             total_reward += reward
#             new_line = "\n"
#             print(f"reward = {reward}{new_line} total_reward = {round(total_reward,2)}{new_line}{goal_reached}{new_line}")
#             if goal_reached:
#                 break

#         b = a.reset()




    
        


