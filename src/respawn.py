#!/usr/bin/env python

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from json import load
from typing import Tuple 

# Load settings from settings.json.
settings_path = os.path.dirname(os.path.realpath(__file__)) 
with open (settings_path+'/settings.json', 'r') as f:
    settings = load(f)

class Respawn():
    """
    Class for managing respawn and deletion of models in Gazebo.

    Attributes:
        model_path (str): The path to the SDF model file.
        model_name (str): The name of the model to be spawned.
        goal_position (Pose): The position where the goal model will be spawned.
        init_goal_x (float): Initial x-coordinate of the goal position.
        init_goal_y (float): Initial y-coordinate of the goal position.
        index (int): Index used for selecting random goals.
        last_index (int): Index of the last selected goal.
        sub_model: Subscriber for model states topic.
        check_model (bool): Flag to check if the model exists in Gazebo.

    Methods:
        checkModel: Check if the model exists in Gazebo.
        respawn_model: Respawn the model in Gazebo.
        delete_model: Delete the model from Gazebo.
        set_goal: Set a new goal position for the model.
    """

    def __init__(self):
        self.model_path = os.path.dirname(os.path.realpath(__file__))
        self.model_path_goal = self.model_path.replace('/kanqn/src', '/dqn_turtlebot3/worlds/model.sdf')
        self.f_goal = open(self.model_path_goal, 'r')
        self.model_goal = self.f_goal.read()
        self.model_name = "goal"
        self.goal_position = Pose()
        self.init_goal_x = 1.0
        self.init_goal_y = 1.0
        self.last_goal_x = 0.0
        self.last_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.index = 0
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.stage_number = 3

    def checkModel(self, model: ModelStates) -> None:
        """
        Check if the model exists in Gazebo.

        Args:
            model (ModelStates): Current state of models in Gazebo.
        """
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawn_model(self) -> None:
        """Respawn the model in Gazebo."""
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                try:
                    spawn_model_prox("goal", self.model_goal, 'robotos_name_space', self.goal_position, "world")
                except rospy.ServiceException:
            	    rospy.loginfo("gazebo/spawn_sdf_model for goal service call failed")
                rospy.loginfo(f"Goal position : {self.goal_position.position.x:.1f}, {self.goal_position.position.y:.1f}")
                break
            else:
                pass

    def delete_model(self) -> None:
        """Delete the model from Gazebo."""
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                try:
                    del_model_prox("goal")
                except rospy.ServiceException:
                    rospy.loginfo("gazebo/delete_model service call failed") 
                break
            else:
                pass

    def set_goal(self, position_check: bool = False, delete: bool = False) -> Tuple[float, float]:
        """
        Set a new goal position for the model.

        Args:
            position_check (bool): Flag to check if the position is valid.
            delete (bool): Flag to delete the current model.

        Returns:
            tuple: New goal position (x, y).
        """
        if delete:
            self.delete_model()

        if self.stage_number == 3:
            while position_check:
                goal_x = random.randrange(-16, 17) / 10.0
                goal_y = random.randrange(-16, 17) / 10.0
                # Check if the goal position is not origin and was not near the previous goal position.
                
                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True
                elif 0 <= abs(goal_x) <= 1.0 or 0 <= abs(goal_y) <= 1.0:   
                    position_check = True
                else:
                    position_check = False
                # Check if the goal position is not in obstacle positions.
                

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        if self.stage_number == 4:
            while position_check:
                goal_x = 2.3
                goal_y =-4.5
            
                position_check = False

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        while position_check:
            goal_x_list = settings["GOAL_LIST_X"]
            goal_y_list = settings["GOAL_LIST_Y"]

            self.index = random.randrange(0, len(goal_x_list))

            if self.last_index == self.index:
                position_check = True
            else:
                self.last_index = self.index
                position_check = False

            self.goal_position.position.x = goal_x_list[self.index]
            self.goal_position.position.y = goal_y_list[self.index]

        time.sleep(0.5)
        self.respawn_model()
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y

