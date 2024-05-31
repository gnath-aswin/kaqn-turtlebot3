#!/usr/bin/env python
import rospy
import math
from typing import Tuple, List
from statistics import mean
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry

class Evaluation():
    """
    Class for evaluating robot performance.

    Methods:
        get_time: Get current time from ROS Clock message.
        get_pose: Get current robot position from ROS Odometry message.
        mean_distance: Calculate mean distance traveled by the robot.
        mean_time: Calculate mean time taken for each evaluation cycle.
    """

    def get_time(self) -> int:
        """
        Get current time from ROS Clock message.

        Returns:
            int: Current time in seconds.
        """
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('clock', Clock, timeout=5)
            except rospy.exceptions.ROSException:
                pass
        time = data.clock.secs
        return time

    def get_pose(self) -> Tuple[float, float]:
        """
        Get current robot position from ROS Odometry message.

        Returns:
            tuple: Current position (x, y).
        """
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('odom', Odometry, timeout=5)
            except rospy.exceptions.ROSException:
                pass
        posx, posy = data.pose.pose.position.x, data.pose.pose.position.y
        return posx, posy
    
    def mean_distance(self, pos_list: List[List[Tuple[float, float]]]) -> Tuple[List[float], float]:        
        """
        Calculate mean distance traveled by the robot.

        Args:
            pos_list (list): List of position sublists.

        Returns:
            tuple: List of total distances for each sublist and the mean distance.
        """
        distance = []
        for sublist in pos_list:
            total_distance = sum(math.dist(value, sublist[idx + 1]) for idx, value in enumerate(sublist[:-1]))
            distance.append(total_distance)
        mean_dist = mean(distance)

        return distance, mean_dist

    def mean_time(self, time_list: List[List[int]]) -> float:
        """
        Calculate mean time taken for each evaluation cycle.

        Args:
            time_list (list): List of time sublists.

        Returns:
            float: Mean time taken.
        """
        total_time = 0
        num_sublists = len(time_list)
        for sublist in time_list:
            for idx, value in enumerate(sublist[:-1]):
                total_time += sublist[idx + 1] - value

        mean_time = total_time / (num_sublists)
        return mean_time

    def coefficient_of_variation(self, rewards: List[float]) -> float:
        """
        Calculate coefficient of variation of the rewards.  

        Args:
        rewards (list): List of rewards.

        Returns:
        float: Coefficient of variation.
        """
        cov = math.sqrt(mean([((reward - mean(rewards))**2) for reward in rewards])) / abs(mean(rewards))
        return cov