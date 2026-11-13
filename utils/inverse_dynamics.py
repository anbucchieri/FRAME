"""
ID_Xsens -- inverse kinematics (Pink, differential IK on Pinocchio) of the person-scaled DHM
against an Xsens movement CSV (*_com_mov.csv), plus the helpers inverse dynamics needs.

frame-kinematics drives it through the `IKEngine` protocol (frame_kinematics/interfaces.py).
"""

import numpy as np
import pandas as pd
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask

import utils.process_mvnx_file as process_mvnx_file
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d  

import os
import pickle

class ID_Xsens:
    def __init__(self, urdf_file_path, input_mocap_file):
        # Create model from URDF
        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_file_path,
            package_dirs=["."],
            root_joint=pin.JointModelFreeFlyer(),
            verbose=True,
        )

        # Set initial robot configuration
        self.configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)

        # Convert mvnx to csv movement file and get capture rate
        # self.input_mocap_file, self.xsens_sample_rate = process_mvnx_file.write_Xsens_movement_data_to_csv(mvnx_file_path)
        self.input_mocap_file = input_mocap_file
        #TODO do we need to filter xsens position/orientation data? Does not seem needed when looking at few input files

        # Define start and end frames for IK
        self.start_frame = 0
        self.end_frame = 0
        
        self.tasks = {
            "pelvis": FrameTask(
                "pelvis",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "lumbar_spine": FrameTask(
                "lumbar_spine",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "thoracic_spine": FrameTask(
                "thoracic_spine",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "neck": FrameTask(
                "neck",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "head": FrameTask(
                "head",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=5.0,  # [cost] / [rad]
            ),
            "right_arm": FrameTask(
                "right_arm",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=20.0,  # [cost] / [rad]
            ),
            "right_forearm": FrameTask(
                "right_forearm",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=20.0,  # [cost] / [rad]
            ),
            "right_palm": FrameTask(
                "right_palm",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=5.0,  # [cost] / [rad]
            ),
            "left_arm": FrameTask(
                "left_arm",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=20.0,  # [cost] / [rad]
            ),
            "left_forearm": FrameTask(
                "left_forearm",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=20.0,  # [cost] / [rad]
            ),
            "left_palm": FrameTask(
                "left_palm",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=5.0,  # [cost] / [rad]
            ),
            "right_thigh": FrameTask(
                "right_thigh",
                position_cost=5.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "right_shank": FrameTask(
                "right_shank",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "right_foot": FrameTask(
                "right_foot",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=5.0,  # [cost] / [rad]
            ),
            "left_thigh": FrameTask(
                "left_thigh",
                position_cost=5.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "left_shank": FrameTask(
                "left_shank",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "left_foot": FrameTask(
                "left_foot",
                position_cost=20.0,  # [cost] / [m]
                orientation_cost=5.0,  # [cost] / [rad]
            ),
            "right_clavicular": FrameTask(
                "right_clavicular",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            ),
            "left_clavicular": FrameTask(
                "left_clavicular",
                position_cost=10.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
            )
        }
        
        # Add clavicular task if using a human model with a clavicular
        for frame in self.robot.model.frames:
            if frame.name == 'right_clavicular':
                self.tasks["right_clavicular"] = FrameTask("right_clavicular", position_cost=5.0, orientation_cost=1.0)
            elif frame.name == 'left_clavicular':
                self.tasks["left_clavicular"] = FrameTask("left_clavicular", position_cost=5.0, orientation_cost=1.0)            

        # Configure IK solver
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

        self.frequency = 500
        rate = RateLimiter(self.frequency)
        self.dt = rate.period

        self.nb_iter_solver = 500 # For first iteration, we don't have a good initial guess (configuration), so we need several differential IK iterations to converge

        # Define mapping between DHM and Xsens segments
        self.dhm_2_xsens = {"pelvis":"Pelvis",
                  "lumbar_spine":"L5",
                  "thoracic_spine":"T8", 
                  "neck":"Neck",
                  "head":"Head",
                  "right_clavicular":"RightShoulder",
                  "right_arm":"RightUpperArm",
                  "right_forearm":"RightForeArm",
                  "right_palm":"RightHand",
                  "left_clavicular":"LeftShoulder",
                  "left_arm":"LeftUpperArm",
                  "left_forearm":"LeftForeArm",
                  "left_palm":"LeftHand",
                  "right_thigh":"RightUpperLeg",
                  "right_shank":"RightLowerLeg",
                  "right_foot":"RightFoot",
                  "left_thigh":"LeftUpperLeg",
                  "left_shank":"LeftLowerLeg",
                  "left_foot":"LeftFoot",
                  }

        self.velocity = []
        self.q_revised = np.zeros((self.robot.nv,))  # Define q_revised as a 1D array
        self.q_prev = self.q_revised.copy()
        self.q_prev2 = self.q_prev.copy()
       
        self.body_pose={ta:{} for ta in self.tasks}
        self.body_positions={ta:[] for ta in self.tasks}
        self.body_orientations={ta:[] for ta in self.tasks}
        self.angular_velocity = {ta:[] for ta in self.tasks}
        self.angular_acceleration = {ta:[] for ta in self.tasks}
        self.linear_acceleration = {ta:[] for ta in self.tasks}
        
        self.saved_configurations = []
        self.saved_prev = []
        self.saved_velocity = []
        self.saved_accelerations = []
        self.com = []
        self.acom = []
        self.cutoff = 15
        self.fs = self.frequency
        self.xsens_frequency = 240
        self.division_factor = 1
        self.joint_names = list(self.robot.model.names)[2:] 
        
    def save_log_data(self, log_torque, log_q, log_pose, filename):
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'wb') as file:
            pickle.dump((log_torque, log_q, log_pose), file)    
            
    def butterworth_filter(self, data, order=4, filter_type='low'):
        """
        Apply a Butterworth filter to the data.

        Parameters:
        - data: array-like, the input signal to filter.
        - cutoff: float, the cutoff frequency (Hz) for the filter.
        - fs: float, the sampling frequency of the data (Hz).
        - order: int, the order of the Butterworth filter (default is 4).
        - filter_type: str, 'low', 'high', 'bandpass', or 'bandstop' (default is 'low').

        Returns:
        - filtered_data: array-like, the filtered signal.
        """
        nyquist = 0.5 * self.fs  # Nyquist frequency
        normalized_cutoff = self.cutoff / nyquist  # Normalize the cutoff frequency
        b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data   
    
    def resample_xsens_data(self):
        original_time = np.arange(len(self.data_xsens)) / self.xsens_frequency
        target_duration = original_time[-1]
        
        new_num_samples = int(np.round(target_duration * self.frequency)) + 1
        new_time = np.linspace(0, target_duration, new_num_samples)
        
        # Interpolate each column
        resampled_dict = {}
        for col in self.data_xsens.columns:
            f_interp = interp1d(original_time, self.data_xsens[col], kind='linear', fill_value="extrapolate")
            resampled_dict[col] = f_interp(new_time)
            
        
        resampled_data = pd.DataFrame(resampled_dict)
        resampled_data['Time'] = new_time  # optional, to track time in output
        
        self.data_xsens = resampled_data
        
        # Update end frame
        # self.end_frame = len(self.data_xsens)
        # print(f"RESHAPE XSENS DATA: {self.end_frame}")
        return len(self.data_xsens)
    
    def uploadMocapFile(self):
        self.data_xsens = self.input_mocap_file #pd.read_csv(self.input_mocap_file)
        if self.data_xsens.empty :
            print("Failed to load data")
            
        # Update end frame
        if self.end_frame == 0:
            self.end_frame = len(self.data_xsens)
   
    def getStartFrame(self):
        return int(self.start_frame)

    def getEndFrame(self):
        return int(self.end_frame)

    def getSampleRate(self):
        return self.xsens_sample_rate

    def getMocapFileCSV(self):
        return self.input_mocap_file

    def setStartFrame(self, start_frame):
        self.start_frame = start_frame

    def setEndFrame(self, end_frame):
        self.end_frame = end_frame

    def setNbIterSolver(self, nb_iter_solver):
        self.nb_iter_solver = nb_iter_solver

    def getConfiguration(self):
        return self.configuration.q

    def getConfigurationDict(self):
        config = dict()
        # joint_names = list(self.robot.model.names)[2:] # remove 'universe' and 'root_joint'

        config['root_joint'] = np.array(self.configuration.q[0:7])

        for j in self.joint_names:
            config[j] = float(self.configuration.q[(self.robot.model.getJointId(j)-2)+7]) # -2 to remove 'universe' and 'root_joint' index; +7 corresponds to pos+quat of the root

        return config

    def computeIK(self, timeframe):        
        if timeframe < self.start_frame or timeframe > self.end_frame:
            raise ValueError("Wrong timeframe for IK")

        if self.data_xsens.empty:
            raise ValueError("No mocap file loaded")
        
        for task in self.tasks:
            converted_task = self.dhm_2_xsens[task]
            columns_matching_task = [col for col in self.data_xsens.columns if col.startswith(converted_task) and (col.endswith('_X') or col.endswith('_Y') or col.endswith('_Z')) and '_q' not in col]
            target_translation = np.array([self.data_xsens.loc[timeframe, col] for col in columns_matching_task])
            
            columns_matching_task_ori = [col for col in self.data_xsens.columns if col.startswith(converted_task) and col.endswith('_q')]
            target_orientation = np.array([self.data_xsens.loc[timeframe, col] for col in columns_matching_task_ori])
            quaternion_rearranged = target_orientation[[1, 2, 3, 0]] # Xsens uses (qw, qx, qy, qz) but Pinocchio uses (qx, qy, qz, qw)
            target_rotation = pin.Quaternion(quaternion_rearranged).matrix()
            
            # Update target
            target = pin.SE3(target_rotation, target_translation)
            self.tasks[task].set_target(target)

        for step in range(self.nb_iter_solver):
            self.velocity = solve_ik(self.configuration, self.tasks.values(), self.dt, solver=self.solver)             
            self.configuration.integrate_inplace(self.velocity, self.dt)

        # self.viz.display(self.configuration.q)
        self.configuration.update(self.configuration.q) # update -> runs forward kinematics and computes collision-pair distances, if applicable
        
    def getVelAcc(self, timeframe):
        quat_q = self.configuration.q[3:7]
        rot_q =  pin.Quaternion(quat_q).matrix()
        euler_q = pin.rpy.matrixToRpy(rot_q)
    
        self.q_revised[:3] = self.configuration.q[:3]
        self.q_revised[3:6] = euler_q
        self.q_revised[6:] = self.configuration.q[7:]
        
        ddq = (self.q_revised - 2 * self.q_prev + self.q_prev2) / (self.dt * self.dt)

        self.configuration.update(self.configuration.q) # update -> runs forward kinematics and computes collision-pair distances, if applicable
        frame_names = [frame.name for frame in self.robot.model.frames]
        # frame_ids = [self.robot.model.getFrameId(n) for n in frame_names]
        
        pin.forwardKinematics(self.robot.model, self.robot.data, self.configuration.q, self.velocity, ddq)
        target_index = timeframe - self.start_frame
        for ta in self.tasks:
            frame_id = self.robot.model.getFrameId(ta)
            position = self.robot.framePlacement(self.configuration.q, frame_id).translation
            rotation = self.robot.framePlacement(self.configuration.q, frame_id).rotation
            quaternion = pin.Quaternion(rotation).coeffs() #[x, y, z, w]
           
            # get angular and linear velocities of body frames 
            v_frame = pin.getFrameVelocity(self.robot.model, self.robot.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
            # a_frame = pin.getFrameAcceleration(self.robot.model, self.robot.data, frame_id, pin.WORLD)
            a_frame = pin.getFrameClassicalAcceleration(self.robot.model, self.robot.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        
            self.angular_velocity[ta].append(v_frame.angular)
            self.angular_acceleration[ta].append(a_frame.angular)
            self.linear_acceleration[ta].append(a_frame.linear)
            
            self.body_pose[ta][target_index] = np.concatenate((position, quaternion))
            self.body_positions[ta].append(position)
            self.body_orientations[ta].append(rotation)
        
        self.saved_configurations.append(self.configuration.q) 
        self.saved_velocity.append(self.velocity)
        self.saved_accelerations.append(ddq)
        
        pin.forwardKinematics(self.robot.model, self.robot.data, self.configuration.q, self.velocity, ddq)
        pin.centerOfMass(self.robot.model, self.robot.data, self.configuration.q, self.velocity, ddq)
        
        temp = np.array(self.robot.data.com[0])
        self.com[target_index, :] = temp
        temp = np.array(self.robot.data.acom[0])
        self.acom[target_index, :] = temp
        
        
        
        # self.acom.append(self.robot.data.acom[0])   
        # acom_local = self.robot.data.acom #wrt joint local reference frame
        # acom_world = acom_local
        # for j in range(2, np.array(acom_local).shape[0]):
        #     acom_world[j] = self.robot.data.oMi[j].rotation.dot(acom_local[j])
            
        # self.acom.append(acom_world) 
        
    def runFullIK(self, output_file_path_segments=None, output_file_path_joints=None):
        self.com = np.zeros((self.end_frame-self.start_frame, 3))
        self.acom = np.zeros((self.end_frame-self.start_frame, 3))
        
        for timeframe in range(self.start_frame, self.end_frame):
            
            if timeframe > self.start_frame: 
                    self.setNbIterSolver(1) # Several iterations are needed for IK to converge only for 1st frame (after we start from a good initial guess to have good results in one iteration)
            self.q_prev = self.q_revised
            self.q_prev2 = self.q_prev
            
            self.computeIK(timeframe)
            self.getVelAcc(timeframe)
    
        log_q = self.temp_save_log_q()
        return log_q
    
    def skew_matrix(self, v):
        """Returns the skew symmetric matrix of a 3D vector."""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])

    def get_wrench_transform(self, TF: np.array):
    
        pos_dst_in_src = TF[:3, 3]
        rot_dst_in_src = TF[:3, :3]

        transform = np.zeros([6, 6])

        transform[0:3, 0:3] = rot_dst_in_src
        transform[3:6, 0:3] = self.skew_matrix(pos_dst_in_src) @ rot_dst_in_src
        transform[3:6, 3:6] = rot_dst_in_src

        return transform

    def temp_save_log_q(self):
        log_q = {name: [] for name in self.robot.model.names[2:]}
        saved_configurations = np.array(self.saved_configurations)
        names_no_rj = self.robot.model.names[2:]
        names_no_rj_list = list(names_no_rj)
        saved_configurations_no_rj = saved_configurations[:, 7:]
    
        
        for time_instant in range(saved_configurations_no_rj.shape[0]):
            for name in names_no_rj_list:
                joint_index = names_no_rj_list.index(name)  # Retrieve the column index for the joint name
                log_q[name].append(saved_configurations_no_rj[time_instant, joint_index])
                
        return log_q
                
        
        
    def runInverseDynamics(self, reference, force_RX, force_LX, moment_RX, moment_LX, log_path):        
        log_torque = {name: [] for name in self.robot.model.names[2:]}
        log_q = {name: [] for name in self.robot.model.names[2:]}
        log_pose = {ta:{} for ta in self.tasks}
    
        saved_configurations = np.array(self.saved_configurations)
        saved_velocity = np.array(self.saved_velocity)            
        saved_accelerations = np.array(self.saved_accelerations)
        
        q_filtered = np.zeros((saved_configurations.shape[0], saved_configurations.shape[1]))
        dq_filtered = np.zeros((saved_velocity.shape[0], saved_velocity.shape[1]))
        ddq_filtered = np.zeros((saved_accelerations.shape[0], saved_accelerations.shape[1]))
        
        for j in range(saved_configurations.shape[1]):
            q_filtered[:, j] = self.butterworth_filter(saved_configurations[:, j])
            
        for j in range(saved_velocity.shape[1]):
            dq_filtered[:, j] = self.butterworth_filter(saved_velocity[:, j])
            ddq_filtered[:, j] = self.butterworth_filter(saved_accelerations[:, j])
            
        # # === Calculate external wrenches due to contact with the ground === #
        q =[]
        dq=[]
        ddq=[]
        saved_torques = np.zeros((ddq_filtered.shape[0], ddq_filtered.shape[1]))
        external_wrenches_R = []
        external_wrenches_L = []
        
        feet_names = ['right_foot', 'left_foot']        
        feet_ids = [self.robot.model.getFrameId(n) for n in feet_names]
        
        # print(feet_ids)
        # for frame in self.robot.model.frames:
        #     print(f"{frame.name} : {self.robot.model.getFrameId(frame.name)}")
       
        ##############################################################################################
        for i in range(ddq_filtered.shape[0]):   
            q = q_filtered[i, :]
            dq = dq_filtered[i, :]
            ddq = ddq_filtered[i, :]
            
            idx = i # test static pose
            force_R = force_RX[idx, :]
            moment_R = moment_RX[idx, :]
            force_L = force_LX[idx, :]
            moment_L = moment_LX[idx, :]  
            
            ##################### computeFrameJacobian() ###########################################
            # The result of this function is equivalent to call first computeJointJacobians(model,data,q), 
            # then updateFramePlacements(model,data) and then call getJointJacobian(model,data,jointId,LOCAL,J), 
            # but forwardKinematics and updateFramePlacements are not fully computed. 
            
            pin.forwardKinematics(self.robot.model, self.robot.data, q)
            pin.updateFramePlacements(self.robot.model, self.robot.data) 
       
            J_R = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, feet_ids[0], reference)
            J_L = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, feet_ids[1], reference)
            
            w_R = pin.Force(force_R, moment_R)
            w_L = pin.Force(force_L, moment_L)
   
            taw_R = J_R.T @ w_R 
            taw_L = J_L.T @ w_L
            
            # === Calculate torques at the joints === #
            external_wrenches_R.append(taw_R)
            external_wrenches_L.append(taw_L)
            
            torques = pin.rnea(self.robot.model, self.robot.data, q, dq, ddq) 
            
            saved_torques[i, :] = torques - (taw_R + taw_L)   # TODO: CHECK IF ITS MINUS OR PLUS
        
        names_no_rj = self.robot.model.names[2:]
        names_no_rj_list = list(names_no_rj)
        saved_configurations_no_rj = saved_configurations[:, 7:]
        torques_no_rj = saved_torques[:, 6:]
        
        for time_instant in range(torques_no_rj.shape[0]):
            for name in names_no_rj_list:
                joint_index = names_no_rj_list.index(name)
                log_torque[name].append(torques_no_rj[time_instant, joint_index])
        
        for time_instant in range(saved_configurations_no_rj.shape[0]):
            for name in names_no_rj_list:
                joint_index = names_no_rj_list.index(name)  # Retrieve the column index for the joint name
                log_q[name].append(saved_configurations_no_rj[time_instant, joint_index])
                
        log_pose = self.body_pose  
        
        self.save_log_data(log_torque, log_q, log_pose, log_path)
