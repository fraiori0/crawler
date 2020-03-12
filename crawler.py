import pybullet as p
import numpy as np
import scipy as sp
import scipy.linalg as lna
import time
import pybullet_data
from math import *

class Discrete_Low_Pass:
    def __init__(self, dim, dt, fc, K=1):
        #fc is the cut-off frequency
        #since each filter keeps a internal state, a different filter should 
            # be intialized for each signal to filter
        self.dim = dim
        self.x = np.array([0]*self.dim)
        self.dt = dt
        self.fc = fc
        self.K = K
    def reset(self):
        self.x = np.array([0]*self.dim)
    def filter(self, signal):
        # input signal should be a NUMPY ARRAY
        self.x = (1-self.dt*self.fc)*self.x + self.K*self.dt*self.fc * signal
        return self.x

class Integrator_Forward_Euler:
    def __init__(self, dt, x0):
        self.dt = dt
        self.x = np.array(x0)
    def integrate(self,xd):
        #xd should be passed as NUMPY ARRAY
        self.x = self.x + xd*self.dt
        return self.x
    def reset(self, x0):
        self.x = np.array(x0)

class Crawler:

    def __init__(self, dt_simulation, urdf_path="/home/fra/Uni/Tesi/crawler", base_position=[0,0,0.2], base_orientation=[0,0,0,1]):
        self.scale=1
        #NOTE: Properties in this block of code must be manually matched to those defined in the Xacro file
        self.spine_segments         = 4
        self.body_length            = self.scale * 0.5
        self.spine_segment_length   = self.scale * self.body_length/self.spine_segments
        self.leg_length             = self.scale * self.body_length/8
        self.body_sphere_radius     = self.scale * self.spine_segment_length/2
        self.foot_sphere_radius     = self.scale * self.body_sphere_radius/3
        #
        self.Id = p.loadURDF("%s/crawler.urdf" % urdf_path, base_position, base_orientation, globalScaling=self.scale)
        self.num_joints = p.getNumJoints(self.Id)
        self.mass = 0
        for i in range(-1,self.num_joints):
            self.mass += p.getDynamicsInfo(self.Id, i)[0]
        #Value of the girdle flexion angle for having symmetric contact with both feet and the girdle collision sphere 
            #slightly reduced to avoid requiring compenetration when setting foot constraints and using position control to keep this value
        self.neutral_contact_flexion_angle = asin((self.body_sphere_radius-self.foot_sphere_radius)/self.leg_length)-0.01
        self.control_indices = self.generate_control_indices()
        self.constraints = {
            "right_foot": 0,
            "left_foot": 0
        }
        #NOTE: joints_state_array doesn't include the state of the base to keep the correct id-numbers of the links
        self.links_state_array=[0]*self.num_joints
        self.set_links_state_array()
        #
        self.COM_y_0=self.COM_position_world()[1]
        #
        self.mask_base = [0,1,2,3,4,5]
        self.mask_joints = list(range(6,6+self.num_joints))
        self.mask_act = list(np.arange(6, 6+self.num_joints-4,2))
        self.mask_nact = list(np.arange(7, 6+self.num_joints-4,2))+list(range(6+self.num_joints-4,6+self.num_joints))
        self.mask_right_girdle = [6+self.num_joints-4, 6+self.num_joints-3]
        self.mask_left_girdle = [6+self.num_joints-2, 6+self.num_joints-1]
        #
        self.dt_simulation=dt_simulation
        self.low_pass_lateral = Discrete_Low_Pass(dim=len(self.mask_act),dt=self.dt_simulation, fc=50*self.dt_simulation, K=1)
        self.integrator_lateral = Integrator_Forward_Euler(self.dt_simulation,[0]*len(self.mask_act))
        
    def set_low_pass_lateral(self, fc, K=1):
        self.low_pass_lateral = Discrete_Low_Pass(
            dim=len(self.mask_act),
            dt=self.dt_simulation, 
            fc=fc*self.dt_simulation, 
            K=K)
            
    def COM_position_world(self):
    #return the position of the center of mass in the world coordinates, as a NUMPY ARRAY
        COM = np.asarray(p.getBasePositionAndOrientation(self.Id)[0])*(p.getDynamicsInfo(self.Id, -1)[0])
        for i in range(0,self.num_joints):
            link_COM_pos = np.asarray(p.getLinkState(self.Id, i)[0])
            link_mass = p.getDynamicsInfo(self.Id, i)[0]
            COM += link_COM_pos*link_mass
        COM = (COM/self.mass)
        return COM

    def COM_velocity_world(self):
    #return the linear velocity of the center of mass in the world coordinates, as a NUMPY ARRAY
        COMv = np.asarray(p.getBaseVelocity(self.Id)[0])*(p.getDynamicsInfo(self.Id, -1)[0])
        for i in range(0,self.num_joints):
            link_COM_vel = np.asarray(p.getLinkState(self.Id, i, computeLinkVelocity=1)[6])
            link_mass = p.getDynamicsInfo(self.Id, i)[0]
            COMv += link_COM_vel*link_mass
        COMv = (COMv/self.mass)
        return COMv

    def set_COM_y_0(self):
        self.COM_y_0 = self.COM_position_world()[1]
        return

    def turn_off_joint(self, joint_index):
        p.setJointMotorControl2(self.Id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0)
        return

    def turn_off_crawler(self):
        for i in range(0,p.getNumJoints(self.Id)):
            p.setJointMotorControl2(self.Id, i, controlMode=p.VELOCITY_CONTROL, force=0)
        return

    def generate_control_indices(self):
    #this function relies on knowledge of the order of the joints in the crawler model
    #if the URDF is modified in ways different than just amodeldding more segments to the spine this function should be updated properly
        lat_joints_i = tuple(range(0,(self.num_joints-4),2))
        r_girdle_abd_i = self.num_joints-4
        r_girdle_flex_i = self.num_joints-3
        l_girdle_abd_i = self.num_joints-2
        l_girdle_flex_i = self.num_joints-1
        return (lat_joints_i,r_girdle_abd_i,r_girdle_flex_i,l_girdle_abd_i,l_girdle_flex_i)

    def fix_right_foot(self, leg_index=17):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["right_foot"]:
            constId=self.constraints["right_foot"]
            print("Error: remove right foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, leg_index, -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,leg_index)[0]))
            self.constraints["right_foot"]=constId
        return constId

    def fix_left_foot(self, leg_index=19):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["left_foot"]:
            constId=self.constraints["left_foot"]
            print("Error: remove left foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, leg_index, -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,leg_index)[0]))
            self.constraints["left_foot"]=constId
        return constId
    
    def free_right_foot(self):
        p.removeConstraint(self.constraints["right_foot"])
        self.constraints["right_foot"]=0
        return
    def free_left_foot(self):
        p.removeConstraint(self.constraints["left_foot"])
        self.constraints["left_foot"]=0
        return        

    def set_feet_constraints(self, RL=(False,False)):
        if RL[0]:
            self.fix_right_foot()
        elif not (self.constraints["right_foot"]):
            #NOTE: careful to not switch the order of this operations
            self.free_right_foot()
        if RL[1]:
            self.fix_left_foot()
        elif not (self.constraints["left_foot"]):
            self.free_left_foot()
        return

    def set_links_state_array(self):
        #NOTE: this doesn't include the state of the base to keep the correct id-numbers of the links
        for i in range(0,self.num_joints):
            link_state = p.getLinkState(self.Id,
                i,
                computeLinkVelocity=1,
                computeForwardKinematics=1)
            self.links_state_array[i]={"world_com_trn":link_state[0],
                "world_com_rot":link_state[1],
                "loc_com_trn":link_state[2],
                "loc_com_rot":link_state[3],
                "world_link_pos":link_state[4],
                "world_link_rot":link_state[5],
                "world_com_vt":link_state[6],
                "world_com_vr":link_state[7]
                }
        return    

    def get_joints_pos_tuple(self):
        return list(zip(*(p.getJointStates(self.Id,list(range(0,self.num_joints))))))[0]

    def get_joints_speeds_tuple(self):
        return list(zip(*(p.getJointStates(self.Id,list(range(0,self.num_joints))))))[1]
    
    def get_base_Eulers(self):
        return  p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.Id)[1])
    
    def get_R_base_to_world(self):
        # returned as a (3,3) NUMPY ARRAY
        # R * x_base = x_world
        quat_base = p.getBasePositionAndOrientation(self.Id)[1]
        R = np.reshape(np.array(p.getMatrixFromQuaternion(quat_base)),(3,3))
        return R
    
    def get_link_COM_jacobian_world(self,link_index, R, joints_pos, link_state_set=False):
        #NOTE: p.calculateJacobian outputs a Jacobian that gives the linear/angular velocity expressed
            # in a reference frame oriented like the base but fixed in the global reference frame
            # To get the velocities in the global reference frame we should transform the Jacobian (see report)
        # R and joint_pos should be passed already computed (through the proper class' methods) to avoid recomputing them
            # even if the simulation has not stepped between two calls and the state is still the same
        # For the same reason self.set_links_state_array() should be called once (every time-step) just before using this function
        if not link_state_set:
            self.set_links_state_array()
        ###
        Ji = np.asarray(
            p.calculateJacobian(self.Id,
                link_index,
                self.links_state_array[link_index]["loc_com_trn"],
                joints_pos,
                [0.0]*(self.num_joints),
                #self.get_joints_speeds_tuple(),
                [0.0]*(self.num_joints))
            )
        Jt_i = Ji[0]
        Jtbrw_i = (R.dot(Jt_i[:,[0,1,2]])).dot(R.T)
        Jtqw_i = R.dot(Jt_i[:,self.mask_joints])
        Jtw_i = np.concatenate((Jtbrw_i, Jt_i[:,[3,4,5]], Jtqw_i),1)
        ### rotation Jacobian
        # Jr_i = Ji[1]
        # Jrqw_i = R.dot(Jr_i[:,self.mask_joints])
        # Jrw_i = np.concatenate((Jr_i[:,self.mask_base], Jrqw_i),1)
        ###
        #returned as NUMPY ARRAY
        return Jtw_i

    def COM_trn_jacobian(self):
        #Jacobian of the COM is computed as the weighted mean (with respect to masses) of the Jacobian of the links
        #The transational jacobian of the base is just the identity matrix multiplying the base translational velocity
            # plus null terms associated to the base angular speed and the joints speed terms (thus 3+self.num_joints)
            # NOTE: angular speed values are null just because the COM of the girdle is at the origin of the link frame
        self.set_links_state_array()
        joints_pos = self.get_joints_pos_tuple()
        R = self.get_R_base_to_world()
        ###
        Jbase_t = np.asarray([  [0.0]*3 + [1.0,0.0,0.0] + [0.0]*(self.num_joints),
                                [0.0]*3 + [0.0,1.0,0.0] + [0.0]*(self.num_joints),
                                [0.0]*3 + [0.0,0.0,1.0] + [0.0]*(self.num_joints) ])
        JM_t = Jbase_t*(p.getDynamicsInfo(self.Id,-1)[0])
        ###
        for i in range(0,self.num_joints):
            Jtw_i = self.get_link_COM_jacobian_world(i, R=R, joints_pos=joints_pos, link_state_set=True)
            JM_t += Jtw_i * (p.getDynamicsInfo(self.Id,i)[0])
        ###
        JM_t = JM_t/self.mass
        #returned as NUMPY ARRAY
        return JM_t

    def solve_null_COM_y_speed_optimization(self, K, k0=1):
        #Return the desired joint speeds of the spinal lateral joints to be used for velocity control
        #NOTE: self.COM_y_0 should be set once at the start of each step phase
        #NOTE: since the dorsal joints of the spine and the DOFs of the base are not actuated, xd_desired 
            # is corrected (see report). Joints of the girdles are also not considered as actuated since their 
            # speed is set independently
        #Use constrained (convex) optimization to solve inverse kinematic, coupled with closed loop inverse kinematic
            #to (try to) make the error converge esponentially
        #q0d are projected inside the nullspace of the Jacobian (J) and can be chosen to minimize a cost function
            # Chosen cost function is H(q)=sum(q_i ^2), to keep joints as near to 0 as possible while respecting the requirements
            # given by inverse kinematics.
            # H(q) minimum is found through gradient descent, by chosing q0d=-k0*gradient(H(q)) (easy for quadratic H(q))
        #NOTE: refer to the notation in the report
        ###
        bd = np.array(p.getBaseVelocity(self.Id)[1] + p.getBaseVelocity(self.Id)[0]) #(angular velocity, linear velocity)
        bd = np.reshape(bd,(bd.shape[0],1))
        qd = np.asarray(p.getBaseVelocity(self.Id)[1] + p.getBaseVelocity(self.Id)[0] + self.get_joints_speeds_tuple())
        qd = np.reshape(qd,(qd.shape[0],1))
        qda = qd[self.mask_act]
        qdn = qd[self.mask_nact]
        ###
        W = np.eye(qda.shape[0]) #weight matrix for the different joints ("how much should they try to minimize the cost of H(q)")
        Winv = lna.inv(W)
        Jy = self.COM_trn_jacobian()[1]
        Jy = np.reshape(Jy,(1,Jy.shape[0]))
        Jyb = Jy[:,self.mask_base]
        Jya = Jy[:,self.mask_act]
        Jyn = Jy[:,self.mask_nact]
        Jwr = Winv.dot(Jya.T).dot(lna.inv(Jya.dot(Winv.dot(Jya.T))))
        P = np.eye(qda.shape[0])-Jwr.dot(Jya)
        ###
        q0da = -2*k0*qda
        ###
        xd_desired = 0 - Jyb.dot(bd) - Jyn.dot(qdn)
        #COM_vy = Jy.dot(qd)
        e = self.COM_y_0-self.COM_position_world()[1]
        ###
        qda = np.ndarray.flatten(Jwr.dot(xd_desired + K*e) + P.dot(q0da))
        # print("xd_desired = ", xd_desired)
        # print("base linear v = ", p.getBaseVelocity(self.Id)[0])
        # print("base angular v = ", p.getBaseVelocity(self.Id)[1])
        # print("qdn", np.ndarray.flatten(qdn))
        # print("Jyb.dot(bd) = ", Jyb.dot(bd))
        # print("Jyn.dot(qdn)", Jyn.dot(qdn))
        #returned flattened as a NUMPY ARRAY (qda.shape,)
        return (qda, e)

    def generate_fmax_array_lateral(self,fmax_last):
        # Similar to self.generate_gain_matrix_lateral()
        fmax_array = list()
        half_spine_index = int(len(self.control_indices[0])/2)
        end_spine_index = len(self.control_indices[0])
        for i in range(1, half_spine_index+1):
            fmax_array.append(fmax_last*i)
        for i in range(half_spine_index,end_spine_index):
            fmax_array.append(fmax_last*(end_spine_index-i))
        return fmax_array

    def controlV_spine_lateral(self, K, fmax, k0=1, velocityGain=0.01, filtered=False):
        #For now it is able to keep a low error (order of 0.1 on the y speed of the COM)
            # until it reachs the limits of the joints.
            # Performances are limited probably by the PD controller on single joints.
            # Best performances seem to be obtained with low value of fmax
        # K should be generated with self.generate_gain_matrix_lateral()
        #NOTE: self.COM_y_0 should be set once at the start of each step phase, since it's used to compute the error
        control = self.solve_null_COM_y_speed_optimization(K=K,k0=k0)
        qda = control[0]
        qd = qda
        qdaf = self.low_pass_lateral.filter(qda)
        if filtered:
            qd = qdaf
        e = control[1]
        #qdaf = list(map(self.low_pass.filter,qda)) !!!NO!!! there should be a separate filter for each component
        for index, joint_i in enumerate(self.control_indices[0]):
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.VELOCITY_CONTROL, 
                targetVelocity=qd[index],
                force=fmax[index],
                velocityGain=velocityGain
                )
        # e is a NUMPY ARRAY
        return (qda,qdaf,e)
    
    def controlP_spine_lateral(self, K, fmax, positionGain=1, velocityGain=1, filtered=False):
        ###
        # NOTE: set correct initial value for the integrator before calling this function inside a for loop!!!
        ###
        #For now it is able to keep a low error (order of 0.1 on the y speed of the COM)
            # until it reachs the limits of the joints.
            # Performances are limited probably by the PD controller on single joints.
            # Best performances seem to be obtained with low value of fmax
        # K should be generated with self.generate_gain_matrix_lateral()
        control = self.solve_null_COM_y_speed_optimization(K=K)
        qda = control[0]
        e = control[1]
        if filtered:
            qda = self.low_pass_lateral.filter(qda)
        qa = self.integrator_lateral.integrate(qda)
        #qdaf = list(map(self.low_pass.filter,qda))
        for index, joint_i in enumerate(self.control_indices[0]):
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.POSITION_CONTROL, 
                targetPosition=qa[index],
                force=fmax[index],
                positionGain=positionGain,
                velocityGain=velocityGain
                )
        # e is a NUMPY ARRAY
        return (qda,qa,e)
    
    def control_leg_abduction(self, RL, theta0, thetaf, ti, t_stance, force=1, positionGain=1, velocityGain=0.5):
        #ti = time from start of the stance phase, t_stance = total (desired) stance phase duration
        #RL = 0 for right leg, 1 for left leg
        RL=int(RL)
        if (RL!=0 and RL!=1):
            print("ERROR: RL must be either 0 or 1 (right or left stance)")
            return
        theta = (theta0+thetaf)/2 + (theta0-thetaf)*cos(pi*ti/t_stance)/2
        p.setJointMotorControl2(self.Id, 
            self.control_indices[1+RL*2],   #NOTE: must be modified if the leg is changed in the URDF!!!
            p.POSITION_CONTROL,
            targetPosition = theta,
            force = force,
            positionGain = positionGain,
            velocityGain = velocityGain)
        return
    
    # def generate_velocityGain_array_lateral(self, k_last):
    #     # The gain is scaled linearly along the spine, with the maximum in the middle;
    #         # the input k_last correspond to the minimum value of the gain (first and last joint)
    #     k_array = list()
    #     half_spine_index = int(len(self.control_indices[0])/2)
    #     end_spine_index = len(self.control_indices[0])
    #     for i in range(1, half_spine_index+1):
    #         k_array.append(k_last*i)
    #     for i in range(half_spine_index,end_spine_index):
    #         k_array.append(k_last*(end_spine_index-i))
    #     return k_array

    # def girdle_flexion_control(self, RL=(False,False), to_neutral=(True,True), angles=(0,0), fmax=0.1, Kp=1, Kd=0.1):
    #     if (not(RL[0]) and not(RL[1])):
    #         # equivalent of a NOR gate, if no joint need to be controlled avoid every computation and exit
    #         return
    #     if RL[0]:
    #         angle_R = self.neutral_contact_flexion_angle if to_neutral else angles[0]
    #         p.setJointMotorControl2(self.Id,
    #             self.mask_right_girdle[1],
    #             p.POSITION_CONTROL,
    #             targetPosition = angle_R,
    #             fmax = fmax)
        
    #     return
    # def control_stance_abduction(self,thetaf,t, leg="R", fmax=0.1, Kp=1, Kd=0.1):
    #     # control the leg abduction from the current position to thetaf, following a cosine 
    #         # to have null speed at the start and at the end of the movement.
    #         # Sigmoid function could also be used for a slightly different behaviour
    #     #NOTE:theta0>thetaf
    #     if (leg=="R"):
    #         joint_i = self.mask_right_girdle[0]
    #     elif (leg=="L"):
    #         joint_i = self.mask_left_girdle[0]
    #     thetaR0 = p.getJointState(self.Id,joint_i)[0]
    #     thetaRf = -pi/4
    #     thetaR = (thetaR0+thetaf)/2 + (thetaR0-thetaf)*cos(pi*t)/2
    #     p.setJointMotorControl2(self.Id, 
    #         self.mask_right_girdle[0],
    #         p.POSITION_CONTROL,
    #         targetPosition = thetaR,
    #         fmax = 1,
    #         positionGain=1,
    #         velocityGain=0.5)



#NOTE 1: feet's constraints are set with their position using the values provided by the set_links_state_array,
    # might need to be modified if the leg in the URDF change
#NOTE 2: add function for mapping link and joint indices to their name?
#NOTE 3: set_joints_state_array can be taken out of each function and called just 
    # once after stepping the simulation, for now is called inside each function to avoid errors
#NOTE 4: might be nice to add computed torque control with an adaptive part for estimating 
    # the torques required to counteract friction, but must be formulated properly