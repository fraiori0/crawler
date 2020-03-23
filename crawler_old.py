import pybullet as p
import numpy as np
import scipy as sp
import scipy.linalg as lna
import time
import pybullet_data
from math import *
import pinocchio as pin

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
        ### PHYSICAL PROPERTIES AND URDF
        self.scale=1
        #NOTE: Properties in this block of code must be manually matched to those defined in the Xacro file
        self.spine_segments         = 8
        self.body_length            = self.scale * 1
        self.spine_segment_length   = self.scale * self.body_length/self.spine_segments
        self.leg_length             = self.scale * self.body_length/8
        self.body_sphere_radius     = self.scale * self.spine_segment_length/2
        self.foot_sphere_radius     = self.scale * self.body_sphere_radius/3
        #
        self.Id = p.loadURDF(
            "%s/crawler.urdf" % urdf_path, 
            base_position, 
            base_orientation, 
            globalScaling=self.scale,
            flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.num_joints = p.getNumJoints(self.Id)
        self.mass = 1
        for i in range(-1,self.num_joints):
            self.mass += p.getDynamicsInfo(self.Id, i)[0]
        #Value of the girdle flexion angle for having symmetric contact with both feet and the girdle collision sphere 
            #slightly reduced to avoid requiring compenetration when setting foot constraints and using position control to keep this value
        self.neutral_contact_flexion_angle = asin((self.body_sphere_radius-self.foot_sphere_radius)/self.leg_length)-0.0001
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
        ### Masks for seleceting row/columns of np.arrays, generally used with Jacobians or q/qd
        #NOTE: q include a quaternion for base orientation while qd has the angular speed of the base,
            # so q has 1 more element than qd. (But the quaternion must be a norm 1 vector, as usual)
            # All the masks are referred to qd except  the "shifted" version
            #-->PLEASE NOTE THAT PYBULLET ORDER THE MODEL AS BASE,SPINE,RIGHT GIRDLE,LEFT GIRDLE
        self.mask_base = [0,1,2,3,4,5]
        self.mask_joints = list(range(6,6+self.num_joints))
        self.mask_act = list(np.arange(6, 6+self.num_joints-4,2))
        self.mask_act_shifted = list(np.arange(7, 7+self.num_joints-4,2)) 
        self.mask_act_nobase = list(self.control_indices[0])
        self.mask_nact = list(np.arange(7, 6+self.num_joints-4,2))+list(range(6+self.num_joints-4,6+self.num_joints))
        self.mask_nact_nobase = list(np.arange(1, self.num_joints-4,2))+list(range(self.num_joints-4,self.num_joints))
        self.mask_right_girdle = [6+self.num_joints-4, 6+self.num_joints-3]
        self.mask_left_girdle = [6+self.num_joints-2, 6+self.num_joints-1]
        #Mask for pinocchio arrays
            #-->PLEASE NOTE THAT PINOCCHIO ORDER THE MODEL AS 
            # BASE, LEFT GIRDLE, RIGHT GIRDLE, SPINE
        self.mask_act_q_pin = list(np.arange(11, 7+self.num_joints,2))
        self.mask_act_qd_pin = list(np.arange(10, 6+self.num_joints,2))
        self.mask_right_girdle_q_pin = [9, 10]
        self.mask_left_girdle_q_pin = [7, 8]
        self.mask_right_girdle_qd_pin = [8, 9]
        self.mask_left_girdle_qd_pin = [6, 7]
        #
        # q_pin = q[mask_q_py_to_pin]
        self.mask_q_pyb_to_pin = (
            list(range(0,7)) + 
            [7+self.num_joints-2, 7+self.num_joints-1] + 
            [7+self.num_joints-4, 7+self.num_joints-3] + 
            list(range(7,7+self.num_joints-4)))
        self.mask_qd_pyb_to_pin = (
            list(range(0,6)) + 
            [6+self.num_joints-2, 6+self.num_joints-1] + 
            [6+self.num_joints-4, 6+self.num_joints-3] + 
            list(range(6,6+self.num_joints-4)))
        ### Filters and Integrators EACH FILTERED VARIABLE NEED ITS OWN FILTER
        self.dt_simulation = dt_simulation
        self.low_pass_lateral = Discrete_Low_Pass(dim=len(self.mask_act),dt=self.dt_simulation, fc=100*self.dt_simulation, K=1)
        self.low_pass_qd = Discrete_Low_Pass(dim=(self.num_joints + 6),dt=self.dt_simulation, fc=100*self.dt_simulation, K=1)
        self.low_pass_tau_lateral = Discrete_Low_Pass(dim=len(self.mask_act),dt=self.dt_simulation, fc=100*self.dt_simulation, K=1)
        self.integrator_lateral = Integrator_Forward_Euler(self.dt_simulation,[0]*len(self.mask_act))
        ### joints' speeds are stored, so that they can be derived to get joints' accelerations
        self.joints_speeds_prev = np.array(self.get_joints_speeds_tuple())
        self.joints_speeds_curr = np.array(self.get_joints_speeds_tuple())
        tmp_b = p.getBaseVelocity(self.Id)
        self.base_velocity_prev = np.concatenate((np.array(tmp_b[0]), np.array(tmp_b[1])))
        self.base_velocity_curr = self.base_velocity_prev
        self.qd_prev = np.concatenate((self.base_velocity_prev,self.joints_speeds_prev))
        self.qd_curr = np.concatenate((self.base_velocity_curr,self.joints_speeds_curr))
        ###PINOCCHIO INITIALIZATION AND VARIABLES
        self.pinmodel = pin.buildModelFromUrdf("%s/crawler.urdf" % urdf_path,pin.JointModelFreeFlyer())
        self.pin_data = self.pinmodel.createData()
        self.pin_data_eta = self.pinmodel.createData()
        #self.pinmodel.gravity = pin.Motion.Zero()
        
    def set_low_pass_lateral(self, fc, K=1):
        self.low_pass_lateral = Discrete_Low_Pass(
            dim=len(self.mask_act),
            dt=self.dt_simulation, 
            fc=fc*self.dt_simulation, 
            K=K)

    def set_low_pass_qd(self, fc, K=1):
        self.low_pass_qd = Discrete_Low_Pass(
            dim=(self.num_joints+6),
            dt=self.dt_simulation, 
            fc=fc*self.dt_simulation, 
            K=K)
    
    def set_low_pass_tau_lateral(self, fc, K=1):
        self.low_pass_tau_lateral = Discrete_Low_Pass(
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
        p.setJointMotorControl2(self.Id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0.0)
        return

    def turn_off_crawler(self):
        for i in range(0,p.getNumJoints(self.Id)):
            p.setJointMotorControl2(self.Id, i, controlMode=p.VELOCITY_CONTROL, force=0.0)
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

    def fix_right_foot(self):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["right_foot"]:
            constId=self.constraints["right_foot"]
            print("Error: remove right foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, self.control_indices[2], -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,self.control_indices[2])[0]))
            self.constraints["right_foot"]=constId
        return constId

    def fix_left_foot(self):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["left_foot"]:
            constId=self.constraints["left_foot"]
            print("Error: remove left foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, self.control_indices[4], -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,self.control_indices[4])[0]))
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

    def set_velocities(self):
        ### To be called after each simulation step
        self.joints_speeds_prev = self.joints_speeds_curr
        self.joints_speeds_curr = np.array(self.get_joints_speeds_tuple())
        tmp = p.getBaseVelocity(self.Id)
        self.base_velocity_prev = self.base_velocity_curr
        self.base_velocity_curr = np.concatenate((np.array(tmp[0]), np.array(tmp[1])))
        self.qd_prev = np.concatenate((self.base_velocity_prev,self.joints_speeds_prev))
        self.qd_curr = np.concatenate((self.base_velocity_curr,self.joints_speeds_curr))
        return

    def get_q(self):
        # qb = pos(xyz), orient(xyzw (quaternion))
        tmp_b = p.getBasePositionAndOrientation(self.Id)
        q = np.array((tmp_b[0] + tmp_b[1] + self.get_joints_pos_tuple()))
        return q

    def get_base_Eulers(self):
        return  p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.Id)[1])
    
    def get_R_base_to_world(self):
        # returned as a (3,3) NUMPY ARRAY
        # R * x_base = x_world
        quat_base = p.getBasePositionAndOrientation(self.Id)[1]
        R = np.reshape(np.array(p.getMatrixFromQuaternion(quat_base)),(3,3))
        return R
    
    def get_link_COM_jacobian_world(self,link_index, R, joints_pos, link_state_set=False):
        # Compute the Jacobian for the COM of a single link, referred to world global coordinates.
        # R and joint_pos should be passed already computed (through the proper class methods) to avoid recomputing them
            # even if the simulation has not stepped between two calls and the state is still the same.
        # For the same reason self.set_links_state_array() should be called once (every time-step) just before using this function.
        #NOTE (see report): p.calculateJacobian outputs a Jacobian that gives the linear/angular velocity expressed
            # in a reference frame oriented like the base but fixed in the global reference frame
            # To get the velocities in the global reference frame we should transform the Jacobian (see report)
        #
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
            # NOTE: angular speed values are null just because the COM of the girdle link coincides with the origin of the link frame
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
    
    def get_qdd_nparr(self):
        # since this should return the "exact" values the unfiltered version might be better
        # return a backward Euler approximation of acceleration using both unfiltered and filtered speeds value
        print("### Inside get_qdd_parr ###")
        qdd =  (self.qd_curr - self.qd_prev)/self.dt_simulation
        qd_prev_f = self.low_pass_qd.x
        qd_curr_f = self.low_pass_qd.filter(self.qd_curr)
        qdd_f = (qd_curr_f-qd_prev_f)/self.dt_simulation
        print("### Exit get_joints_acc_parr ###")
        return qdd, qdd_f
    
    def solve_null_COM_y_speed_optimization_qdda(self, qda_prev, K, filtered=True):
        ### Compute also qa and qdda from the qda computed inside self.solve_null_COM_y_speed_optimization()
        ### This is probably better in its filtered version, 
            # since this is the desired trajectory to track at joints level
        kin_sol = self.solve_null_COM_y_speed_optimization(K=K)
        qda_curr = kin_sol[0]
        qda_prev_f = self.low_pass_lateral.x
        qda_curr_f = self.low_pass_lateral.filter(qda_curr)
        if filtered:
            qda = qda_curr_f
            qdda = (qda_curr_f - qda_prev_f)/(self.dt_simulation)
        else:
            qda = qda_curr
            qdda = (qda_curr - qda_prev)/(self.dt_simulation)
        qa = self.integrator_lateral.integrate(qda)
        return qa,qda, qdda

    def solve_torques_lateral(self, qa_des, qda_des, qdda_des, qdd_curr, Kp, Kv): 
        ### Everything should be passed as a NUMPY ARRAY
        # ------->  SET_VELOCITIES SHOULD BE CALLED (ONCE) BETWEEN THIS FUNCTION AND P.STEPSIMULATION()  <-------
            # since the accelerations of the not-actuated joints must be the real values, not the desired ones and
            # they need to be extracted by derivation (backward Euler, check if it's stable and work properly or need to be filtered)
        ###
        print("\n### Inside solve_torques_lateral ###")
        # SET QDD AND QDDA VALUES
        qdd_curr_pin = qdd_curr[self.mask_qd_pyb_to_pin]
        print("qdd_curr: ", qdd_curr)
        qdd_des = qdd_curr.copy()
        qdd_des[self.mask_act] = list(qdda_des)
        qdd_des_pin = qdd_des[self.mask_qd_pyb_to_pin]
        print("qdd_curr with desired: ", qdd_des)
        # SET Q AND QD VALUES
        q = self.get_q()
        print("q(r4) = ",np.round(q,4))
        q_pin = q[self.mask_q_pyb_to_pin]
        print("q_pin(r4) = ",np.round(q_pin,4))
        qd = self.qd_curr
        qd_pin = qd[self.mask_qd_pyb_to_pin]
        # SET ERRORS VALUE, BOTH E and ED
        ed = qda_des - qd[self.mask_act]
        print("ed(r4): ", np.round(ed,4))
        ed = np.reshape(ed,(ed.shape[0],1))
        e = qa_des - q[self.mask_act_shifted]
        print("e(r4): ", np.round(e,4))
        e = np.reshape(e,(e.shape[0],1))
        ### DYNAMICS (Pinocchio library)
        # see https://github.com/stack-of-tasks/pinocchio/blob/master/bindings/python/multibody/data.hpp)
            # to check which attributes data may contain 
            # (each attribute should be set with the proper function, before being called
            # e.g. pin.crba() just set data.M to the correct value)
        # Since joints are ordered differenty in PyBullet and Pinocchio, when calling Pinocchio's functions
            # q_pin, qd_pin and qdd_pin should be passed instead of q,qd, and qdd
        #
        # Compute mass matrix with Pinocchio library 
        pin.crba(self.pinmodel,self.pin_data,q_pin)
        M = self.pin_data.M
        print("Shape M (if 26x26) things are ok")
        Ma = M[:,self.mask_act_qd_pin][self.mask_act_qd_pin,:]
        # COMPUTE THE DISTURBANCE TERM (eta) 
            # in the dynamics equations, using the real term for qdd (not the desired one).
            # Stored in self.pin_data_eta to avoid overwriting terms that might be useful later
            # M*qdd + C*qd + G = tau_real + eta, pin.rnea() compute the left term of this equation
        pin.rnea(self.pinmodel,self.pin_data_eta,q_pin,qd_pin,qdd_curr_pin)
        taub_real = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) #the 6DOFs of the base are passive, no action can be directly applied to them
        joint_states = np.array(p.getJointStates(self.Id,list(range(self.num_joints))))
        tauj_real = joint_states[:,3]
        tau_real = np.concatenate((taub_real,tauj_real)).astype(np.double)
        eta = self.pin_data_eta.tau - tau_real
        print("Eta: ",np.round(eta,4))
        #eta = np.reshape(eta, (eta.shape[0],1))
        # INVERSE DYNAMICS WITH QDDA
        pin.rnea(self.pinmodel,self.pin_data,q_pin,qd_pin,qdd_des_pin)
        tau_exact = np.array(self.pin_data.tau) - eta
        print("TAU_EXACT:  ",np.round(tau_exact,4))
        print("SHAPE TAU_EXACT:  ", tau_exact.shape)
        #print("TAU_EXACT:  ", np.round(tau_exact,4))
        tau_exact = np.reshape(tau_exact, (tau_exact.shape[0],1))
        print("SHAPE TAU_EXACT AFTER RESHAPING:  ", tau_exact.shape)
        #
        tau_act_closed_loop = tau_exact[self.mask_act_qd_pin] + Ma.dot(Kv).dot(ed) + Ma.dot(Kp).dot(e)
        print("tau_exact[self.mask_act_qd_pin]: ",np.ndarray.flatten(np.round((tau_exact[self.mask_act_qd_pin]),4)))
        print("Ma.dot(Kv).dot(ed): ", np.ndarray.flatten(np.round(Ma.dot(Kv).dot(ed),4)))
        print("Ma.dot(Kp).dot(e): ", np.ndarray.flatten(np.round(Ma.dot(Kp).dot(e),4)))
        print("tau_act_closed_loop: ", np.ndarray.flatten(np.round(tau_act_closed_loop,4)))
        tau_act_closed_loop = np.reshape(tau_act_closed_loop, (tau_act_closed_loop.shape[0],))
        print("### Exit solve_torque_lateral ###\n")
        return tau_act_closed_loop, np.ndarray.flatten(e)

    def controlT_spine_lateral(self, qda_des_prev, K, Kp, Kv, filtered_des=True, filtered_real=False, filtered_tau=False):
        # K is the value of the gain for the inner loop, that computes the speeds desired for the lateral joint
        print("\n### Inside controlT_spine_lateral ###")
        qa_des, qda_des, qdda_des = self.solve_null_COM_y_speed_optimization_qdda(qda_des_prev,K=K,filtered=filtered_des)
        print("qa_des: ", qa_des)
        print("qda_des_prev: ", qda_des_prev)
        print("qda_des: ", qda_des)
        print("qdda_des: ", qdda_des)
        if filtered_real:
            qdd_curr = self.get_qdd_nparr()[1]
        else:
            qdd_curr = self.get_qdd_nparr()[0]
        print("qdd_curr: ", qdd_curr)
        edd = qdd_curr[self.mask_act]-qdda_des
        print("edd: ", (edd))
        torque_solution = self.solve_torques_lateral(
            qa_des=qa_des,
            qda_des=qda_des,
            qdda_des=qdda_des,
            qdd_curr=qdd_curr,
            Kp=Kp,
            Kv=Kv)
        tau_act = torque_solution[0]
        eq = torque_solution[1]
        print("tau_act: ", np.round(tau_act,4), "\n with shape", tau_act.shape)
        tau_act_f = self.low_pass_tau_lateral.filter(tau_act)
        print("tau_act_f: ", np.round(tau_act_f,4), "\n with shape", tau_act_f.shape)
        if filtered_tau:
            tau_applied=tau_act_f
        else:
            tau_applied=tau_act
        for index, joint_i in enumerate(self.control_indices[0]):
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.TORQUE_CONTROL,
                force=tau_applied[index]
                )
        eCOM = self.COM_y_0-self.COM_position_world()[1]
        print("### Exit controlT_spine_lateral ###\n")
        return [qa_des,qda_des,qdda_des] , eq, eCOM, [tau_act,tau_act_f]
    
    def control_leg_abduction(self, RL, theta0, thetaf, ti, t_stance, fmax=1, positionGain=1, velocityGain=0.5):
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
            force = fmax,
            positionGain = positionGain,
            velocityGain = velocityGain)
        return
    
    def control_leg_flexion(self, RL, fmax=1, positionGain=1, velocityGain=0.5):
        #RL = 0 for right leg, 1 for left leg
        RL=int(RL)
        if (RL!=0 and RL!=1):
            print("ERROR: RL must be either 0 or 1 (right or left stance)")
            return
        theta = self.neutral_contact_flexion_angle * (1-RL*2)
        p.setJointMotorControl2(self.Id, 
            self.control_indices[2+RL*2],   #NOTE: must be modified if the leg is changed in the URDF!!!
            p.POSITION_CONTROL,
            targetPosition = theta,
            force = fmax,
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
#NOTE 5: values of variable in the previous step might be stored as class attributes, giving methods to 
    # set them everytime before calling p.stepsimulation()