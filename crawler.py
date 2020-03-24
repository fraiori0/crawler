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

class Model_State:
    def __init__(self, nq, dt, modelId):
        ### Simulation parameter
        # time step of the simulation
        self.modelId = modelId
        self.dt = dt
        ###
        # DOFs number
        # total number of DOFs
            #NOTE: q include a quaternion for base orientation while qd has the angular speed of the base,
            # so q has 1 more element than qd. (Quaternion must be a norm 1 vector, as usual)
        self.nq = nq
        self.nqd = nq-1
        # number of DOFs of the base and the internal joints
        self.nqb = 7
        self.nqbd = 6
        self.nqj = self.nq - self.nqb
        self.nqjd = self.nqd - self.nqbd
        ###
        # Arrays containing the values
            # the _prev version of qd stores the value from the previous time-step, 
            # to compute acceleration with backward Euler
        # Base
        self.qb = np.array(([0.0]*self.nqb))
        #self.qb[6]=1
        self.qbd_prev = np.array(([0.0]*self.nqbd))
        self.qbd = np.array(([0.0]*self.nqbd))
        self.qbdd = np.array(([0.0]*self.nqbd))
        # Joints
        self.qj = np.array([0.0]*self.nqj)
        self.qjd_prev = np.array([0.0]*self.nqjd)
        self.qjd = np.array([0.0]*self.nqjd)
        self.qjdd = np.array([0.0]*self.nqjd)
        # Complete
        self.q = np.array([0.0]*self.nq)
        self.qd_prev = np.array([0.0]*self.nqd)
        self.qd = np.array([0.0]*self.nqd)
        self.qdd = np.array([0.0]*self.nqd)
        # Torque vector
        self.tau = np.array([0.0]*self.nqd)
        #
        self.joint_indices_q = list(range(self.nqb,self.nq))
        self.joint_indices_qd = list(range(self.nqbd,self.nqd))
        ###
    
    def set_vel(self,qd_new):
        self.qd_prev = self.qd
        self.qd = qd_new
        self.qbd_prev = self.qd_prev[list(range(6))]
        self.qjd_prev = self.qd_prev[list(range(6,self.nqd))]
        self.qbd = self.qd[list(range(6))]
        self.qjd = self.qd[list(range(6,self.nqd))]

    def set_acc(self):
        self.qdd = (self.qd - self.qd_prev)/self.dt
        self.qbdd = self.qdd[list(range(6))]
        self.qjdd = self.qdd[list(range(6,self.nqd))]
    
    def update(self):
        #NOTE: this function should be called only ONCE PER TIME-STEP, otherwise qdd will be set to 0
        info_qb = p.getBasePositionAndOrientation(self.modelId)
        info_qbd = p.getBaseVelocity(self.modelId)
        qbd_new = np.concatenate((info_qbd[0],info_qbd[1]))
        info_j = np.array(p.getJointStates(self.modelId,list(range(self.nqjd))))
        qjd_new = info_j[:,1].astype(np.double)
        qd_new = np.concatenate((qbd_new,qjd_new))#.astype(np.double)
        self.qb = np.concatenate((info_qb[0],info_qb[1]))
        self.qj = info_j[:,0].astype(np.double)
        self.q = np.concatenate((self.qb,self.qj))
        self.set_vel(qd_new)
        self.set_acc()
        self.tau[self.joint_indices_qd] = info_j[:,3].astype(np.double)



class Crawler:

    def __init__(self, dt_simulation, urdf_path="/home/fra/Uni/Tesi/crawler", base_position=[0,0,0.5], base_orientation=[0,0,0,1]):
        ### PHYSICAL PROPERTIES AND URDF ###
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
        self.mass = 0.0
        for i in range(-1,self.num_joints):
            self.mass += p.getDynamicsInfo(self.Id, i)[0]
        self.dt_simulation = dt_simulation
        ### STATE VARIABLES ###
        # q, qd, qdd an tau are stored inside a Model_State object
            # NOTE: desired.update() should NEVER be called, self.desired is just for storing purpose
        self.state = Model_State((self.num_joints+7),self.dt_simulation, self.Id)
        # state of each links, see set_links_state_array() to see how it is composed
        #NOTE: links_state_array doesn't include the state of the base to keep the correct id-numbers of the links
        self.links_state_array=[0]*self.num_joints
        self.set_links_state_array()
        ### AUXILIARY VARIABLES ###
        # Value of the girdle flexion angle for having symmetric contact with both feet and the girdle collision sphere 
            # slightly reduced to avoid requiring compenetration when setting foot constraints and using position control to keep this value
        self.neutral_contact_flexion_angle = asin((self.body_sphere_radius-self.foot_sphere_radius)/self.leg_length)-0.0001
        self.control_indices = self.generate_control_indices()
        self.constraints = {
            "right_foot": 0,
            "left_foot": 0
        }
        #
        self.COM_y_ref=self.COM_position_world()[1]
        ### MASKS ###
        # Masks for selecting row/columns of np.arrays, generally used with Jacobians or q/qd/qdd
        #NOTE: q include a quaternion for base orientation while qd has the angular speed of the base,
            # so q has 1 more element than qd. (But the quaternion must be a norm 1 vector, as usual)
        # --> All the masks are referred to qd except  the "shifted" version
        # NOTE: PYBULLET ORDER THE MODEL AS (BASE,SPINE,RIGHT GIRDLE,LEFT GIRDLE)
        self.mask_base = list(range(6))
        self.mask_joints = list(range(6,6+self.num_joints))
        self.mask_joints_shifted = list(range(7,7+self.num_joints))
        self.mask_act = list(np.arange(6, 6+self.num_joints-4,2))
        self.mask_act_shifted = list(np.arange(7, 7+self.num_joints-4,2)) 
        self.mask_act_nobase = list(self.control_indices[0])
        self.mask_nact = list(np.arange(7, 6+self.num_joints-4,2))+list(range(6+self.num_joints-4,6+self.num_joints))
        self.mask_nact_nobase = list(np.arange(1, self.num_joints-4,2))+list(range(self.num_joints-4,self.num_joints))
        self.mask_right_girdle = [6+self.num_joints-4, 6+self.num_joints-3]
        self.mask_left_girdle = [6+self.num_joints-2, 6+self.num_joints-1]
        # Masks for pinocchio arrays
            # NOTE: PINOCCHIO ORDER THE MODEL AS (BASE, LEFT GIRDLE, RIGHT GIRDLE, SPINE)
        self.mask_act_q_pin = list(np.arange(11, 7+self.num_joints,2))
        self.mask_act_qd_pin = list(np.arange(10, 6+self.num_joints,2))
        self.mask_right_girdle_q_pin = [9, 10]
        self.mask_left_girdle_q_pin = [7, 8]
        self.mask_right_girdle_qd_pin = [8, 9]
        self.mask_left_girdle_qd_pin = [6, 7]
        # Masks for getting pinocchio arrays of q/qd/qdd from the correspondetn PyBullet arrays
            # q_pin = q[mask_q_py_to_pin]
            # qd_pin = qd[mask_qd_py_to_pin]
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
        self.mask_q_pin_to_pyb = (
            list(range(0,7)) +
            list(range(11,11+self.num_joints-4)) +
            [9,10] +
            [7,8]
        )
        self.mask_qd_pin_to_pyb = (
            list(range(0,6)) +
            list(range(10,10+self.num_joints-4)) +
            [8,9] +
            [6,7]
        )
        ### FILTERS AND INTEGRATORS ###
        # NOTE: EACH FILTERED(/integrated) VARIABLE NEED ITS OWN FILTER(/integrator)
        self.low_pass_qd = Discrete_Low_Pass(dim=self.state.nqd,dt=self.dt_simulation, fc=100*self.dt_simulation, K=1)
        self.low_pass_tau = Discrete_Low_Pass(dim=self.state.nqd,dt=self.dt_simulation, fc=100*self.dt_simulation, K=1)
        self.low_pass_lateral_qa = Discrete_Low_Pass(dim=len(self.mask_act),dt=self.dt_simulation, fc=100*self.dt_simulation, K=1)
        self.integrator_lateral_qa = Integrator_Forward_Euler(self.dt_simulation,[0]*len(self.mask_act))
        ###PINOCCHIO INITIALIZATION AND VARIABLES
        self.pinmodel = pin.buildModelFromUrdf("%s/crawler.urdf" % urdf_path,pin.JointModelFreeFlyer())
        self.pin_data = self.pinmodel.createData()
        self.pin_data_eta = self.pinmodel.createData()
        #self.pinmodel.gravity = pin.Motion.Zero()
        
    def set_low_pass_lateral_qa(self, fc, K=1):
        self.low_pass_lateral_qa = Discrete_Low_Pass(
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
    
    def set_low_pass_tau(self, fc, K=1):
        self.low_pass_tau = Discrete_Low_Pass(
            dim=self.state.nqd,
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

    def set_COM_y_ref(self):
        # set the current COM y position as the one to measure error from
            # (useful when trying to keep a fixed y position)
        self.COM_y_ref = self.COM_position_world()[1]

    def turn_off_joint(self, joint_index):
        p.setJointMotorControl2(self.Id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0.0)

    def turn_off_crawler(self):
        for i in range(0,p.getNumJoints(self.Id)):
            p.setJointMotorControl2(self.Id, i, controlMode=p.VELOCITY_CONTROL, force=0.0)

    def generate_control_indices(self):
    # this function relies on knowledge of the order of the joints in the crawler model
    # NOTE: if the URDF is modified in ways different than just adding more segments to the spine this function
        # NEED TO be updated properly
        lat_joints_i = tuple(range(0,(self.num_joints-4),2))
        #abduction then flexion 
        r_leg_i = (self.num_joints-4, self.num_joints-3)
        l_leg_i = (self.num_joints-2, self.num_joints-1)
        return (lat_joints_i,r_leg_i,l_leg_i)

    def fix_right_foot(self):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["right_foot"]:
            constId=self.constraints["right_foot"]
            print("Error: remove right foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, self.control_indices[1][1], -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,self.control_indices[1][1])[0]))
            self.constraints["right_foot"]=constId
        return constId

    def fix_left_foot(self):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["left_foot"]:
            constId=self.constraints["left_foot"]
            print("Error: remove left foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, self.control_indices[2][1], -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,self.control_indices[2][1])[0]))
            self.constraints["left_foot"]=constId
        return constId
    
    def free_right_foot(self):
        p.removeConstraint(self.constraints["right_foot"])
        self.constraints["right_foot"]=0

    def free_left_foot(self):
        p.removeConstraint(self.constraints["left_foot"])
        self.constraints["left_foot"]=0    

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

    # def get_joints_pos_tuple(self):
    #     return list(zip(*(p.getJointStates(self.Id,list(range(0,self.num_joints))))))[0]

    # def get_joints_speeds_tuple(self):
    #     return list(zip(*(p.getJointStates(self.Id,list(range(0,self.num_joints))))))[1]

    # def set_velocities(self):
    #     ### To be called after each simulation step
    #     self.joints_speeds_prev = self.joints_speeds_curr
    #     self.joints_speeds_curr = np.array(self.get_joints_speeds_tuple())
    #     tmp = p.getBaseVelocity(self.Id)
    #     self.base_velocity_prev = self.base_velocity_curr
    #     self.base_velocity_curr = np.concatenate((np.array(tmp[0]), np.array(tmp[1])))
    #     self.qd_prev = np.concatenate((self.base_velocity_prev,self.joints_speeds_prev))
    #     self.qd_curr = np.concatenate((self.base_velocity_curr,self.joints_speeds_curr))
    #     return

    # def get_q(self):
    #     # qb = pos(xyz), orient(xyzw (quaternion))
    #     tmp_b = p.getBasePositionAndOrientation(self.Id)
    #     q = np.array((tmp_b[0] + tmp_b[1] + self.get_joints_pos_tuple()))
    #     return q

    def get_base_Eulers(self):
        return  p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.Id)[1])
    
    def get_R_base_to_world(self):
        # returned as a (3,3) NUMPY ARRAY
        # R * x_base = x_world
        quat_base = p.getBasePositionAndOrientation(self.Id)[1]
        R = np.reshape(np.array(p.getMatrixFromQuaternion(quat_base)),(3,3))
        return R
    
    def get_link_COM_jacobian_trn_world(self, link_index, R):
        # Compute the Jacobian for the COM of a single link, referred to world global coordinates.
        # R should be passed already computed (through the proper class methods) to avoid recomputing them
            # even if the simulation has not stepped between two calls and the state is still the same.
        # For the same reason self.set_links_state_array() should be called once (every time-step) just before using this function.
        #NOTE (see report): p.calculateJacobian outputs a Jacobian that gives the linear/angular velocity expressed
            # in a reference frame oriented like the base but fixed in the global reference frame
            # To get the velocities in the global reference frame we should transform the Jacobian (see report)
        # Jtbtw stands for 
            # Jacobian(for the)Translational(motion; columns multiplying the)Base_Translational(components; referred to the)_World(reference frame)
        # p.calculateJacobian() gives a Jacobian with first the base rotation and then the base translation:
            # this Jacobian is rearranged so that it should multiply a state 
            # composed as (base_translation, base_rotation, joints), as usual
        ###
        J = np.asarray(
            p.calculateJacobian(self.Id,
                link_index,
                self.links_state_array[link_index]["loc_com_trn"],
                #joints_pos,
                self.state.qj.tolist(),
                [0.0]*(self.num_joints),
                #self.get_joints_speeds_tuple(),
                [0.0]*(self.num_joints))
            )
        Jt = J[0]
        Jtbrw = (R.dot(Jt[:,[0,1,2]])).dot(R.T)
        Jtbtw = Jt[:,[3,4,5]]
        Jtqw = R.dot(Jt[:,self.mask_joints])
        Jtw_i = np.concatenate((Jtbtw, Jtbrw, Jtqw),1)
        ### rotation Jacobian
        # Jr_i = Ji[1]
        # Jrqw_i = R.dot(Jr_i[:,self.mask_joints])
        # Jrw_i = np.concatenate((Jr_i[:,self.mask_base], Jrqw_i),1)
        ###
        #returned as NUMPY ARRAY
        return Jtw_i

    def get_COM_trn_jacobian(self):
        #Jacobian of the COM is computed as the weighted mean (with respect to masses) of the Jacobian of the links
        #The transational jacobian of the base is just the identity matrix multiplying the base translational velocity
            # plus null terms associated to the base angular speed and the joints speed terms (thus 3+self.num_joints)
            # NOTE: angular speed values are null just because the COM of the girdle link coincides with the origin of the link frame
        R = self.get_R_base_to_world()
        ###
        Jbase_t = np.asarray([  [1.0,0.0,0.0] + [0.0]*3 + [0.0]*(self.num_joints),
                                [0.0,1.0,0.0] + [0.0]*3 + [0.0]*(self.num_joints),
                                [0.0,0.0,1.0] + [0.0]*3 + [0.0]*(self.num_joints) ])
        JM_t = Jbase_t*(p.getDynamicsInfo(self.Id,-1)[0])
        ###
        for i in range(0,self.num_joints):
            Jtw_i = self.get_link_COM_jacobian_trn_world(i, R=R)
            JM_t += Jtw_i * (p.getDynamicsInfo(self.Id,i)[0])
        ###
        JM_t = JM_t/self.mass
        #returned as NUMPY ARRAY
        return JM_t

    # def set__COM_trn_jacobian(self):
    #     self.JM = self.get_COM_trn_jacobian()

    def solve_null_COM_y_speed_optimization(self, K, k0=1, verbose=False):
        #Return the desired joint speeds of the spinal lateral joints to be used for velocity control
        #NOTE: self.COM_y_ref should be set once at the start of each step phase
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
        qd = np.reshape(self.state.qd,(self.state.qd.shape[0],1))
        bd = qd[self.mask_base]
        qda = qd[self.mask_act]
        qdn = qd[self.mask_nact]
        ###
        # Weight matrix for the different joints
            # ("how much should they try to minimize the cost of H(q)")
        W = np.eye(qda.shape[0])
        Winv = lna.inv(W)
        # Jy = row of the COM's Jacobian corresponding to y-axis motion
        Jy = self.get_COM_trn_jacobian()[1]
        Jy = np.reshape(Jy,(1,Jy.shape[0]))
        Jyb = Jy[:,self.mask_base]
        Jya = Jy[:,self.mask_act]
        Jyn = Jy[:,self.mask_nact]
        Jwr = Winv.dot(Jya.T).dot(lna.inv(Jya.dot(Winv.dot(Jya.T))))
        P = np.eye(qda.shape[0])-Jwr.dot(Jya)
        ###
        q0da = -2*k0*qda
        ###
        # Desired speed
        xd_desired = 0 - Jyb.dot(bd) - Jyn.dot(qdn)
        # Error
        eCOMy = self.COM_y_ref-self.COM_position_world()[1]
        ###
        # Solution of the Inverse Kinematics.
            # Returned flattened as a NUMPY ARRAY (qda.shape,)
        qda = np.ndarray.flatten(Jwr.dot(xd_desired + K*eCOMy) + P.dot(q0da))
        if verbose:
            print("xd_desired = ", xd_desired)
            print("qdn", np.ndarray.flatten(qdn))
            print("Jyb.dot(bd) = ", Jyb.dot(bd))
            print("Jyn.dot(qdn)", Jyn.dot(qdn))
            print("qda: ", qda)
        return (qda, eCOMy)

    def generate_fmax_array_lateral(self,fmax_last):
        # Generate the fmax array for the lateral joints 
        # to be used with position or velocity control, like with self.controlV_spine_lateral()
        fmax_array = list()
        half_spine_index = int(len(self.control_indices[0])/2)
        end_spine_index = len(self.control_indices[0])
        for i in range(1, half_spine_index+1):
            fmax_array.append(fmax_last*i)
        for i in range(half_spine_index,end_spine_index):
            fmax_array.append(fmax_last*(end_spine_index-i))
        return fmax_array

    def controlV_spine_lateral(self, K, fmax, k0=1, velocityGain=0.01, filtered=False):
        ## See report for all the factors affecting the performances
        #NOTE: self.COM_y_ref should be set once at the start of each STEP (of the animal, not time-step),
            # since it's used to compute the error
        control = self.solve_null_COM_y_speed_optimization(K=K,k0=k0)
        qda = control[0]
        qd = qda
        qdaf = self.low_pass_lateral_qa.filter(qda)
        if filtered:
            qd = qdaf
        e = control[1]
        for index, joint_i in enumerate(self.control_indices[0]):
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.VELOCITY_CONTROL, 
                targetVelocity=qd[index],
                force=fmax[index],
                velocityGain=velocityGain
                )
        return (qda,qdaf,e)
    
    def solve_null_COM_y_speed_optimization_qdda(self, qda_prev, K, filtered=True):
        # Compute also qa and qdda from the qda computed inside self.solve_null_COM_y_speed_optimization().
        # This is probably better in its filtered version, 
            # since this is the desired trajectory to track at joints level
        kin_sol = self.solve_null_COM_y_speed_optimization(K=K)
        qda_curr = kin_sol[0]
        qda_prev_f = self.low_pass_lateral_qa.x
        qda_curr_f = self.low_pass_lateral_qa.filter(qda_curr)
        if filtered:
            qda = qda_curr_f
            qdda = (qda_curr_f - qda_prev_f)/(self.dt_simulation)
        else:
            qda = qda_curr
            qdda = (qda_curr - qda_prev)/(self.dt_simulation)
        qa = self.integrator_lateral_qa.integrate(qda)
        return qa,qda, qdda
    
    def generate_abduction_trajectory(self, theta0, thetaf, ti, t_stance):
        #ti = time from start of the walking locomotion, t_stance = total (desired) stance phase duration
        theta = (theta0+thetaf)/2 + (theta0-thetaf)*cos(pi*ti/t_stance)/2
        thetad = -pi*(theta0-thetaf)*sin(pi*ti/t_stance)/(2*t_stance)
        thetadd = -pi*pi*(theta0-thetaf)*cos(pi*ti/t_stance)/(2*t_stance*t_stance)
        return theta, thetad, thetadd
    
    def generate_joints_trajectory(self, theta0, thetaf, ti, t_stance, RL, qa_des, qda_des, qdda_des):
        #ti = time from start of the walking locomotion, t_stance = total (desired) stance phase duration
        # RL=0 if right leg stance, 1 if left leg stance (stance leg is the one with the constraied foot)
        # q[3:6] must be a quaternion
        q_des = self.state.q.copy()
        qd_des = self.state.qd.copy()
        qdd_des = self.state.qdd.copy()
        # Right leg abdction trajectory
        right_abd = self.generate_abduction_trajectory(theta0,thetaf,ti,t_stance)
        q_des[(7 + self.control_indices[1][0])] = right_abd[0]
        qd_des[(6 + self.control_indices[1][0])] = right_abd[1]
        qdd_des[(6 + self.control_indices[1][0])] = right_abd[2]
        # Left leg abdution trajectory, 
            # same as the right leg but translated backward temporally and with an opposite sign 
            # (see joint's reference frame)
        left_abd = self.generate_abduction_trajectory(theta0,thetaf,(ti-t_stance),t_stance)
        q_des[(7 + self.control_indices[2][0])] = -(left_abd[0])
        qd_des[(6 + self.control_indices[2][0])] = -(left_abd[1])
        qdd_des[(6 + self.control_indices[2][0])] = -(left_abd[2])
        # Stance leg flexion
        q_des[(7 + self.control_indices[1+RL][1])] = (1-2*RL)*self.neutral_contact_flexion_angle
        qd_des[(6 + self.control_indices[1+RL][1])] = 0
        qdd_des[(6 + self.control_indices[1+RL][1])] = 0
        # Lateral spinal joints
        q_des[self.mask_act_shifted] = qa_des
        qd_des[self.mask_act] = qda_des
        qdd_des[self.mask_act] = qdda_des
        pass
        return q_des, qd_des, qdd_des
    
    def solve_computed_torque_control(self, q_des, qd_des, qdd_des, Kp, Kv, verbose=False): 
        ### Everything should be passed as a NUMPY ARRAY
        # ------->  Remember to update self.state once (and only once) every time-step  <-------
        # Set the state variable to be used with Pinocchio
        q_pin = self.state.q[self.mask_q_pyb_to_pin]
        qd_pin = self.state.qd[self.mask_qd_pyb_to_pin]
        qdd_pin = self.state.qdd[self.mask_qd_pyb_to_pin]
        qdd_des_pin = qdd_des[self.mask_qd_pyb_to_pin]        
        # Set errors value
        ed = qd_des - self.state.qd
        ed = np.reshape(ed,(ed.shape[0],1))
        # the position error is different, since q has 1DOF more than qd;
        # to match the shape of the two error, the error term of the base configuration will be cosidered
        # [0,0,0,0,0,0], instead of computing it properly through quaternion rotation
        e = np.concatenate(([0.0,0.0,0.0,0.0,0.0,0.0],(q_des[self.mask_joints_shifted] - self.state.q[self.mask_joints_shifted])))
        print(self.state.q[self.mask_joints_shifted])
        q_des[self.mask_joints_shifted]
        e = np.reshape(e,(e.shape[0],1))
        ### DYNAMICS (Pinocchio library)
        # see https://github.com/stack-of-tasks/pinocchio/blob/master/bindings/python/multibody/data.hpp)
            # to check which attributes data may contain.
            # Each attribute should be set with the proper function, before being called
            # e.g. pin.crba() just set data.M to the correct value)
        # Since joints are ordered differenty in PyBullet and Pinocchio, when calling Pinocchio's functions
            # q_pin, qd_pin and qdd_pin should be passed instead of q,qd, and qdd, 
            # use the proper mask to generate them
        # Compute mass matrix with Pinocchio library 
        pin.crba(self.pinmodel, self.pin_data, q_pin)
        M = self.pin_data.M
        # Estimate of the disturbance term (eta) 
            # Computing the inverse dynamics, using the real term for qdd (not the desired one).
            # Stored in self.pin_data_eta to avoid overwriting terms that might be useful later.
            # pin.rnea() compute the left term of: M*qdd + C*qd + G = tau_real + eta
        pin.rnea(self.pinmodel,self.pin_data_eta,q_pin,qd_pin,qdd_pin)
        eta = self.pin_data_eta.tau[self.mask_qd_pin_to_pyb] - self.state.tau
        # INVERSE DYNAMICS WITH QDDA
        pin.rnea(self.pinmodel,self.pin_data,q_pin,qd_pin,qdd_des_pin)
        tau_act = self.pin_data.tau[self.mask_qd_pin_to_pyb] - eta
        #print("TAU_EXACT:  ", np.round(tau_exact,4))
        tau_act = np.reshape(tau_act, (tau_act.shape[0],1))
        #
        tau_act_closed_loop = tau_act + M.dot(Kv).dot(ed) + M.dot(Kp).dot(e)
        tau_act_closed_loop = np.reshape(tau_act_closed_loop, (tau_act_closed_loop.shape[0],))
        return tau_act_closed_loop, np.ndarray.flatten(e)

    def apply_torques(self, tau_des, filtered=True):
        # filtered version is better to avoid applying discontinuous torques to the joints
        if filtered:
            tau = self.low_pass_tau.filter(tau_des)
        else: 
            tau = tau_des
        #flatten control_indices to use for setting torques with a single loop
        index = [i for index_tuple in self.control_indices for i in index_tuple]
        #print("---->>> INDEX: ", index)
        for joint_i in index:
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.TORQUE_CONTROL,
                force=tau[joint_i + self.state.nqbd]
                )
        return tau
    
    def control_leg_abduction(self, RL, theta0, thetaf, ti, t_stance, fmax=1, positionGain=1, velocityGain=0.5):
        #ti = time from start of the stance phase, t_stance = total (desired) stance phase duration
        #RL = 0 for right leg, 1 for left leg
        RL=int(RL)
        if (RL!=0 and RL!=1):
            print("ERROR: RL must be either 0 or 1 (right or left stance)")
            return
        theta = (theta0+thetaf)/2 + (theta0-thetaf)*cos(pi*ti/t_stance)/2
        p.setJointMotorControl2(self.Id, 
            self.control_indices[1+RL][0],   #NOTE: must be modified if the leg is changed in the URDF!!!
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
            self.control_indices[1+RL][1],   #NOTE: must be modified if the leg is changed in the URDF!!!
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