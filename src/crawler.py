import time
from math import *

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
import scipy as sp
import scipy.linalg as lna


class Discrete_Low_Pass:
    """Create a first order discrete low-pass filter
    x(k+1) = (1-dt*fc)*x(k) + K*dt*fc*u(k)
    """
    def __init__(self, dim, dt, fc, K=1):
        """initialize the filter

        Parameters
        ----------
        dim : [float]
            dimension of the input signal (1D-array,
            each component filtered as a separate signal)\n
        dt : [float]
            sampling time\n
        fc : [float]
            cut-off frequency\n
        K : int, optional
            filter's gain, by default 1\n

        Warnings
        -------
        Each filter keeps a internal state, so a different filter object should 
            be initialized for each 1-dimensional signal.\n
        Different signals shouldn't be passed to the same filter.
        """
        self.dim = dim
        self.x = np.array([0]*self.dim)
        self.dt = dt
        self.fc = fc
        self.K = K
    def reset(self):
        """Reset filter's state to an array of 0s
        """
        self.x = np.array([0]*self.dim)
    def filter(self, signal):
        """Give input and update the filter's state(=output) accordingly

        Parameters
        ----------
        signal : [np.array(self.dim)]
            input signal

        Returns
        -------
        [np.array(self.dim)]
            filter state, equal to the output of the filter
        """
        # input signal should be a NUMPY ARRAY
        self.x = (1-self.dt*self.fc)*self.x + self.K*self.dt*self.fc * signal
        return self.x

class Integrator_Forward_Euler:
    """Forward-Euler integrator
    """
    def __init__(self, dt, x0):
        self.dt = dt
        self.x = np.array(x0)
    def integrate(self,xd):
        #xd should be passed as NUMPY ARRAY
        self.x = self.x + xd*self.dt
        return self.x.copy()
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
        self.low_pass = Discrete_Low_Pass(dim = self.nqd,dt=self.dt,fc=50)
    
    def set_low_pass(self, fc):
        self.low_pass = Discrete_Low_Pass(dim = self.nqd,dt=self.dt,fc=fc)

    def set_vel(self,qd_new):
        self.qd_prev = self.qd
        self.qd = qd_new
        self.qbd_prev = self.qd_prev[list(range(6))]
        self.qjd_prev = self.qd_prev[list(range(6,self.nqd))]
        self.qbd = self.qd[list(range(6))]
        self.qjd = self.qd[list(range(6,self.nqd))]

    def set_acc(self, filtered=False):
        if filtered:
            self.qdd = self.low_pass.filter((self.qd - self.qd_prev)/self.dt)
        else:
            self.qdd = (self.qd - self.qd_prev)/self.dt
        self.qbdd = self.qdd[list(range(6))]
        self.qjdd = self.qdd[list(range(6,self.nqd))]
    
    def update(self, filtered_acc=False):
        #NOTE: this function should be called only ONCE PER TIME-STEP, otherwise qdd will be set to 0
        info_qb = p.getBasePositionAndOrientation(self.modelId)
        info_qbd = p.getBaseVelocity(self.modelId)
        qbd_new = np.concatenate((info_qbd[0],info_qbd[1]))
        info_j = np.array(p.getJointStates(self.modelId,list(range(self.nqjd))))
        qjd_new = info_j[:,1].astype(np.double)
        qd_new = np.concatenate((qbd_new,qjd_new))
        self.qb = np.concatenate((info_qb[0],info_qb[1]))
        self.qj = info_j[:,0].astype(np.double)
        self.q = np.concatenate((self.qb,self.qj))
        self.set_vel(qd_new)
        self.set_acc(filtered=filtered_acc)
        self.tau[self.joint_indices_qd] = info_j[:,3].astype(np.double)

class Crawler:
    """
    Crawler class.\n
    An object of this class can be used to simulate in PyBullet a
    crawler.urdf model.\n
    See readme.md for a reference on how this model is built.

    Contain a Model_State, that is used to store and 
    """
    def __init__(self, dt_simulation, urdf_path="/home/fra/Uni/Tesi/crawler", base_position=[0,0,0.5], base_orientation=[0,0,0,1], mass_distribution=False, scale=1):
        """
        Instantiate a Crawler and spawn a crawler model in current PyBullet's physics server

        Parameters
        ----------
        dt_simulation : float
            time-step of PyBullet's simulation in which the model will be used\n
        urdf_path : str, optional
            path to folder in which "crawler.urdf" is, by default "/home/fra/Uni/Tesi/crawler"\n
        base_position : list, optional
            starting position of the base when spawning the model in PyBullet, by default [0,0,0.5]\n
        base_orientation : list [quaternion (x,y,z,w)], optional
            starting orientation of the base reference frame when spawning the model in PyBullet, by default [0,0,0,1]\n
        mass_distribution : bool, optional
            set to True to use "crawler_mass_distribution.urdf" instead of "crawler.urdf",
            they should be in the same folder, by default False\n
        scale : float, optional
            scale the dimension of the model (not the mass), by default 1

        Warnings
        ----------
        Physical properties in the first block (limited by "###") must match be manually matched
        to the values used in the XACRO that generates the URDF file\n

        Notes
        ----------
        Notable variables:\n

        self.state is an object of the class Model_State that is used
        to store and update the state of the joints of the model.\n
        q has 7 + #joints elements, since the base orientation is described by a quaternion.\n
        The derivatives of the state (qd and qdd) instead have 6 + #joints elements. \n
        The state vector is ordered as (base_pos, base_or, joints)\n

        self.control_indices contains the indices of the active joints, ordered as
        (spinal lateral joints, right shoulder joints, left shoulder joints)\n

        self.mask* are used for selecting the desired quantities from numpy.arrays containing data.
        Check inline comments for more on this\n


        """
        ### PHYSICAL PROPERTIES AND URDF ###
        self.scale=scale
        #NOTE: Properties in this block of code must be manually matched to those defined in the Xacro file
        self.spine_segments         = 5
        self.body_length            = self.scale * 0.076
        self.spine_segment_length   = self.body_length/(self.spine_segments+2)
        self.leg_length             = self.body_length/9.4
        self.body_sphere_radius     = self.body_length/16
        self.foot_sphere_radius     = self.body_sphere_radius/3
        self.neutral_contact_flexion_angle = asin((self.body_sphere_radius-self.foot_sphere_radius)/self.leg_length)-0.0001
        ###
        if not (mass_distribution):
            self.Id = p.loadURDF(
                "%s/crawler.urdf" % urdf_path, 
                base_position, 
                base_orientation, 
                globalScaling=self.scale,
                flags=p.URDF_USE_INERTIA_FROM_FILE)
        else:
            self.Id = p.loadURDF(
                "%s/crawler_mass_distribution.urdf" % urdf_path, 
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
        self.control_indices = self.generate_control_indices()
        self.constraints = {
            "right_foot": 0,
            "left_foot": 0,
            "last_link": 0
        }
        #
        self.COM_y_ref=self.COM_position_world()[1]
        ### MASKS ###
        # Masks for selecting row/columns of np.arrays, generally used with Jacobians or q/qd/qdd
        #NOTE: q include a quaternion for base orientation while qd has the angular speed of the base,
            # so q has 1 more element than qd. (But the quaternion must be a norm 1 vector, as usual)
        # --> All this masks are referred to qd (6+#joints elements) except  the "shifted" version
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
        self.mask_both_legs = list(range(6+self.num_joints-4,6+self.num_joints))
        self.mask_both_legs_shifted = list(range(7+self.num_joints-4,7+self.num_joints))
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
        # "virtual twin" of PyBullet's model that can be used to run 
        # functions from the Pinocchio library.
        # A "virtual twin" in an environment with zero gravity is also defined
        if not mass_distribution:
            self.pinmodel = pin.buildModelFromUrdf("%s/crawler.urdf" % urdf_path,pin.JointModelFreeFlyer())
            self.pinmodel_zerog = pin.buildModelFromUrdf("%s/crawler.urdf" % urdf_path,pin.JointModelFreeFlyer())
            self.pinmodel_zerog.gravity = pin.Motion.Zero()
            self.pinmodel_fixed = pin.buildModelFromUrdf("%s/crawler.urdf" % urdf_path)
            self.pinmodel_fixed_zerog = pin.buildModelFromUrdf("%s/crawler.urdf" % urdf_path)
            self.pinmodel_fixed_zerog.gravity = pin.Motion.Zero()
        else:
            self.pinmodel = pin.buildModelFromUrdf("%s/crawler_mass_distribution.urdf" % urdf_path,pin.JointModelFreeFlyer())
            self.pinmodel_zerog = pin.buildModelFromUrdf("%s/crawler_mass_distribution.urdf" % urdf_path,pin.JointModelFreeFlyer())
            self.pinmodel_zerog.gravity = pin.Motion.Zero()
            self.pinmodel_fixed = pin.buildModelFromUrdf("%s/crawler_mass_distribution.urdf" % urdf_path)
            self.pinmodel_fixed_zerog = pin.buildModelFromUrdf("%s/crawler_mass_distribution.urdf" % urdf_path)
            self.pinmodel_fixed_zerog.gravity = pin.Motion.Zero()
        self.pin_data = self.pinmodel.createData()
        self.pin_data_zerog = self.pinmodel_zerog.createData()
        self.pin_data_fixed = self.pinmodel_fixed.createData()
        self.pin_data_fixed_zerog = self.pinmodel_fixed_zerog.createData()
        self.pin_data_eta = self.pinmodel.createData()
        #self.pinmodel.gravity = pin.Motion.Zero()
        ### LAMBDA FUNCTIONS (to avoid redefining them inside each function)
        # for the generation of traveling wave angle variables
        self.num_lat = len(self.control_indices[0])
        self.trav_wave_theta = lambda t,i,A,f,th0: A*np.sin(-2*pi*f*t + pi*i/self.num_lat + th0)
        self.trav_wave_thetad = lambda t,i,A,f,th0: -A*2*pi*f*np.cos(2*pi*f*t + pi*i/self.num_lat + th0)
        self.trav_wave_thetadd = lambda t,i,A,f,th0: A*2*pi*f*2*pi*f*np.sin(2*pi*f*t + pi*i/self.num_lat + th0)
        
    def set_low_pass_lateral_qa(self, fc, K=1):
        """
        Set cut-off frequency for the low pass filter to be used on the spinal
        lateral joints desired velocities

        Parameters
        ----------
        fc : float
           cut-off frequency\n
        K : float, optional
            filter gain, set to 1 to avoid scaling the output value, by default 1
        """
        self.low_pass_lateral_qa = Discrete_Low_Pass(
            dim=len(self.mask_act),
            dt=self.dt_simulation, 
            fc=fc*self.dt_simulation, 
            K=K)

    def set_low_pass_qd(self, fc, K=1):
        """
        Set cut-off frequency for the low pass filter to be used on the state derivative before deriving
        it to obtain accelerations (if needed, currently not used)

        Parameters
        ----------
        fc : float
           cut-off frequency\n
        K : float, optional
            filter gain, set to 1 to avoid scaling the output value, by default 1
        """
        self.low_pass_qd = Discrete_Low_Pass(
            dim=(self.num_joints+6),
            dt=self.dt_simulation, 
            fc=fc*self.dt_simulation, 
            K=K)
    
    def set_low_pass_tau(self, fc, K=1):
        """
        Set cut-off frequency for the low pass filter acting on the tau applied to the joint

        Parameters
        ----------
        fc : float
           cut-off frequency\n
        K : float, optional
            filter gain, set to 1 to avoid scaling the output value, by default 1
        """
        self.low_pass_tau = Discrete_Low_Pass(
            dim=self.state.nqd,
            dt=self.dt_simulation, 
            fc=fc*self.dt_simulation, 
            K=K)
    
    def COM_position_world(self):
        """
        Returns
        -------
        np.array(3xfloat)
            COM position referred to global reference frame
        """
        COM = np.asarray(p.getBasePositionAndOrientation(self.Id)[0])*(p.getDynamicsInfo(self.Id, -1)[0])
        for i in range(0,self.num_joints):
            link_COM_pos = np.asarray(p.getLinkState(self.Id, i)[0])
            link_mass = p.getDynamicsInfo(self.Id, i)[0]
            COM += link_COM_pos*link_mass
        COM = (COM/self.mass)
        return COM

    def COM_velocity_world(self):
        """
        Returns
        -------
        np.array(3xfloat)
            COM velocity referred to global reference frame
        """
        #return the linear velocity of the center of mass in the world coordinates, as a NUMPY ARRAY
        COMv = np.asarray(p.getBaseVelocity(self.Id)[0])*(p.getDynamicsInfo(self.Id, -1)[0])
        for i in range(0,self.num_joints):
            link_COM_vel = np.asarray(p.getLinkState(self.Id, i, computeLinkVelocity=1)[6])
            link_mass = p.getDynamicsInfo(self.Id, i)[0]
            COMv += link_COM_vel*link_mass
        COMv = (COMv/self.mass)
        return COMv

    def set_COM_y_ref(self):
        """
        set the current COM y position as the one to measure error from
        (useful when trying to keep a fixed y position)
        """
        self.COM_y_ref = self.COM_position_world()[1]

    def turn_off_joint(self, joint_index):
        """
        Set a joint to move free without resistance (except friction and damping defined in the URDF)
        Parameters
        ----------
        joint_index : int
            index of the joint to turn off (PyBullet order)
        """
        p.setJointMotorControl2(self.Id, joint_index, controlMode=p.VELOCITY_CONTROL, force=0.0)

    def turn_off_crawler(self):
        """
        Apply turn_off_joint() to every joint in the model
        """
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
        """
        Fix the right foot with a spherical hinge joint at the center of the foot's spherical shape
        Returns
        -------
        int
            PyBullet's id for the constraint
        """
        if self.constraints["right_foot"]:
            constId=self.constraints["right_foot"]
            print("Error: remove right foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, self.control_indices[1][1], -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,self.control_indices[1][1])[0]))
            self.constraints["right_foot"]=constId
        return constId

    def fix_left_foot(self):
        """
        Fix the left foot with a spherical hinge joint at the center of the foot's spherical shape
        Returns
        -------
        int
            PyBullet's id for the constraint
        """
        if self.constraints["left_foot"]:
            constId=self.constraints["left_foot"]
            print("Error: remove left foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, self.control_indices[2][1], -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,self.control_indices[2][1])[0]))
            self.constraints["left_foot"]=constId
        return constId
    
    def free_right_foot(self):
        """
        remove right foot constraint
        """
        p.removeConstraint(self.constraints["right_foot"])
        self.constraints["right_foot"]=0

    def free_left_foot(self):
        """
        remove left foot constraint
        """
        p.removeConstraint(self.constraints["left_foot"])
        self.constraints["left_foot"]=0    

    def invert_feet(self):
        """
        Invert how the feet are constrained
        """
        if self.constraints["right_foot"]:
            self.free_right_foot()
        else:
            self.fix_right_foot()
        if self.constraints["left_foot"]:
            self.free_left_foot()
        else:
            self.fix_left_foot()

    def fix_tail(self, second_last = False):
        """
        Fix the last link with a spherical hinge joint at the center of the spherical shape
        Returns
        -------
        int
            PyBullet's id for the constraint
        """
        #constraint is generated at the origin of the center of mass of the last link
        if self.constraints["last_link"]:
            constId=self.constraints["last_link"]
            print("Error: remove right foot constraint before setting a new one")
        else:
            last_link_index = self.control_indices[0][-1]+1 - 2*int(second_last)
            constId = p.createConstraint(self.Id, last_link_index, -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0], childFramePosition=list(p.getLinkState(self.Id,last_link_index)[0]))
            self.constraints["last_link"]=constId
        print("world_pos_const: ", list(p.getLinkState(self.Id,last_link_index)[0]))
        return constId
    
    def free_tail(self):
        """
        remove tail constraint
        """
        p.removeConstraint(self.constraints["last_link"])
        self.constraints["last_link"]=0 

    def set_links_state_array(self):
        """
        Update the data in self.links_state_array
        """
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
        """
        Returns a tuple with the current position of each joint of the model
        """
        return list(zip(*(p.getJointStates(self.Id,list(range(0,self.num_joints))))))[0]

    def get_base_Eulers(self):
        """
        Returns
        -------
        list [3xfloat]
            Euler's angle (XYZ) desribing the base orientation
        """
        return  p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.Id)[1])
    
    def get_R_base_to_world(self):
        """
        Returns
        -------
        np.array[3,3 matrix]
            Rotation matrix such that R * x_base = x_world
        """
        # returned as a (3,3) NUMPY ARRAY
        # R * x_base = x_world
        quat_base = p.getBasePositionAndOrientation(self.Id)[1]
        R = np.reshape(np.array(p.getMatrixFromQuaternion(quat_base)),(3,3))
        return R
    
    def get_link_COM_jacobian_trn_world(self, link_index, R):
        """
        Compute the Jacobian for the translational velocity of COM of a single link,
        referred to world global coordinates\n
        vCOMi = J*qd

        Parameters
        ----------
        link_index : int\n

        R
            should be passed already computed through the proper class method, self.get_R_base_to_world()

        Returns
        -------
        np.array[3,n matrix]
            Jacobian matrix

        Notes
        -------
        Since p.calculateJacobian() outputs a Jacobian that gives the linear/angular velocity expressed 
        in a reference frame oriented like the base but fixed in the global reference frame.\n
        This Jacobian is transformed to be used with values expressed in global coordinates (see Thesis report)\n
        Furthemore, p.calculateJacobian() gives a Jacobian with first the base rotation and then the base translation:
        this Jacobian is rearranged so that it should multiply a state 
        composed as (base_translation, base_rotation, joints), as usual
        """
        joints_pos = self.get_joints_pos_tuple()
        J = np.asarray(
            p.calculateJacobian(self.Id,
                link_index,
                self.links_state_array[link_index]["loc_com_trn"],
                joints_pos,
                #self.state.qj.tolist(),
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
        """
        Compute the translational Jacobian of the center of the mass of the whole model, referred to
        global reference frame

        Returns
        -------
        np.array[3,n matrix]
            Jacobian matrix

        Notes
        -------
        Jacobian of the COM is computed as the weighted mean (with respect to masses) of the Jacobian of the links.\n
        The transational jacobian of the base is just the identity matrix multiplying the base translational velocity
        plus null terms associated to the base angular speed and the joints speed terms (thus 3+self.num_joints).\n
        Angular speed values are null just because the COM of the girdle link coincides with the origin of the link frame.
        """
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

    def solve_null_COM_y_speed_optimization(self, K, k0=1, verbose=False):
        """
        Solve Closed-loop Inverse Kinematics for the spinal lateral joints,
        for having a null y-axis speed of the model's COM

        Parameters
        ----------
        K : float
            Gain of the closed loop \n
        k0 : float, optional
            parameters for the gradient descent, used for choosing the values of the additional joints speeds
            in the nullspace of the Jacobian, by default 1\n
        verbose : bool, optional
             by default False

        Returns
        -------
        tuple
            (qda, eCOMy)\n
            qda = np.array, desired spinal lateral joints' velocities\n
            eCOMy = float, error on the position of the center of mass, computed using the value set with 
            self.set_COM_y_ref()

        Notes
        -------
        See report for the mathematical description
        self.COM_y_ref() should be called once at the start of each step phase.\n
        Since the dorsal joints of the spine and the DOFs of the base are not actuated, xd_desired not equal to 0 but
        is corrected (see Thesis' report). The joints of the girdles are also not considered as actuated since their 
        speed is set independently.\n
        An additional control target can be followed through the choice q0da, that doesn't affect the primary 
        target due to the projection in the Jacobian's null space done by P. Here qd0a is chosen as the 
        gradient descent of a function to optimize.
        """
        #q0d are projected inside the nullspace of the Jacobian (J) and can be chosen to minimize a cost function
            # Here, the chosen cost function is H(q)=sum(q_i ^2), to keep joints as near to 0 as possible while
            # respecting the requirements given by inverse kinematics.
            # H(q) minimum is found through gradient descent, by chosing q0d=-k0*gradient(H(q))
            # (easy for quadratic H(q))
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
            print("bd\n", np.round(bd,3))
            print("qda\n", np.round(qda,3))
            print("Jyb: ", np.ndarray.flatten(np.round(Jyb,4)))
            print("Jy: ", np.round(Jy,4))
            print("xd_desired = ", xd_desired)
            print("qdn", np.ndarray.flatten(qdn))
            print("Jyb.dot(bd) = ", Jyb.dot(bd))
            print("Jyn.dot(qdn)", Jyn.dot(qdn))
            print("qda: ", qda)
        return (qda, eCOMy)

    def generate_fmax_array_lateral(self,fmax_last):
        """
        Generate an array with the max force value of the lateral joints

        Parameters
        ----------
        fmax_last : float
            maximum force(/torque) that the last spinal joint should exert

        Returns
        -------
        list(float)

        Notes
        -------
        The maximum forces are computed by increasing linearly fmax_last toward the center of the spine
        (and then decreasing it again for the joints toward the head),
        since the mid-section of the body needs to move a greater mass.\n
        The generated array is meant to be used when calling functions like p.SetMotorControlArray()
        with position or velocity control mode, in particular its made to be passed directly
        when calling the method self.controlV_spine_lateral()
        """
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
    
    def solve_null_COM_y_speed_optimization_qdda(self, qda_prev, K,k0=1, filtered=True):
        """
        Run Closed-loop Inverse Kinematics and compute also the desired position and accelerations,
        integrating and derivating qda with backward Euler's approximation

        Parameters
        ----------
        qda_prev : np.array(float)
            value of the desired joints' speeds during the last time-step\n
        K : float
            CLIK gain, to be passed to self.solve_null_COM_y_speed_optimization()\n
        k0 : float, optional
            to be passed to self.solve_null_COM_y_speed_optimization(), by default 1\n
        filtered : bool, optional
            whether to filter qda using self.low_pass_lateral_qa, by default True

        Returns
        -------
        tuple(np.array,np.array,np.array)
            (qa,qda,qdda)
        """
        # Compute also qa and qdda from the qda computed inside self.solve_null_COM_y_speed_optimization().
        # This is probably better in its filtered version, 
            # since this is the desired trajectory to track at joints level
        kin_sol = self.solve_null_COM_y_speed_optimization(K=K,k0=k0)
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
    
    def generate_abduction_trajectory(self, theta0, thetaf, ti, t_stance, cos_fun=True):
        """
        Generate desired position, speed and acceleration for an abduction joint at time ti, following a cosinusoidal profile
        that have null speed at the extremities of the trajectory

        Parameters
        ----------
        theta0 : float
            upper limit of the abduction\n
        thetaf : float
            lower limit of the abduction\n
        ti : float
            time\n
        t_stance : float
            duration of the stance phase (50% symmetric duty cycle, equal to 1/2 of step duration)\n
        cos_fun : bool, optional
            if False use a linear profile instead of a cosine, by default True\n

        Returns
        -------
        tuple(3xfloat)
            (theta, thetad, thetadd)
        """
        #ti = time from start of the walking locomotion, t_stance = total (desired) stance phase duration
        if cos_fun:
            theta = (theta0+thetaf)/2 + (theta0-thetaf)*cos(pi*ti/t_stance)/2
            thetad = -pi*(theta0-thetaf)*sin(pi*ti/t_stance)/(2*t_stance)
            thetadd = -pi*pi*(theta0-thetaf)*cos(pi*ti/t_stance)/(2*t_stance*t_stance)
        else:
            theta = theta0 + (thetaf-theta0)*ti/t_stance
            thetad = (thetaf-theta0)/t_stance
            thetadd = 0
        return theta, thetad, thetadd
    
    def generate_joints_trajectory(self, theta0, thetaf, ti, t_stance, qa_des, qda_des, qdda_des, cos_abd=True):
        """
        Generate the desired position, speed and acceleration for all the joints of the model

        Parameters
        ----------
        theta0 : float
            upper limit of leg abduction\n
        thetaf : float
            upper limit of leg abduction\n
        ti : float
            time\n
        t_stance : float
            duration of a single step (50% symmetric duty cycle)\n
        qa_des : np.array
            array of the desired position for the spinal lateral joints.
            Can be generated using self.solve_null_COM_y_speed_optimization_qdda()\n
        qda_des : np.array
            array of the desired speed for the spinal lateral joints.
            Can be generated using self.solve_null_COM_y_speed_optimization_qdda()\n
        qdda_des : np.array
            array of the desired acceleration for the spinal lateral joints.
            Can be generated using self.solve_null_COM_y_speed_optimization_qdda()\n
        cos_abd : bool, optional
            Whether to follow a cosinusoidal function for the abduction of the legs, by default True

        Returns
        -------
        (q_des, qd_des, qdd_des, e)
            q_des, qd_des, qdd_des = desired values of speed,acceleration of the model's joints\n
            e = between q_des and the actual joints' positions
        """
        #ti = time from start of the walking locomotion, t_stance = total (desired) stance phase duration
        # RL=0 if right leg stance, 1 if left leg stance (stance leg is the one with the constraied foot)
        # q[3:6] must be a quaternion
        q_des = self.state.q.copy()
        qd_des = self.state.qd.copy()
        qdd_des = self.state.qdd.copy()
        # Right leg abduction trajectory
        right_abd = self.generate_abduction_trajectory(theta0,thetaf,ti,t_stance, cos_fun=cos_abd)
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
        # # Right leg flexion
        q_des[(7 + self.control_indices[1][1])] = self.neutral_contact_flexion_angle
        qd_des[(6 + self.control_indices[1][1])] = 0
        qdd_des[(6 + self.control_indices[1][1])] = 0
        # Left leg flexion
        q_des[(7 + self.control_indices[2][1])] = -self.neutral_contact_flexion_angle
        qd_des[(6 + self.control_indices[2][1])] = 0
        qdd_des[(6 + self.control_indices[2][1])] = 0
        # Lateral spinal joints
        q_des[self.mask_act_shifted] = qa_des 
        qd_des[self.mask_act] = qda_des 
        qdd_des[self.mask_act] = qdda_des
        # q_des[self.mask_act_shifted] = np.array([1]*len(self.mask_act_shifted))*0.1*sin(ti*pi)
        # qd_des[self.mask_act] = np.array([1]*len(self.mask_act_shifted))*pi*0.1*cos(ti*pi)
        # qdd_des[self.mask_act] = np.array([1]*len(self.mask_act_shifted))*(-pi*pi*0.1*sin(ti*pi))
        # position error, base is set to 0
        e = np.concatenate(([0.0,0.0,0.0,0.0,0.0,0.0],(q_des[self.mask_joints_shifted] - self.state.q[self.mask_joints_shifted])))
        return q_des, qd_des, qdd_des, e
    
    def generate_joints_trajectory_stop(self, theta0, thetaf, ti, t_stance, qa_des, qda_des, qdda_des):
        """[DEPRECATED - ONLY FOR TESTING]
        """
        #ti = time from start of the walking locomotion, t_stance = total (desired) stance phase duration
        # RL=0 if right leg stance, 1 if left leg stance (stance leg is the one with the constraied foot)
        # q[3:6] must be a quaternion
        q_des = self.state.q.copy()
        qd_des = self.state.qd.copy()
        qdd_des = self.state.qdd.copy()
        # Right leg abduction trajectory
        right_abd = self.generate_abduction_trajectory(theta0,thetaf,t_stance,t_stance)
        q_des[(7 + self.control_indices[1][0])] = right_abd[0]
        qd_des[(6 + self.control_indices[1][0])] = right_abd[1]
        qdd_des[(6 + self.control_indices[1][0])] = 0
        # Left leg abdution trajectory, 
            # same as the right leg but translated backward temporally and with an opposite sign 
            # (see joint's reference frame)
        left_abd = self.generate_abduction_trajectory(theta0,thetaf,(t_stance-t_stance),t_stance)
        q_des[(7 + self.control_indices[2][0])] = -(left_abd[0])
        qd_des[(6 + self.control_indices[2][0])] = -(left_abd[1])
        qdd_des[(6 + self.control_indices[2][0])] = 0
        # # Right leg flexion
        # q_des[(7 + self.control_indices[1][1])] = self.neutral_contact_flexion_angle
        # qd_des[(6 + self.control_indices[1][1])] = 0
        # qdd_des[(6 + self.control_indices[1][1])] = 0
        # # Left leg flexion
        # q_des[(7 + self.control_indices[2][1])] = -self.neutral_contact_flexion_angle
        # qd_des[(6 + self.control_indices[2][1])] = 0
        # qdd_des[(6 + self.control_indices[2][1])] = 0
        # Lateral spinal joints
        q_des[self.mask_act_shifted] = qa_des
        qd_des[self.mask_act] = qda_des
        qdd_des[self.mask_act] = qdda_des
        # position error, base is set to 0
        e = np.concatenate(([0.0,0.0,0.0,0.0,0.0,0.0],(q_des[self.mask_joints_shifted] - self.state.q[self.mask_joints_shifted])))
        return q_des, qd_des, qdd_des, e
    
    def solve_computed_torque_control(self, q_des, qd_des, qdd_des, Kp, Kv, verbose=False):
        """Computed Torque Control (CTC) for following the desired joints' trajectories
        
        Parameters
        ----------
        q_des : np.array
            desired joints' positions\n
        qd_des : np.array
            desired joints' positions\n
        qdd_des : np.array
            desired joints' positions\n
        Kp : np.array(#joints,#joints matrix)
            proportional gain matrix for the CTC\n
        Kv : np.array(#joints,#joints matrix)
            derivative gain matrix for the CTC\n
        verbose : bool, optional
            by default False

        Returns
        -------
        (tau, e)
            tau = np.array of the torques to apply to the joints
            e = joints' positional error

        Notes
        -------
        Exploit the Pinocchio library to compute inverse dynamics and the mass matrix.\n
        External disturbances (forces from contacts and friction) are estimated for the current time-step.
        See inline comments and Thesis report for details
        """
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
        tau_act = self.pin_data.tau[self.mask_qd_pin_to_pyb] #- eta
        #print("TAU_EXACT:  ", np.round(tau_exact,4))
        tau_act = np.reshape(tau_act, (tau_act.shape[0],1))
        #
        tau_act_closed_loop = tau_act + M.dot(Kv).dot(ed) + M.dot(Kp).dot(e)
        tau_act_closed_loop = np.reshape(tau_act_closed_loop, (tau_act_closed_loop.shape[0],))
        return tau_act_closed_loop, np.ndarray.flatten(e)
    
    def solve_computed_torque_control_sliding(self, q_des, qd_des, qdd_des, Kp, Kv, rho, verbose=False): 
        """[DEPRECATED]
        Similar to self.solve_computed_torque_control() but add a sliding term to apply a continuous sliding mode
        control.
        Currently doesn't really improve the CTC, possibily due to discretization (chattering) and bad disturbance
        estimation (should be modifed to exploit the disturbance estimate from the CTC)
        """
        ### Everything should be passed as a NUMPY ARRAY
        # See comments in the function solve_computed_torque_control
        q_pin = self.state.q[self.mask_q_pyb_to_pin]
        qd_pin = self.state.qd[self.mask_qd_pyb_to_pin]
        qdd_pin = self.state.qdd[self.mask_qd_pyb_to_pin]
        qdd_des_pin = qdd_des[self.mask_qd_pyb_to_pin]        
        # Set errors value
        ed = qd_des - self.state.qd
        ed = np.reshape(ed,(ed.shape[0],1))
        e = np.concatenate(([0.0,0.0,0.0,0.0,0.0,0.0],(q_des[self.mask_joints_shifted] - self.state.q[self.mask_joints_shifted])))
        e = np.reshape(e,(e.shape[0],1))
        delta = self.compute_sliding_delta(Kp=Kp, Kv=Kv, rho=rho, e=e, ed=ed)
        # print("DELTA:\n", delta)
        # print("E:\n", e, "\n")
        delta = np.reshape(delta,(delta.shape[0],1))
        ###
        pin.crba(self.pinmodel, self.pin_data, q_pin)
        M = self.pin_data.M
        # # Estimate of the disturbance term (eta) 
        pin.rnea(self.pinmodel,self.pin_data_eta,q_pin,qd_pin,qdd_pin)
        eta = self.pin_data_eta.tau[self.mask_qd_pin_to_pyb] - self.state.tau
        # INVERSE DYNAMICS WITH QDDA
        pin.rnea(self.pinmodel,self.pin_data,q_pin,qd_pin,qdd_des_pin)
        tau_act = self.pin_data.tau[self.mask_qd_pin_to_pyb] - eta
        #print("TAU_EXACT:  ", np.round(tau_exact,4))
        tau_act = np.reshape(tau_act, (tau_act.shape[0],1))
        #
        tau_act_closed_loop = tau_act + M.dot(Kv).dot(ed) + M.dot(Kp).dot(e) + M.dot(delta)
        print("tau_act[20]          ", tau_act[20])
        print("M.dot(Kv).dot(ed)    ", (M.dot(Kv).dot(ed))[20])
        print("M.dot(Kp).dot(e)     ", (M.dot(Kp).dot(e))[20])
        print("M.dot(delta) ", (M.dot(delta))[20])
        tau_act_closed_loop = np.reshape(tau_act_closed_loop, (tau_act_closed_loop.shape[0],))
        return tau_act_closed_loop, np.ndarray.flatten(e)

    def compute_sliding_delta(self, Kp, Kv, rho, e, ed, q=100):
        """
        Variable structure control.
        Compute the sliding additional term used by self.solve_computed_torque_control_sliding(), see docstring for it
        """
        A11 = np.empty(Kp.shape)
        A12 = np.eye(Kv.shape[1])
        A21 = - Kp.copy()
        A22 = - Kv.copy()
        A = np.block([[A11, A12],[A21, A22]])
        Q = q * np.eye(A.shape[0])
        P = lna.solve_continuous_lyapunov(A, Q)
        B1 = A11.copy()
        B2 = A12.copy()
        B = np.vstack((B1,B2))
        x = np.vstack((e,ed))
        sliding_vector = (B.T).dot((P.T).dot(x))
        delta = rho * sliding_vector/(lna.norm(sliding_vector))
        return delta

    def apply_torques(self, tau_des, filtered=True):
        """Apply desired torques to the joints

        Parameters
        ----------
        tau_des : [np.array((6+#joints,),float)]
            array containing the torque desired for each DOF.
            Torques from this array will only be applied to the actuated DOF (shoulders' and spinal lateral joints).\n
        filtered : bool, optional
            Filter the desired torques with self.low_pass_tau, by default True.\n
            Filtered version is better to avoid applying discontinuous torques to the joints

        Returns
        -------
        [np.array((6+#joints,),float)]
            Array with the applied torques. If filtered=False it's equal to the input tau_des
        """
        # 
        if filtered:
            tau = self.low_pass_tau.filter(tau_des)
        else: 
            tau = tau_des
        #flatten control_indices to use for setting torques with a single loop
        indexes = [i for index_tuple in self.control_indices for i in index_tuple]
        for joint_i in indexes:
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.TORQUE_CONTROL,
                force=tau[joint_i + self.state.nqbd]
                )
            #print("joint_i: ", joint_i, "tau %d: " %(joint_i + self.state.nqbd), np.round(tau[joint_i + self.state.nqbd],3))
        ### Uncomment to use velocity control on the flexion movement
        # p.setJointMotorControl2(
        #         self.Id, 
        #         self.control_indices[1][1], 
        #         p.VELOCITY_CONTROL,
        #         targetVelocity=0,
        #         force=1
        #         )
        # p.setJointMotorControl2(
        #         self.Id, 
        #         self.control_indices[2][1], 
        #         p.VELOCITY_CONTROL,
        #         targetVelocity=0,
        #         force=1
        #         )
        return tau
    
    def generate_Kp(self, Kp_lat, Kp_r_abd, Kp_l_abd, Kp_flex):
        """Generate proportional gain matrix for Computed Torque Control

        Parameters
        ----------
        Kp_lat : [float]
            minimum proportional gain of the lateral joints\n
        Kp_r_abd : [float]
            right-abduction proportional gain\n
        Kp_l_abd : [float]
            left-abduction proportional gain\n
        Kp_flex : [float]
            flexion proportional gain

        Returns
        -------
        [np.array()]
            Proportional gain matrix
        
        Notes
        -------
        The lateral gain are generated starting from Kp_lat and scaling them exponentially
        (1.2 factor) toward the center of the body.\n
        See line with comment "# i-th lateral joint" to modify how they are scaled.
        """
        nlat_joints = (self.state.nqd-6-4)//2
        Kp_spinal_list = list()
        for i in reversed(range(1,nlat_joints+1)):
            Kp_spinal_list.append(Kp_lat*((1.2)**i))   # i-th lateral joint
            Kp_spinal_list.append(0)            # i-th dorsal joint
        Kp_diag_list = (
            [0,0,0,0,0,0] +
            Kp_spinal_list + 
            [Kp_r_abd, Kp_flex] +
            [Kp_l_abd, Kp_flex]
            )
        print(Kp_diag_list)
        Kp = np.diag(Kp_diag_list)
        return Kp

    def generate_Kv(self, Kv_lat, Kv_r_abd, Kv_l_abd, Kv_flex):
        """Generate derivative gain matrix for Computed Torque Control

        Parameters
        ----------
        Kv_lat : [float]
            minimum derivative gain of the lateral joints\n
        Kv_r_abd : [float]
            right-abduction derivative gain\n
        Kv_l_abd : [float]
            left-abduction derivative gain\n
        Kv_flex : [float]
            flexion derivative gain

        Returns
        -------
        [np.array()]
            Derivative gain matrix
        
        Notes
        -------
        The lateral gain are generated starting from Kv_lat and scaling them exponentially
        (1.2 factor) toward the center of the body.\n
        See line with comment "# i-th lateral joint" to modify how they are scaled.
        """
        nlat_joints = (self.state.nqd-6-4)//2
        Kv_spinal_list = list()
        for i in reversed(range(1,nlat_joints+1)):
            Kv_spinal_list.append(Kv_lat*((1.2)**i))   # i-th lateral joint
            Kv_spinal_list.append(0)            # i-th dorsal joint
        Kv_diag_list = (
            [0,0,0,0,0,0] +
            Kv_spinal_list + 
            [Kv_r_abd, Kv_flex] +
            [Kv_l_abd, Kv_flex]
            )
        print(Kv_diag_list)
        Kv = np.diag(Kv_diag_list)
        return Kv

    def generate_fmax_list(self, fmax_lat, fmax_r_abd, fmax_l_abd, fmax_flex):
        """Generate the list of maximum force allowed for the joints, to be used with
        the P or PD control already implemented in PyBullet

        Parameters
        ----------
        fmax_lat : [float]
            minimum value of the max torque of the lateral joints.
            See inline comments to see how this value is scaled along the body\n
        fmax_r_abd : [float]
            max right-abduction torque\n
        fmax_l_abd : [float]
            max left-abduction torque\n
        fmax_flex : [float]
            max flexion torque

        Returns
        -------
        [list(#joints)]
            List of maximum torques that the joints can apply, to be used with to p.SetJointMotorControl2.
            The list include 0 for the passive spinal dorsal joints, to keep index consistency and allow selection 
            through self.control_indices
        """
        # 
        nlat_joints = (self.state.nqd-6-4)//2
        lat_multiplier = list(range(1,1+(nlat_joints//2))) + list(reversed(range(1, 1+(nlat_joints+1)//2)))
        fmax_spinal_list = list()
        #If the next line is not commented the lateral joints will have decreasing values
            ## starting from the girdle, otherwise they will have the max value at the center of the body
            ## If the foot act as fixed constraint leave this line uncommented.
        #lat_multiplier = list(reversed(range(1,nlat_joints+1)))
        for val in lat_multiplier:
            fmax_spinal_list.append(val*fmax_lat)   # i-th lateral joint
            fmax_spinal_list.append(0)              # i-th dorsal joint
        fmax_list = (
            fmax_spinal_list + 
            [fmax_r_abd, fmax_flex] +
            [fmax_l_abd, fmax_flex]
            )
        return fmax_list
    
    def generate_pGain_list(self, pGain_lat, pGain_r_abd, pGain_l_abd, pGain_flex):
        # Generate the list of pGain for the joints, to be used with PD control
        nlat_joints = (self.state.nqd-6-4)//2
        lat_multiplier = list(range(1,1+(nlat_joints//2))) + list(reversed(range(1, 1+(nlat_joints+1)//2)))
        pGain_spinal_list = list()
        #If the next line is not commented the lateral joints will have decreasing values
            # starting from the girdle, otherwise they will have the max value at the center of the body
            # If the foot act as fixed contraint leave this line uncommented.
            # pGain_lat is always the minimum value
        #lat_multiplier = list(reversed(range(1,nlat_joints+1)))
        for val in lat_multiplier:
            pGain_spinal_list.append(val*pGain_lat)   # i-th lateral joint
            pGain_spinal_list.append(0)              # i-th dorsal joint
        pGain_list = (
            pGain_spinal_list + 
            [pGain_r_abd, pGain_flex] +
            [pGain_l_abd, pGain_flex]
            )
        return pGain_list
    
    def generate_vGain_list(self, vGain_lat, vGain_r_abd, vGain_l_abd, vGain_flex):
        # Generate the list of vGain for the joints, to be used with PD control
        nlat_joints = (self.state.nqd-6-4)//2
        lat_multiplier = list(range(1,1+(nlat_joints//2))) + list(reversed(range(1, 1+(nlat_joints+1)//2)))
        vGain_spinal_list = list()
        #If the next line is not commented the lateral joints will have decreasing values
            # starting from the girdle, otherwise they will have the max value at the center of the body
            # If the foot act as fixed contraint leave this line uncommented.
            # vGain_lat is always the minimum value
        #lat_multiplier = list(reversed(range(1,nlat_joints+1)))
        for val in lat_multiplier:
            vGain_spinal_list.append(val*vGain_lat)   # i-th lateral joint
            vGain_spinal_list.append(0)              # i-th dorsal joint
        vGain_list = (
            vGain_spinal_list + 
            [vGain_r_abd, vGain_flex] +
            [vGain_l_abd, vGain_flex]
            )
        return vGain_list
    
    def PD_control(self, q_des, fmax_list, pGain_list, vGain_list, include_base=True):
        index = [i for index_tuple in self.control_indices for i in index_tuple]
        if include_base:
            increment=self.state.nqbd
        else:
            increment=0
        for joint_i in index:
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.POSITION_CONTROL,
                targetPosition=q_des[joint_i + increment],
                force=fmax_list[joint_i],
                positionGain = pGain_list[joint_i],
                velocityGain = vGain_list[joint_i]
                )
        return q_des
    
    def P_control_vel(self, qd_des, fmax_list, pGain_list):
        index = [i for index_tuple in self.control_indices for i in index_tuple]
        for joint_i in index:
            p.setJointMotorControl2(
                self.Id, 
                joint_i, 
                p.VELOCITY_CONTROL,
                targetVelocity=qd_des[joint_i + self.state.nqbd],
                force=fmax_list[joint_i],
                velocityGain = pGain_list[joint_i],
                )
        return qd_des
    
    def control_leg_abduction(self, RL, theta0, thetaf, ti, t_stance, fmax=1., positionGain=1., velocityGain=0.5):
        """Call p.setJointMotorControl2() for the abduction joint of the selected leg, to follow
            report's trajectory (cosine function)
        
        Parameters
        ----------
        RL : [int] 0 or 1
            0 for right leg, 1 for left leg\n
        theta0 : [float, radians]
            check report on cosine function for abduction\n
        thetaf : [float, radians]
            check report on cosine function for abduction\n
        ti : [float]
            time from the start of the current step\n
        t_stance : [type]
            stance phase duration (50% symmetric duty cycle)\n
        fmax : flaot, optional
            max torque, by default 1.\n
        positionGain : int, optional
            p.setJointMotorControl2() parameter, by default 1.\n
        velocityGain : float, optional
            p.setJointMotorControl2() parameter, by default 0.5
        """
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
        """Call p.setJointMotorControl2() for the flexion joint of the selected leg,
        to keep position at self.neutral_contact_angle
        
        Parameters
        ----------
        RL : [int] 0 or 1
            0 for right leg, 1 for left leg\n
        fmax : flaot, optional
            max torque, by default 1.\n
        positionGain : int, optional
            p.setJointMotorControl2() parameter, by default 1.\n
        velocityGain : float, optional
            p.setJointMotorControl2() parameter, by default 0.5
        """
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
    
    def set_bent_position(self, theta_rg, theta_lg, A_lat, theta_lat_0):
        """Set the model in a bent position, position of the spinal lateral joints is
        generated using a traveling wave.

        Parameters
        ----------
        theta_rg : [float]
            right abduction angle\n
        theta_lg : [float]
            left abduction angle\n
        A_lat : [float]
            amplitude of the traveling wave\n
        theta_lat_0 : [float]
            offset of the traveling wave

        Notes
        -------
        Traveling wave's wavelength is twice the body-length

        """
        #Body is set using traveling wave equation for t=0
        n_lat = len(self.control_indices[0])
        for i, joint_i in enumerate(self.control_indices[0]):
            theta_lat_i = A_lat*sin(2*pi*(i)/(2*n_lat) + theta_lat_0)
            p.resetJointState(self.Id, joint_i, theta_lat_i)
        p.resetJointState(self.Id, self.control_indices[1][0],theta_rg)
        p.resetJointState(self.Id, self.control_indices[2][0],theta_lg)
        # p.resetJointState(self.Id, self.control_indices[1][1],self.neutral_contact_flexion_angle)
        # p.resetJointState(self.Id, self.control_indices[2][1],-self.neutral_contact_flexion_angle)
        return
    
    def traveling_wave_lateral_trajectory_t(self,t,A,f,th0):
        """Use a traveling wave equation to generate desired trajectory for the spinal lateral joints

        Parameters
        ----------
        t : [float]
            time\n
        A : [float]
            amplitude of the traveling wave\n
        f : [float]
            frequency of the traveling wave, tipically 1/(2*t_stance)\n
        th0 : [float]
            offset of the traveling wave

        Returns
        -------
        [tuple]
            (qa,qda,qdda) tuple of numpy.array containing desired pos,vel,acc ONLY of the spinal lateral joints
        """
        # f = frequency, set to 2*stance_duration (it's the frequency of the walking behaviour)
        n_lat = len(self.control_indices[0])
        qa = np.arange(n_lat)
        qda = np.arange(n_lat)
        qdda = np.arange(n_lat)
        qa = self.trav_wave_theta(t,qa,A,f,th0)
        qda = self.trav_wave_thetad(t,qda,A,f,th0)
        qdda = self.trav_wave_thetadd(t,qdda,A,f,th0)
        return qa, qda, qdda
    
    def abduction_trajectory_t(self, t, th0, thf, f, th_offset):
        """it's almost a duplicate of self.generate_abduction_trajectory().
        

        Parameters
        ----------
        t : [float]
            Time\n
        th0 : [float]
            Start angle of the trajectory. See report on the cosine function used for abduction\n
        thf : [float]
            Finish angle of the trajectory. See report on the cosine function used for abduction\n
        f : [float]
            Frequency of the movement, tipically 1/(2*t_stance)\n
        th_offset : [type]
            Offset of the trajectory. Tipically left leg has a Pi offset with respect to the right one\n

        Returns
        -------
        [tuple]
            desired pos,vel,acc for a joint

        Notes
        -------
        Include the th_offset input so that, instead of selecting right or left leg,
        trajectories can be generated with the same function just by using an offset equal to Pi
        """
        theta = (th0+thf)/2 + (th0-thf)*cos(2*pi*f*t + th_offset)/2
        thetad = -2*pi*f*(th0-thf)*sin(2*pi*f*t + th_offset)/2
        thetadd = -2*pi*f*2*pi*f*(th0-thf)*cos(2*pi*f*t + th_offset)/2
        return theta, thetad, thetadd
    
    def compose_joints_trajectories_t(self, t, f, A_lat,th0_lat, th0_abd, thf_abd, include_base = False):
        """Generates arrays with the desired pos,vel,acc of all the joints,
        ordered correctly following PyBullet indexing of model's joints, at time=t

        Parameters
        ----------
        t : [float]
            Time\n
        f : [float]
            Frequency of the movements\n
        A_lat : [float]
            Amplitude of the traveling wave for spinal lateral joint trajectories\n
        th0_lat : [float]
            Offset of the traveling wave for spinal lateral joint trajectories\n
        th0_abd : [float]
            Start angle of the abduction trajectory\n
        thf_abd : [float]
            Finish angle of the abduction trajectory\n
        include_base : bool, optional
            Prepend the state of the base (and its derivative) to the returned array, by default False

        Returns
        -------
        [tuple]
            desired pos,vel,acc of all the joints of the model
        
        Notes
        -------
        Trajectories are generated using a traveling wave for the spinal lateral joints,
        the report's cosine function for the abduction joints. Check code for the flexion movement.
        Desired values of pos,vel,acc of the passive joints (and the base) are set to the current real value.
        """
        #
        # similar to generate_joints_trajectories()
        qj = np.zeros(self.num_joints)
        qdj = np.zeros(self.num_joints)
        qddj = np.zeros(self.num_joints)
        # body
        qa,qda,qdda = self.traveling_wave_lateral_trajectory_t(t,A_lat,f,th0_lat)
        qj[self.mask_act_nobase] = qa
        qdj[self.mask_act_nobase] = qda
        qddj[self.mask_act_nobase] = qdda
        # right abduction
        qra,qdra,qddra = self.abduction_trajectory_t(t,th0_abd,thf_abd,f,th_offset=0)
        qj[self.control_indices[1][0]] = qra
        qdj[self.control_indices[1][0]] = qdra
        qddj[self.control_indices[1][0]] = qddra
        # right flexion
        qj[self.control_indices[1][1]] = self.neutral_contact_flexion_angle + 0.03*np.sin(2*pi*f*t)
        qdj[self.control_indices[1][1]] = 0. + 0.03*2*pi*f*np.cos(2*pi*f*t)
        qddj[self.control_indices[1][1]] = 0. - 0.03*2*pi*f*2*pi*f*np.sin(2*pi*f*t)
        # left abduction
        qla,qdla,qddla = self.abduction_trajectory_t(t,th0_abd,thf_abd,f,th_offset=pi)
        qj[self.control_indices[2][0]] = -qla
        qdj[self.control_indices[2][0]] = -qdla
        qddj[self.control_indices[2][0]] = -qddla
        # left flexion
        qj[self.control_indices[2][1]] = -self.neutral_contact_flexion_angle + 0.03*np.sin(2*pi*f*t + pi)
        qdj[self.control_indices[2][1]] = 0. + 0.03*2*pi*f*np.cos(2*pi*f*t + pi)
        qddj[self.control_indices[2][1]] = 0. - 0.03*2*pi*f*2*pi*f*np.sin(2*pi*f*t + pi)
        # If set to True, prepend the current state of the base, stored in self.state
        if include_base:
            qj = np.concatenate((self.state.qb, qj))
            qdj = np.concatenate((self.state.qbd, qdj))
            qddj = np.concatenate((self.state.qbdd, qddj))
        return qj, qdj, qddj
    
    def generate_trajectory_time_array(self,duration,steps, f, A_lat,th0_lat, th0_abd, thf_abd,include_base=True):
        """Generate an array containing time-series of the desired pos,vel,acc
        for all the joints(/DOFs, if the base is included) of the model

        Parameters
        ----------
        duration : [float]
            time span defining for how long the trajectory should be generated\n
        steps : [int]
            number of steps into which subdivide "duration"\n
        f : parameter for self.compose_joints_trajectories_t()\n
        A_lat : parameter for self.compose_joints_trajectories_t()\n
        th0_lat : parameter for self.compose_joints_trajectories_t()\n
        th0_abd : parameter for self.compose_joints_trajectories_t()\n
        thf_abd : parameter for self.compose_joints_trajectories_t()\n
        include_base : parameter for self.compose_joints_trajectories_t()

        Returns
        -------
        [tuple]
            (q_time_array, qd_time_array, qdd_time_array) np.arrays containing time-series for the
            desired pos,vel,acc of models joints(/DOFs, if the base is included).
            q_time_array[i,:] will contain the desired pos,vel,acc for the i-th step.
        """
        if (steps != int(duration/self.dt_simulation)):
            print("ERROR: Step count is wrong, doesn't match self.dt_simulation")
            return
        if include_base:
            q_time_array = np.zeros((steps,self.state.nq))
            qd_time_array = np.zeros((steps,self.state.nqd))
            qdd_time_array = np.zeros((steps,self.state.nqd))
        else:
            q_time_array = np.zeros((steps,self.state.nqj))
            qd_time_array = np.zeros((steps,self.state.nqjd))
            qdd_time_array = np.zeros((steps,self.state.nqjd))
        for i in range(steps):
            q_time_array[i], qd_time_array[i], qdd_time_array[i] = self.compose_joints_trajectories_t(
                t=i*self.dt_simulation,
                f=f,
                A_lat=A_lat,
                th0_lat=th0_lat,
                th0_abd=th0_abd,
                thf_abd=thf_abd,
                include_base=include_base)
        return q_time_array, qd_time_array, qdd_time_array
    
    def combined_cosine_fun(self,A,f1,n,t_off=0,delta=0, bias=0):
        """[TO BE DOCUMENTED IN THE REPORT]
        Generate a periodic signal combining two cosine function. It's suggested to plot the signal to see
        how different parameters affect the signal's shape

        Parameters
        ----------
        A : [float]
            amplitude\n
        f1 : [float]
            main frequency\n
        n : [int]
            ratio f2/f1\n
        t_off : float, optional
            time offset of the whole signal, by default 0\n
        delta : float, optional
            angle offset of second cosine, by default 0\n
        bias : float, optional
            bias, by default 0

        Returns
        -------
        [function]
            return a lambda function such that calling fun(t) generate the signal value for time t
        """
        # f1 and f2 must generate a periodic signal with periodicity at least that of the 
        # walking cycle (f_walk = 1/(2*t_stance))
        # to simplify this, f1 is considered as equal to f_walk and f2 = n * f1, with n an integer number
        # Furthermore, to produce symmetric gait, n should be EVEN
        # t_off represents a global translation in time
        # delta represent an offset between the two wave, expressed in radians
        # For sensible value, taken from EMG, see report
        fun = lambda t: A*cos(2*pi*f1*(t+t_off))*cos(2*pi*n*f1*(t+t_off) + delta) + bias
        return fun
    
    def get_torques_profile_fun(self, A, f1, n, t_off, delta, bias):
        """[TO BE DOCUMENTED IN THE REPORT]
        Generate torques for all ACTUATED joints, based on self.combined_cosine_fun()

        Parameters
        ----------
        A : [tuple(tuple,tuple,tuple)]
            Check parameter of self.combined_cosine_fun()\n
        f1 : [tuple(tuple,tuple,tuple)]
            Check parameter of self.combined_cosine_fun()\n
        n : [tuple(tuple,tuple,tuple)]
            Check parameter of self.combined_cosine_fun()\n
        t_off : [tuple(tuple,tuple,tuple)]
            Check parameter of self.combined_cosine_fun()\n
        delta : [tuple(tuple,tuple,tuple)]
            Check parameter of self.combined_cosine_fun()\n
        bias : [tuple(tuple,tuple,tuple)]
            Check parameter of self.combined_cosine_fun()\n

        Returns
        -------
        [tuple(tuple,tuple,tuple)]
            nested tuple (see Warnings) containing lambda function that generates signal based on the parameters
            given as input. Example, calling output_to_this_fun[0][1](t) return the value, at time t, associated to 
            the second lateral joint of the spine.

        Warnings
        -------
        Everything (both the input and the output of this function) should be packed as a nested tuple 
        like self.control_indices. -->  ((spine_lateral),(r_abd,r_flex),(l_abd, l_flex))
        """
        # Variables should be passed as nested tuple,
            # packed like control_indices ((spine_lateral),(r_abd,r_flex),(l_abd, l_flex))
        # Returns a nested tuple (packed like control_indices) of lambda functions, all dependent only on t
        n_lat = len(self.control_indices[0])
        if not (n_lat==len(A[0])):
            print("Error: wrong dimensions of the input")
            return
        #lateral joints
        tau_lat_fun = [0]*n_lat
        for i,(A_i, f1_i, n_i, t_off_i, delta_i, bias_i) in enumerate(zip(A[0], f1[0], n[0], t_off[0], delta[0], bias[0])):
            tau_lat_fun[i] = self.combined_cosine_fun(A_i,f1_i,n_i,t_off_i,delta_i,bias_i)
        #right girdle
        tau_right_fun = [0]*2
        for i,(A_i, f1_i, n_i, t_off_i, delta_i, bias_i) in enumerate(zip(A[1], f1[1], n[1], t_off[1], delta[1], bias[1])):
            tau_right_fun[i] = self.combined_cosine_fun(A_i,f1_i,n_i,t_off_i,delta_i,bias_i)
        #left girdle
        tau_left_fun = [0]*2
        for i,(A_i, f1_i, n_i, t_off_i, delta_i, bias_i) in enumerate(zip(A[2], f1[2], n[2], t_off[2], delta[2], bias[2])):
            tau_left_fun[i] = self.combined_cosine_fun(A_i,f1_i,n_i,t_off_i,delta_i,bias_i)
        # convert tuple, to avoid further modification of the output outside this function 
        tau_lat_fun = tuple(tau_lat_fun)
        tau_right_fun = tuple(tau_right_fun)
        tau_left_fun = tuple(tau_left_fun)
        return (tau_lat_fun, tau_right_fun, tau_left_fun)
    
    def generate_torques_time_array(self,tau_array_fun,duration,t0=0., include_base=True):
        """Generate an array containing the time-series of the desired torque value
        for each joint(/each DOF, if base is included)

        Parameters
        ----------
        tau_array_fun : [nested tuple]
            , output of self.get_torques_profile_fun()
        duration : [float]
            time span defining for how long the trajectory should be generated\n
        t0 : float, optional
            start time of the time-series, by default 0.
        include_base : bool, optional
            prepend values for the base DOF, by default True

        Returns
        -------
        [np.array]
            time-series of the torque values for each joint. output_to_this_fun[i,:] return the array containing 
            the values for each joint at the i-th step
        """
        # input should be the output of torques_profile_fun, packed lambda function depending only on t
        # the output contains, as rows,the value of the torque to apply at each joint (even the passive ones, equal to 0)
        steps = int(duration/self.dt_simulation)
        t_array = np.linspace(t0,t0+duration,steps)
        tau_array = np.zeros((steps,self.num_joints))
        for i,t in enumerate(t_array):
            for joint_index,tau_fun_i in zip(self.control_indices[0], tau_array_fun[0]):
                tau_array[i][joint_index] = tau_fun_i(t)
            for joint_index,tau_fun_i in zip(self.control_indices[1], tau_array_fun[1]):
                tau_array[i][joint_index] = tau_fun_i(t)
            for joint_index,tau_fun_i in zip(self.control_indices[2], tau_array_fun[2]):
                tau_array[i][joint_index] = tau_fun_i(t)
        if include_base:
            tau_array = np.hstack((np.zeros((steps,self.state.nqbd)),tau_array))
        return tau_array
    
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
