import pybullet as p
import numpy as np
import scipy as sp
import scipy.linalg as lna
import time
import pybullet_data
from math import *

class Discrete_Low_Pass:
    def __init__(self, dt, tc, K=1):
        self.x = 0
        self.dt = dt
        self.tc = tc
        self.K = K
    def reset(self):
        self.x = 0
    def filter(self, signal):
        self.x = (1-self.dt/self.tc)*self.x + self.K*self.dt/self.tc * signal
        return self.x

class Crawler:

    def __init__(self, spine_segments, urdf_path="/home/fra/Uni/Tesi/crawler", base_position=[0,0,0.2], base_orientation=[0,0,0,1]):
        self.scale=1
        #NOTE: Properties in this block of code must be manually matched to those defined in the Xacro file
        self.spine_segments         = spine_segments
        self.body_length            = self.scale * 1
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
        self.mask_base = [0,1,2,3,4,5]
        self.mask_joints = list(range(6,6+self.num_joints))
        self.mask_act = list(np.arange(6, 6+self.num_joints-4,2))
        self.mask_nact = list(np.arange(7, 6+self.num_joints-4,2))+list(range(6+self.num_joints-4,6+self.num_joints))
        self.mask_right_girdle = [6+self.num_joints-4, 6+self.num_joints-3]
        self.mask_left_girdle = [6+self.num_joints-2, 6+self.num_joints-1]
        #
        self.low_pass = Discrete_Low_Pass(dt=1/240, tc=10, K=1)
        

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
        lat_joints_i = tuple(range(1,(self.num_joints-4),2))
        r_girdle_flex_i = self.num_joints-4
        r_girdle_abd_i = self.num_joints-3
        l_girdle_flex_i = self.num_joints-2
        l_girdle_abd_i = self.num_joints-1
        return (lat_joints_i,r_girdle_flex_i,r_girdle_abd_i,l_girdle_flex_i,l_girdle_abd_i)

    def fix_right_foot(self, leg_index=17):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["right_foot"]:
            constId=self.constraints["right_foot"]
            print("Error: remove right foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, leg_index, -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[self.leg_length, 0, 0], childFramePosition=list(p.getLinkState(self.Id,leg_index)[0]))
            self.constraints["right_foot"]=constId
        return constId

    def fix_left_foot(self, leg_index=19):
        #constraint is generated at the origin of the center of mass of the leg, i.e. at center of the spherical "foot"
        if self.constraints["left_foot"]:
            constId=self.constraints["left_foot"]
            print("Error: remove left foot constraint before setting a new one")
        else:
            constId = p.createConstraint(self.Id, leg_index, -1,-1, p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[self.leg_length, 0, 0], childFramePosition=list(p.getLinkState(self.Id,leg_index)[0]))
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
    
    def test_link_COM_jacobians(self,link_index):
        #Jacobian of the COM is computed as the weighted mean (with respect to masses) of the Jacobian of the links
        #The transational jacobian of the base is just the identity matrix multiplying the base translational velocity
            # plus null terms associated to the base angular speed and the joints speed terms (thus 3+self.num_joints)
            # NOTE: angular speed values are null just because the COM of the girdle is at the origin of the link frame
        self.set_links_state_array()
        joints_pos = self.get_joints_pos_tuple()
        R = self.get_R_base_to_world()
        ###
        Ji = np.asarray(
            p.calculateJacobian(self.Id,
                link_index,
                self.links_state_array[link_index]["loc_com_trn"],
                joints_pos,
                #[0.0]*(self.num_joints),
                self.get_joints_speeds_tuple(),
                [0.0]*(self.num_joints))
            )
        Jt_i = Ji[0]
        Jtbrw_i = (R.dot(Jt_i[:,[0,1,2]])).dot(R.T)
        Jtqw_i = R.dot(Jt_i[:,self.mask_joints])
        Jtw_i = np.concatenate((Jtbrw_i, Jt_i[:,[3,4,5]], Jtqw_i),1)
        # Jtqw_i = R.dot(Jt_i[:,self.mask_joints])
        # Jtw_i = np.concatenate((Jt_i[:,self.mask_base], Jtqw_i),1)
        ###
        Jr_i = Ji[1]
        Jrqw_i = R.dot(Jr_i[:,self.mask_joints])
        Jrw_i = np.concatenate((Jr_i[:,self.mask_base], Jrqw_i),1)
        ###
        qd = np.asarray(p.getBaseVelocity(self.Id)[1] + p.getBaseVelocity(self.Id)[0] + self.get_joints_speeds_tuple())
        qd = np.reshape(qd,(qd.shape[0],1))
        ###
        vt_J = tuple(np.ndarray.flatten(Jtw_i.dot(qd)))
        vt_true = self.links_state_array[link_index]["world_com_vt"]
        print("True_t: ",vt_true, "\nJaco_t: ",vt_J)
        et = round(lna.norm(np.asarray(vt_J)-np.asarray(vt_true))/lna.norm(np.asarray(vt_true)),4)
        print("et_rel = ",et)
        ###
        vr_J = tuple(np.ndarray.flatten(Jrw_i.dot(qd)))
        vr_true = self.links_state_array[link_index]["world_com_vr"]
        print("True_r: ",vr_true, "\nJaco_r: ",vr_J)
        er = round(lna.norm(np.asarray(vr_J)-np.asarray(vr_true))/lna.norm(np.asarray(vt_true)),4)
        print("er_rel = ",er)
        ###
        print("Jt: ",np.round(Jtw_i,3))
        print("Jr: ",np.round(Jrw_i,3))
        
        #returned as NUMPY ARRAY
        return

    def COM_trn_jacobian(self):
        #Jacobian of the COM is computed as the weighted mean (with respect to masses) of the Jacobian of the links
        #The transational jacobian of the base is just the identity matrix multiplying the base translational velocity
            # plus null terms associated to the base angular speed and the joints speed terms (thus 3+self.num_joints)
            # NOTE: angular speed values are null just because the COM of the girdle is at the origin of the link frame
        self.set_links_state_array()
        Jbase_t = np.asarray([  [0.0]*3 + [1.0,0.0,0.0] + [0.0]*(self.num_joints),
                                [0.0]*3 + [0.0,1.0,0.0] + [0.0]*(self.num_joints),
                                [0.0]*3 + [0.0,0.0,1.0] + [0.0]*(self.num_joints) ])
        JM_t = Jbase_t*(p.getDynamicsInfo(self.Id,-1)[0])
        joints_pos = self.get_joints_pos_tuple()
        for i in range(0,self.num_joints):
            Ji_t = np.asarray(
                p.calculateJacobian(self.Id,
                    i,
                    self.links_state_array[i]["loc_com_trn"],
                    joints_pos,
                    #[0.0]*(self.num_joints),
                    self.get_joints_speeds_tuple(),
                    [0.0]*(self.num_joints))
                )[0]
            JM_t += Ji_t * (p.getDynamicsInfo(self.Id,i)[0])
        JM_t = JM_t/self.mass
        #returned as NUMPY ARRAY
        return JM_t


    def solve_null_COM_y_speed(self, K=1, k0=0.001):
        #Return the desired joint speeds of the spinal lateral joints to be used for velocity control
        #NOTE: since the dorsal joints of the spine and the DOFs of the base are not actuated, xd_desired 
            # is corrected (see report). Joints of the girdles are also not considered as actuated since their 
            # speed is set independently
        #Use constrained (convex) optimization to solve inverse kinematic, coupled with closed loop inverse kinematic
            #to make the error converge esponentially
        #q0d are projected inside the nullspace of the Jacobian (J) and can be chosen to minimize a cost function
            # Chosen cost function is H(q)=sum(q_i ^2), to keep joints as near to 0 as possible while respecting the requirements
            # given by inverse kinematics.
            # H(q) minimum is found through gradient descent, by chosing q0d=-k0*gradient(H(q)) (easy for quadratic H(q))
        #NOTE: refer to the notation in the report
        mask_base = [0,1,2,3,4,5]
        mask_act = list(np.arange(6, 6+self.num_joints-4,2))
        mask_nact = list(np.arange(7, 6+self.num_joints-4,2))+list(range(6+self.num_joints-4,6+self.num_joints))
        ###
        bd = np.array(p.getBaseVelocity(self.Id)[1] + p.getBaseVelocity(self.Id)[0]) #(angular velocity, linear velocity)
        bd = np.reshape(bd,(bd.shape[0],1))
        #qd = np.asarray(p.getBaseVelocity(self.Id)[1] + p.getBaseVelocity(self.Id)[0] + self.get_joints_speeds_tuple())
        qd = np.array((0,0,0,0,0,0)+self.get_joints_pos_tuple())
        qd = np.reshape(qd,(qd.shape[0],1))
        qda = qd[mask_act]
        qdn = qd[mask_nact]
        #print("qd check (qd, qda, qdn):\n", qd,"\n", qda, "\n", qdn)
        ###
        W = np.eye(qda.shape[0]) #weight matrix for the different joints ("how much should they try to minimize the cost of H(q)")
        Winv = lna.inv(W)
        Jy = self.COM_trn_jacobian()[1]
        Jy = np.reshape(Jy,(1,Jy.shape[0]))
        Jyb = Jy[:,mask_base]
        Jya = Jy[:,mask_act]
        Jyn = Jy[:,mask_nact]
        Jwr = Winv.dot(Jya.T).dot(lna.inv(Jya.dot(Winv.dot(Jya.T))))
        print("Jy.Winv.JyaT",Jya.dot(Winv.dot(Jya.T)))
        #print("SHAPE JWR = ", Jwr.shape)
        P = np.eye(qda.shape[0])-Jwr.dot(Jya)
        #print("SHAPE P = ", P.shape)
        ###
        q0da = -2*k0*qda
        #print("Shape q0da: ", q0da.shape, " should be (8,1)")
        ###
        xd_desired = 0 - Jyb.dot(bd) - Jyn.dot(qdn)
        #print("xd_des: ", xd_desired)
        COM_vy = Jy.dot(qd)
        #COM_vy = self.COM_velocity_world()[1]
        e = xd_desired-COM_vy #since we want the COM to have null speed along y axis the error is simply 0-COMv[y]
        print("e_est: ",(COM_vy-(self.COM_velocity_world()[1])))
        print("P.q0da: ",list(np.ndarray.flatten(P.dot(q0da))))
        print("Jwr.(Ke): ",list(np.ndarray.flatten(Jwr.dot(K*e))))
        print("Jwr.(xd_d): ",list(np.ndarray.flatten(Jwr.dot(xd_desired))))
        #print("e: ", e)
        ###
        #print("SHAPE P.q0d = ", (P.dot(q0da)).shape)
        #print("SHAPE Jwr.dot(xd_desired + K*e) = ", (Jwr.dot(xd_desired + K*e)).shape)
        qd = Jwr.dot(xd_desired + K*e) + P.dot(q0da)
        return qd

    def velocity_control_lateral(self, K=1, fmax=10):
        qd = np.ndarray.flatten(self.solve_null_COM_y_speed(K=K))
        qdf = list(map(self.low_pass.filter,qd))
        #print("YOH qd[3]: ",qd[3])
        #print("YOH qdf[3]: ",qdf[3])
        for index, joint_i in enumerate(self.control_indices[0]):
            p.setJointMotorControl2(self.Id, joint_i, p.VELOCITY_CONTROL, targetVelocity=qdf[index],force=fmax)
        return


#NOTE: feet's constraints are set with their position using the values provided by the set_links_state_array, might need to be modified if the leg in the URDF change
#NOTE: add function for mapping link and joint indices to their name?
#NOTE: set_joints_state_array can be taken out of each function and called just once after stepping the simulation, for now is called inside each function to avoid errors

# def crawler_control(crawlerID, control_i):
#     #the index of the joints to be controlled are considered to be the tuple generated through get_crawler_control_index
#     #control_i = ((indices of lateral spine joint), right girdle flexion, right girdle abduction, left girdle flexion, left girdle abduction)
#     for i in control_i[0]:
#         p.setJointMotorControl2(crawlerID,i,p.POSITION_CONTROL, targetPosition=0,force=20)
#     p.setJointMotorControl2(crawlerID,control_i[1],p.POSITION_CONTROL,targetPosition=0,force=20)
#     p.setJointMotorControl2(crawlerID,control_i[2],p.POSITION_CONTROL,targetPosition=0,force=20)
#     p.setJointMotorControl2(crawlerID,control_i[3],p.POSITION_CONTROL,targetPosition=0,force=20)
#     p.setJointMotorControl2(crawlerID,control_i[4],p.POSITION_CONTROL,targetPosition=0,force=20)
#     #might be nice to add computed torque control with an adaptive part for estimating the torques required to counteract friction, but must be formulated properly
#     return