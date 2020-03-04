import pybullet as p
import numpy as np
import time
import pybullet_data
from math import *

class Discretel_Low_Pass:
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
        lat_joints_i = tuple(range(1,(p.getNumJoints(self.Id)-4),2))
        r_girdle_flex_i = p.getNumJoints(self.Id)-4
        r_girdle_abd_i = p.getNumJoints(self.Id)-3
        l_girdle_flex_i = p.getNumJoints(self.Id)-2
        l_girdle_abd_i = p.getNumJoints(self.Id)-1
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

    # def prova_jacobian(self, link_index, base_eu_d):
    #     #base_eu_d is a tuple containing the derivatives of the euler's angles describing the base orientation
    #     self.set_links_state_array()
    #     COM_vel = np.asarray(p.getLinkState(self.Id, link_index, computeLinkVelocity=1)[6])
    #     joints_pos = self.get_joints_pos_tuple()
    #     Jt, Jr = np.asarray(
    #         p.calculateJacobian(self.Id,
    #             link_index,
    #             self.links_state_array[link_index]["loc_com_trn"],
    #             joints_pos,
    #             #[0.0]*(self.num_joints),
    #             self.get_joints_speeds_tuple(),
    #             [0.0]*(self.num_joints))
    #         )
    #     print("\nJt", Jt)
    #     print("\nJr", Jr)
    #     qdot = np.asarray(p.getBaseVelocity(self.Id)[1] + p.getBaseVelocity(self.Id)[0] + self.get_joints_speeds_tuple())
    #     jacobian_vel= np.dot(Jt,qdot)
    #     print("LINK %d" %link_index)
    #     print("True COM speed: ",COM_vel,"\nComputed w/ Jacobian: ",jacobian_vel)
    #     return
    
    def get_base_Eulers(self):
        return  p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.Id)[1])

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
                    self.get_joints_speeds_tuple(),
                    [0.0]*(self.num_joints))
                )[0]
            JM_t += Ji_t * (p.getDynamicsInfo(self.Id,i)[0])
        JM_t = JM_t/self.mass
        #returned as NUMPY ARRAY
        return JM_t

    def solve_null_COM_y_speed(self):
        #Return the desired joint speeds of the spinal lateral joints to be used for velocity control
        return


#NOTE: constraints are set without using the values provided by the set_joints_state_array, might need to be modified if the leg in the URDF change
#NOTE: add function for mapping link and joint indices to their name
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