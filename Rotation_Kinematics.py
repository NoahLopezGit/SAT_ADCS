#Rotation Kinematics
import numpy as np
import attitudematrices as att


def Angularvelocity2Q_dot(angular_velocity, quaternion):
    """
    I think this should be passive... what exactly does this mean?
    passive would make more sense... you want to describe the motion of the frame for attitude dynamics here.
    """
    q_dot = np.matmul( att.Q2PHI(quaternion), angular_velocity) 
    q_dot = np.multiply( 0.5, q_dot )
    return np.array([ q_dot[0,0], q_dot[0,1], q_dot[0,2], q_dot[0,3] ])


def Update_Q(q_old, angular_velocity, h):
    q_dot = Angularvelocity2Q_dot( angular_velocity,  q_old)
    q_new = np.array([  Euler_Step( q_old[0], q_dot[0], h),
                        Euler_Step( q_old[1], q_dot[1], h),
                        Euler_Step( q_old[2], q_dot[2], h),
                        Euler_Step( q_old[3], q_dot[3], h) ])
    return q_new


def Euler_Step(Yn, Ynprime, h): #input  Yn state and Yn' value, step size
    Ynp1 = Yn + h*Ynprime
    #return updated step
    return Ynp1 
    
#do i need this
def Update_Vec(vec, quat):
    v_tmp = att.Q_Tran_Vec( quat, vec )
    v_tmp = np.multiply( 1/np.linalg.norm(v_tmp), v_tmp) #normalizing vector bc this blows up TODO: find why this is unstable
    return v_tmp