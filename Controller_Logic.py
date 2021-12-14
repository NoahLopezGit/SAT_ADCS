#3-axis controller
import Attitude_Kinematics as att
import numpy as np



def get_torque(proportional_coefficients, delta_q, derivative_coefficients, angular_velocity): #body frame?
    '''
    delta_q should be length 4 column vector (np.matrix style)
    angular_velocity should be length 3 columne vector (np.matrix style)
    '''
    proportional_coefficients = np.diag(proportional_coefficients)
    derivative_coefficients = np.diag(derivative_coefficients)

    torque = -np.matmul( proportional_coefficients, delta_q[0:3] ) \
             -np.matmul( derivative_coefficients, angular_velocity)

    return torque


def get_delta_q(command_quaternion, current_quaternion):
    q1_3    = np.matmul( att.Q2PHI(command_quaternion).transpose(), current_quaternion )
    q4      = np.matmul( current_quaternion.transpose(), command_quaternion )
    delta_q = np.vstack((q1_3,q4))
    return delta_q


if __name__=="__main__":
    #identity quaternion
    q_command   = np.matrix([
        [0.0],
        [0.0],
        [0.0],
        [1.0]
    ]).astype(float)
    #quaternion corresponding to 45 deg rotation about z
    q_actual    = np.matrix([
        [0.0],
        [0.0],
        [0.3826834],
        [0.9238795]
    ]).astype(float)

    print(get_torque( 1.0,get_delta_q(q_command, q_actual) , 1.0, np.matrix([[0.0],[0.0],[0.0]])))