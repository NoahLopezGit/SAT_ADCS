#library containing all attitude dynamics modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import math


'''
-----------------------------------------------------Kinematics----------------------------------------------------
'''
#Calculates 3-2-1 Euler Angles from DCM; TODO: support other euler angle rotations
def Quat2Euler(quaternion): 
    DCM = Quat2DCM(quaternion)
    phi = np.arctan2( DCM[1,2], DCM[2,2] )
    theta = np.sin( DCM[0,2] )
    psi = np.arctan2( DCM[0,1], DCM[0,0] )
    return phi, theta, psi


#Calculates DCM based on quaternion (base is passive version of DCM) 
def Quat2DCM(quaternion):
    DCM = np.matmul( np.transpose(Q2PHI(quaternion)), Q2PSI(quaternion) )
    return DCM


#Transforms vector based on frame rotation (specify active=True for active transform of vector)
def Q_Tran_Vec(quaternion, vector, active=False):
    DCM = np.matmul( np.transpose(Q2PHI(quaternion)), Q2PSI(quaternion) )

    if active==True:
        v_prime = np.matmul( np.transpose(DCM), vector ) 
    elif active==False:
        v_prime = np.matmul( DCM, vector ) #I THINK TRANSPOSE IS ACTIVE.. NO TRANSPOSE DCM IS PASSIVE

    return np.array([ v_prime[0,0], v_prime[0,1], v_prime[0,2] ])


#Calculates PHI matrix based on quat; for quat transformation
def Q2PHI(quaternion):
    q = quaternion
    PHI = np.matrix([
        [  q[3], -q[2],  q[1]  ],
        [  q[2],  q[3], -q[0]  ],
        [ -q[1],  q[0],  q[3]  ],
        [ -q[0], -q[1], -q[2]  ],
    ])
    return PHI


#Calculates PSI matrix based on quat; for quat transformation
def Q2PSI(quaternion):
    q = quaternion
    PSI = np.matrix([
        [  q[3],  q[2], -q[1]  ],
        [ -q[2],  q[3],  q[0]  ],
        [  q[1], -q[0],  q[3]  ],
        [ -q[0], -q[1], -q[2]  ],
    ])
    return PSI


def Q_Tran_Q(q_tran, q_old):
    a = Q2PSI(q_tran)
    transform_matrix = np.matrix([ #if there is a better way to do this that would be great
        [a[0,0], a[0,1], a[0,2], q_tran[0]],
        [a[1,0], a[1,1], a[1,2], q_tran[1]],
        [a[2,0], a[2,2], a[2,2], q_tran[2]],
        [a[3,0], a[3,1], a[3,2], q_tran[3]]
    ])
    q_new = np.matmul(transform_matrix,q_old)
    return  np.array([q_new[0,0],q_new[0,1],q_new[0,2],q_new[0,3]])#q_new


#Finds trasnformation matrix from ECI -> ECEF frame
def ECI2ECEF(datetime_object):
    #reminder; this is a passive transformation from a vector in the ECI frame to its representation in the ECEF frame
    #transpose of A_ei will give the transformation of ECEF to ECI. 

    theta_GMST = Get_Julian_Datetime(datetime_object)
    a = np.cos(theta_GMST)
    b = np.sin(theta_GMST)
    A_ei = np.matrix([
        [ a, b, 0 ],
        [-b, a, 0 ],
        [ 0, 0, 1 ]
    ])
    return A_ei


#Gets julian date time with datetime object; for ECI <--> ECEF transformation
def Get_Julian_Datetime(date):
    """
    Convert a datetime object into julian float.
    Args:
        date: datetime-object of date in question

    Returns: float - Julian calculated datetime.
    Raises: 
        TypeError : Incorrect parameter type
        ValueError: Date out of range of equation
    """

    # Ensure correct format
    if not isinstance(date, datetime.datetime):
        raise TypeError('Invalid type for parameter "date" - expecting datetime')
    elif date.year < 1801 or date.year > 2099:
        raise ValueError('Datetime must be between year 1801 and 2099')

    # Perform the calculation
    julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int(
        (275 * date.month) / 9.0) + date.day + 1721013.5 + (
                          date.hour + date.minute / 60.0 + date.second / math.pow(60,
                                                                                  2)) / 24.0 - 0.5 * math.copysign(
        1, 100 * date.year + date.month - 190002.5) + 0.5

    return julian_datetime


#Calculates quaternion rate from angula velocity
def Angularvelocity2Q_dot(angular_velocity, quaternion):
    """
    I think this should be passive... what exactly does this mean?
    passive would make more sense... you want to describe the motion of the frame for attitude dynamics here.
    """
    q_dot = np.matmul( Q2PHI(quaternion), angular_velocity) 
    q_dot = np.multiply( 0.5, q_dot )
    return np.array([ q_dot[0,0], q_dot[0,1], q_dot[0,2], q_dot[0,3] ])


#Updates quaternion based on measured angular velocity and euler stepping
def Update_Q(q_old, angular_velocity, h):
    q_dot = Angularvelocity2Q_dot( angular_velocity,  q_old)
    q_new = np.array([  Euler_Step( q_old[0], q_dot[0], h),
                        Euler_Step( q_old[1], q_dot[1], h),
                        Euler_Step( q_old[2], q_dot[2], h),
                        Euler_Step( q_old[3], q_dot[3], h) ])
    return q_new


#Propogation of state in time
def Euler_Step(Yn, Ynprime, h): #input  Yn state and Yn' value, step size
    Ynp1 = Yn + h*Ynprime
    #return updated step
    return Ynp1 
    

#Dont Really Need This; just transforms vector by quat and normalizes
def Update_Vec(vec, quat):
    v_tmp = Q_Tran_Vec( quat, vec )
    v_tmp = np.multiply( 1/np.linalg.norm(v_tmp), v_tmp) #normalizing vector bc this blows up TODO: find why this is unstable
    return v_tmp


'''
-----------------------------------------------------Plotting----------------------------------------------------
'''
#3D animation of attitude; TODO: fix so that input is only quaternion
def Animate_Attitude_Set(quaternion_array, h):
    
    if np.shape(quaternion_array)[0] != 4:
        print("Quaternion is not correct shape (needs to be 4 by X)")
        exit()
    timesteps = np.shape(quaternion_array)[1]
    

    #calculate vectors for each quaternion
    data_z = np.zeros((3, np.shape(quaternion_array)[1]))
    data_y = np.zeros((3, np.shape(quaternion_array)[1]))
    data_x = np.zeros((3, np.shape(quaternion_array)[1]))

    unit_z = np.array([0,0,1])
    unit_y = np.array([0,1,0])
    unit_x = np.array([1,0,0])
    for i in range(timesteps):\
        #unit vectors in body frame to their represenation in the inertial frame 
        data_z[:,i] = Q_Tran_Vec(np.transpose(quaternion_array[:,i]), unit_z) #inverse passive transform of unit vector in body gives vector in intertial frame
        data_y[:,i] = Q_Tran_Vec(np.transpose(quaternion_array[:,i]), unit_y)
        data_x[:,i] = Q_Tran_Vec(np.transpose(quaternion_array[:,i]), unit_x)

        print(np.dot(data_x[:,i],data_z[:,i])) #TODO: figure out why these values are not close enough to zero
        print(np.dot(data_y[:,i],data_z[:,i]))

    #Logitistics for plotting vectors
    def update(num, data_z, data_y, data_x, line1, line2, line3):
        b = np.array([[0, data_z[0,num]], [0, data_z[1,num]]]) 
        line1.set_data( b ) 
        a = np.array([0, data_z[2,num]])
        line1.set_3d_properties( a )

        b = np.array([[0, data_y[0,num]], [0, data_y[1,num]]]) 
        line2.set_data( b ) 
        a = np.array([0, data_y[2,num]])
        line2.set_3d_properties( a )

        b = np.array([[0, data_x[0,num]], [0, data_x[1,num]]]) 
        line3.set_data( b ) 
        a = np.array([0, data_x[2,num]])
        line3.set_3d_properties( a )

        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    line1, = ax.plot( [0, data_z[0,0]], [0, data_z[1,0]], [0, data_z[2,0]] )
    line2, = ax.plot( [0, data_y[0,0]], [0, data_y[1,0]], [0, data_y[2,0]] )
    line3, = ax.plot( [0, data_x[0,0]], [0, data_x[1,0]], [0, data_x[2,0]] ) 

    xline,  = ax.plot( [0, 1], [0, 0], [0, 0], 'k' )
    yline,  = ax.plot( [0, 0], [0, 1], [0, 0], 'k' )
    zline,  = ax.plot( [0, 0], [0, 0], [0, 1], 'k')
    # Setting the axes properties
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, update, timesteps, fargs=(data_z, data_y, data_x, line1, line2, line3), interval=h, blit=False)
    #ani.save('matplot003.gif', writer='imagemagick')
    plt.show()