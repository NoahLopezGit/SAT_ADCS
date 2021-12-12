import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
import pandas as pd
import Attitude_Kinematics as att
import Controller_Logic as ct

"""
Satellite EOM model - Noah Lopez
"""


#Global Constants
kp = 0.0
kd = 0.0
controller=True

#output storage
time_vector    = [] 
output_vector   = []

#Satellite Integrator Model
def exnxsofmotion(x_vec, time, J,command_quaternion):
    '''
    Need in for for scipy odeint.
    vector of explicit differential exn (RHS)
    vector of corresponding states (not differential)
    can only go first order so higher order diffeq must be described as system of first order
    '''
    
    w1,w2,w3,q1,q2,q3,q4 = x_vec
    #constructing RHS vector TODO add L functionality here... CLEANUP??! 
    RHS = [
        1/J[0]*( get_torque(command_quaternion,[q1,q2,q3,q4],[w1,w2,w3])[0,0] - ( w2*w3*J[2] - w3*w2*J[1])),
        1/J[1]*( get_torque(command_quaternion,[q1,q2,q3,q4],[w1,w2,w3])[1,0] - ( w3*w1*J[0] - w1*w3*J[2])),
        1/J[2]*( get_torque(command_quaternion,[q1,q2,q3,q4],[w1,w2,w3])[2,0] - ( w1*w2*J[1] - w2*w1*J[0])),
        0.5*( q4*w1 - q3*w2 + q2*w3),
        0.5*( q3*w1 + q4*w2 - q1*w3),
        0.5*(-q2*w1 - q1*w2 + q4*w3),
        0.5*(-q1*w1 - q2*w2 - q3*w3)
    ]

    #for storing values which are not tracked in solution (i.e. torques, ref input)
    time_vector.append(time)
    output_vector.append([
        get_torque(command_quaternion,[q1,q2,q3,q4],[w1,w2,w3])[0,0],
        get_torque(command_quaternion,[q1,q2,q3,q4],[w1,w2,w3])[1,0],
        get_torque(command_quaternion,[q1,q2,q3,q4],[w1,w2,w3])[2,0],
        x_vec[3],
        x_vec[4],
        x_vec[5],
        x_vec[6]
    ])
    return RHS


def get_torque(q_command,q_actual,angular_velocity):
    #converting lists to matrix form
    q_command = np.matrix([
        [q_command[0]],
        [q_command[1]],
        [q_command[2]],
        [q_command[3]]
    ])
    q_actual = np.matrix([
        [q_actual[0]],
        [q_actual[1]],
        [q_actual[2]],
        [q_actual[3]]
    ])
    angular_velocity = np.matrix([
        [angular_velocity[0]],
        [angular_velocity[1]],
        [angular_velocity[2]]
    ])

    #trying normalizing vectors before calculating state
    #q_actual = q_actual/np.linalg.norm(q_actual)

    #need to get sum of disturbance torques and torque from reaction control wheels
    delta_q = ct.get_delta_q(q_command,q_actual)
    controller_torque = ct.get_torque( kp,delta_q, kd, angular_velocity)
    total_torque = controller_torque # + total_disturbance_torque
    if controller==False:
        total_torque=np.matrix([[0.0],[0.0],[0.0]])
    return total_torque #this will return as 3 dim column vector (in matrix)
    

def solver(exn,x0_vec,t_vec,J,command_quaternion):
    sol = sp.odeint(exn, x0_vec,t_vec,args=(J,command_quaternion))
    return sol


def plot_results(t_vec, sol, t2_vec, outputs):
    sol_name = ["W1","W2",'W3',"Q1","Q2","Q3","Q4"]
    output_name = ["T1","T2","T3","Q1","Q2","Q3","Q4"]
    #plot results
    fig_dict = {}
    for i in range(np.shape(sol)[1]):
        if i % 4==0:
            n = 1+i//4
            fig = "sol_fig" + str(n)
            fig_dict[fig] = plt.figure(n)
        solution_vector = sol[:,i]
        fig_dict[fig].add_subplot(4,1,1+i%4)
        plt.plot(
            t_vec, solution_vector
        )
        plt.ylabel(sol_name[i])

    for i in range(np.shape(outputs)[1]): #hopefully this works
        if i % 4==0:
            n = 2+(i+len(sol[0,:]))//4
            fig = "out_fig" + str(n)
            fig_dict[fig] = plt.figure(n)
        output_vector = outputs[:,i]
        fig_dict[fig].add_subplot(4,1,1+i%4)
        plt.plot(
            t2_vec, output_vector
        )
        plt.ylabel(output_name[i])
    
    plt.show()


def save_results(t_vec, sol, t2_vec, outputs):
    #store results
    sol_name = ["W1","W2",'W3',"Q1","Q2","Q3","Q4"]
    output_name = ["T1","T2","T3","Q1","Q2","Q3","Q4"]
    data1 = {}
    data2 = {}

    #adding solution values
    data1["time (s)"] = t_vec
    for i in range(len(sol[0,:])):
        solution_vector = sol[:,i]
        data1["state={0}".format(sol_name[i])] = solution_vector
    #adding output values
    data2["time2 (s)"] = t2_vec
    for i in range(np.shape(outputs)[1]):
        output_vector = outputs[:,i]
        data2["state={0}".format(output_name[i])] = output_vector
    
    #Saving output and solution values 
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    writer = pd.ExcelWriter('State_Integration.xlsx', engine='xlsxwriter')
    df1.to_excel(writer, sheet_name="Solution States", index=False)
    df2.to_excel(writer, sheet_name="Output States", index=False)
    writer.save()


if __name__=="__main__":
    t0 = 0
    tf = 10
    n = 50
    t_vec = np.linspace(t0,tf,n)
    solution = solver(  exnxsofmotion, 
                        [ -0.5,10.0,0.5, 0.0,0.0,0.0,1.0 ],    #initial states
                        t_vec,                  #t_vec to integrate over
                        [100,100,100],          #J (Principle axis MOI) vector
                        [0.0,0.2705981,0.2705981,0.9238795])      #command quaternion
    att.Animate_Attitude_Set(np.array(solution[:,3:7]).transpose(),10/100)
    
    output_vector = np.asarray(output_vector)
    plot_results(t_vec, solution, time_vector, output_vector)
    save_results(t_vec, solution, time_vector, output_vector)
