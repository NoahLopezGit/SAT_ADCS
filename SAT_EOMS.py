from typing import final
import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
import pandas as pd
import Attitude_Kinematics as att
import Disturbance_Torques as dis
import Controller_Logic as ct
import json
import csv
import datetime

"""
Satellite EOM model - Noah Lopez
TODO
-separate use (examples) from import file (need to make it compatible)
-add momentum wheel physics and saturation tracking
-disturbance torques (verify they are correct)
-add momentum dumping model (thrusters)
-make plotting better
?-refactor to work in one list mode (np.matrix columns)
"""


#Global Constants - add to example??
#import from config.txt file
# Opening JSON file
system_config_file = input("Input Controller Configuration File\n")
with open(system_config_file, 'r') as openfile:
  
    #Reading from json file
    config_dict = json.load(openfile)
  
#Setting Constants From Config

kp_list = [
    config_dict["RXNWheels"]["Wheel_1"]["Kp"],
    config_dict["RXNWheels"]["Wheel_2"]["Kp"],
    config_dict["RXNWheels"]["Wheel_3"]["Kp"]
]
kd_list = [
    config_dict["RXNWheels"]["Wheel_1"]["Kd"],
    config_dict["RXNWheels"]["Wheel_2"]["Kd"],
    config_dict["RXNWheels"]["Wheel_3"]["Kd"]
]

controller  = config_dict["Control"]["Controller"]
track       = config_dict["Control"]["Track"]

def parse_sat_orientation_csv(name):
    csv_orientation_list= []
    with open(name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csv_reader:
            date_string =   str(row[0]) + ','       \
                            + str(row[1]) + ','     \
                            + str(row[2]) + ','     \
                            + str(row[3].split(',')[0])
            tmp_position    = [float(x) for x in row[3].split(',')[1:4]]
            tmp_velocity    = [float(x) for x in row[3].split(',')[4:7]]
            dt_object = datetime.datetime.strptime(date_string, "%d,%b,%Y,%H:%M:%S.%f")
            csv_orientation_list.append([dt_object, tmp_position, tmp_velocity])
    
    return csv_orientation_list

csv_orientation_list = parse_sat_orientation_csv("Configs/rv_vec_HST.csv")
initial_date_time = csv_orientation_list[0][0]
#output storage
time_vector     = [] 
output_vector   = []

#Satellite Integrator Model
'''
def get_sat_orientation(time):
    for orientation_time in orientation_time_list:
        if time > orientation_time:

    return sat_orientation
'''
            

def exnxsofmotion(x_vec, time, J,command_quaternion):
    '''
    Need in for for scipy odeint.
    vector of explicit differential exn (RHS)
    vector of corresponding states (not differential)
    can only go first order so higher order diffeq must be described as system of first order
    '''
    
    w1,w2,w3,q1,q2,q3,q4 = x_vec
    #constructing RHS vector TODO add L functionality here... CLEANUP??! 
    torque_tmp = get_torque(time,command_quaternion,[q1,q2,q3,q4],[w1,w2,w3]) #TODO make sure this is allowed
    RHS = [
        1/J[0]*( torque_tmp[0,0] - ( w2*w3*J[2] - w3*w2*J[1])),
        1/J[1]*( torque_tmp[1,0] - ( w3*w1*J[0] - w1*w3*J[2])),
        1/J[2]*( torque_tmp[2,0] - ( w1*w2*J[1] - w2*w1*J[0])),
        0.5*( q4*w1 - q3*w2 + q2*w3),
        0.5*( q3*w1 + q4*w2 - q1*w3),
        0.5*(-q2*w1 + q1*w2 + q4*w3),
        0.5*(-q1*w1 - q2*w2 - q3*w3)
    ]
    #for storing values which are not tracked in solution (i.e. torques, ref input)
    time_vector.append(time)
    output_vector.append([
        torque_tmp[0,0],
        torque_tmp[1,0],
        torque_tmp[2,0],
        2*np.arccos(np.matmul(np.matrix(command_quaternion), np.matrix([q1,q2,q3,q4]).transpose())[0,0]) #angular difference?
    ])
    return RHS


def get_sat_orientation(time,quaternion):
    date_object = initial_date_time + datetime.timedelta(0,time)
    for date_time_orientation in csv_orientation_list:
        if date_object >= date_time_orientation[0]:
            tmp_position    =  np.matrix(date_time_orientation[1]).transpose()
            tmp_velocity    =  np.matrix(date_time_orientation[2]).transpose()
            break
    euler_angles    = np.array(att.Quat2Euler(quaternion)) 

    sat_orientation = dis.Orientation(
        date_object,
        tmp_position,
        tmp_velocity,
        euler_angles
    )
    return sat_orientation


def get_torque(time,q_command,q_actual,angular_velocity):
    #converting lists to matrix form
    q_command = np.matrix(q_command).transpose()
    q_actual = np.matrix(q_actual).transpose()
    angular_velocity = np.matrix(angular_velocity).transpose()

    sat_orientation = get_sat_orientation(time,q_actual)
    #need to get sum of disturbance torques and torque from reaction control wheels
    if controller==False:
        controller_torque=np.matrix([[0.0],[0.0],[0.0]])
    else:
        delta_q = ct.get_delta_q(q_command,q_actual)
        controller_torque = ct.get_torque( kp_list,delta_q, kd_list, angular_velocity)

    total_torque = controller_torque + dis.get_total_torque(sat_orientation) #must be updated at each t_step
    return total_torque #this will return as 3 dim column vector (in matrix)


def solver(initial_conditions,t_vec,J,command_quaternion):
    #want to solve ODE for various steps of t0-tf and normalize between each step (ensures quaternion norm ~ 1)
    #split up by intevervals of t_vec
    time_chunks = []
    for i in range(len(t_vec)//10):
        if i == len(t_vec//10)-1:
            time_chunks.append(t_vec[10*i:])
        time_chunks.append(t_vec[10*i:10*(i+1)+1])

    final_solution = np.array(initial_conditions)
    if track==True:
        for time_chunk in time_chunks: #dont want inital time of 0
            date_object = initial_date_time + datetime.timedelta(0,time_chunk[0])
            for date_time_orientation in csv_orientation_list:
                if date_object >= date_time_orientation[0]:
                    continue
                else:
                    tmp_position    =  np.matrix(date_time_orientation[1]).transpose()
                    tmp_velocity    =  np.matrix(date_time_orientation[2]).transpose()
                    break

            tmp_command_quaternion = att.DCM2Quat(dis.get_Aio(tmp_position,tmp_velocity).transpose())#want inertial to orbital so transpose here
            command_quaternion = [
                tmp_command_quaternion[0,0],
                tmp_command_quaternion[1,0],
                tmp_command_quaternion[2,0],
                tmp_command_quaternion[3,0],
            ]
            sol = sp.odeint(exnxsofmotion, initial_conditions,time_chunk,args=(J,command_quaternion))
            final_solution = np.vstack((final_solution, sol[1:,:]))
            initial_conditions = sol[-1,:] #make sure indexed correctly
            #Normaliation of initial conditions before starting another ODE step
            initial_conditions[3:7] = initial_conditions[3:7]/np.linalg.norm(initial_conditions[3:7])
    else:
        for time_chunk in time_chunks: #dont want inital time of 0
            sol = sp.odeint(exnxsofmotion, initial_conditions,time_chunk,args=(J,command_quaternion))
            final_solution = np.vstack((final_solution, sol[1:,:]))
            initial_conditions = sol[-1,:] #make sure indexed correctly
            #Normaliation of initial conditions before starting another ODE step
            initial_conditions[3:7] = initial_conditions[3:7]/np.linalg.norm(initial_conditions[3:7])
    return final_solution, time_vector, output_vector


def plot_results(t_vec, sol, t2_vec, outputs):
    sol_name = ["W1","W2",'W3',"Q1","Q2","Q3","Q4"]
    output_name = ["T1","T2","T3","err"]

    #plot angular velocities
    fig1 = plt.figure(1)
    for i in range(3):
        solution_vector = sol[:,i]
        fig1.add_subplot(3,1,1+i%3)
        plt.plot(
            t_vec, solution_vector
        )
        plt.ylabel(sol_name[i])
        if i==0: plt.title("Angular Velocities (rads/s)")

    #plot quaternions
    fig2 = plt.figure(2)
    for i in range(4):
        solution_vector = sol[:,3+i]
        fig2.add_subplot(4,1,1+i%4)
        plt.plot(
            t_vec, solution_vector
        )
        plt.ylabel(sol_name[3+i])
        if i==0: plt.title("Quaterions")
    #plot outputs

    fig3 = plt.figure(3)
    for i in range(3): #hopefully this works
        output_vector = outputs[:,i]
        fig3.add_subplot(3,1,1+i%3)
        plt.plot(
            t2_vec, output_vector
        )
        plt.ylabel(output_name[i])
        if i==0: plt.title("Torques (N.m)")
    
    fig4 = plt.figure(4)
    fig4.add_subplot(1,1,1)
    plt.plot( t2_vec, outputs[:,-1])
    plt.title("Attitude Error (rads)")

    '''
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
    '''
    plt.show()


def save_results(t_vec, sol, t2_vec, outputs):
    #store results
    sol_name = ["W1","W2",'W3',"Q1","Q2","Q3","Q4"]
    output_name = ["T1","T2","T3","err"]
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
    difference = (csv_orientation_list[-1][0] - initial_date_time)
    total_seconds = difference.total_seconds()

    t0 = 0
    tf = total_seconds
    n = int(total_seconds//2) + 1 
    t_vec = np.linspace(t0,tf,n)

    #initial_quaternion = att.DCM2Quat(np.matrix(dis.get_Aio(np.matrix(csv_orientation_list[0][1]).transpose(),
    #                                                        np.matrix(csv_orientation_list[0][2]).transpose())).transpose())
    initial_conditions = [  
        0.0,0.0,0.0, 
        0.5,0.5,0.5,0.5
    ]
    solution, time_vector, output_vector = solver( 
                        initial_conditions,    #initial states
                        t_vec,                  #t_vec to integrate over
                        [77217,77217,25000],          #J (Principle axis MOI) vector
                        [ 0.1830127, 0, 0.1830127, 0.9659258 ]       #command quaternion
    )      
    #att.Animate_Attitude_Set(np.array(solution[:,3:7]).transpose(),10/100) #TODO figure out timing parameter
    
    output_vector = np.asarray(output_vector)
    plot_results(t_vec, solution, time_vector, output_vector)
    save_results(t_vec, solution, time_vector, output_vector)
