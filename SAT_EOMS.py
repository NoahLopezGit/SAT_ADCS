import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
import pandas as pd
import Attitude_Kinematics as att

#Satellite Integrator Model
def exnxsofmotion(x_vec, t_vec, J):
    '''
    Need in for for scipy odeint.
    vector of explicit differential exn (RHS)
    vector of corresponding states (not differential)
    can only go first order so higher order diffeq must be described as system of first order
    '''
    w1,w2,w3,q1,q2,q3,q4 = x_vec
    #constructing RHS vector TODO add L functionality here
    RHS = [
        1/J[0]*( -( w2*w3*J[2] - w3*w2*J[1])),
        1/J[1]*( -( w3*w1*J[0] - w1*w3*J[2])),
        1/J[2]*( -( w1*w2*J[1] - w2*w1*J[0])),
        0.5*( q4*w1 - q3*w2 + q2*w3) / qnorm(w1,w2,w3,q1,q2,q3,q4), #kindof a bad way to normalize q I think
        0.5*( q3*w1 - q4*w2 - q1*w3) / qnorm(w1,w2,w3,q1,q2,q3,q4),
        0.5*(-q2*w1 - q1*w2 + q4*w3) / qnorm(w1,w2,w3,q1,q2,q3,q4),
        0.5*(-q1*w1 - q2*w2 - q3*w3) / qnorm(w1,w2,w3,q1,q2,q3,q4),
    ]
    return RHS


def qnorm(w1,w2,w3,q1,q2,q3,q4): #effectively normalizes the quaternion... but is this keeping simulation more accurate (seems to not have much effect)
    q_new = np.array([
        q1 + 0.5*( q4*w1 - q3*w2 + q2*w3),
        q2 + 0.5*( q3*w1 - q4*w2 - q1*w3),
        q3 + 0.5*(-q2*w1 - q1*w2 + q4*w3),
        q4 + 0.5*(-q1*w1 - q2*w2 - q3*w3)
    ])
    qnorm = np.linalg.norm(q_new)
    return qnorm

def solver(exn,x0_vec,t_vec,J):
    sol = sp.odeint(exn, x0_vec,t_vec,args=(J,))
    return sol


def plot_results(sol, t_vec):
    #plot results
    fig_dict = {}
    for i in range(len(sol[0,:])):
        if i % 4==0:
            n = 1+i//4
            fig = "fig" + str(n)
            fig_dict[fig] = plt.figure(n)
        solution_vector = sol[:,i]
        fig_dict[fig].add_subplot(4,1,1+i%4)
        plt.plot(
            t_vec, solution_vector
        )
        plt.title("State{0}".format(i))
    plt.show()


def save_results(sol, t_vec):
    #store results
    data = {}
    data["time (s)"] = t_vec
    for i in range(len(sol[0,:])):
        solution_vector = sol[:,i]
        data["state{0}".format(i)] = solution_vector
    df = pd.DataFrame(data)
    print(df)
    writer = pd.ExcelWriter('State_Integration.xlsx', engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()


if __name__=="__main__":
    t0 = 0
    tf = 20
    n = 201
    t_vec = np.linspace(t0,tf,n)
    solution = solver(exnxsofmotion, [0.1,0.1,1,0,0,0,1], t_vec, [100,100,500])
    
    for instance in solution:
        instance[3:7] = instance[3:7] /np.linalg.norm(instance[3:7]) #trying to normalize quaternions... dont think this will work


    att.Animate_Attitude_Set(np.array(solution[:,3:7]).transpose(),10/100)
    plot_results(solution, t_vec)
    save_results(solution, t_vec)
