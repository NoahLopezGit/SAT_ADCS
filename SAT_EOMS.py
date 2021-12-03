import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
import pandas as pd

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
        0.5*( q4*w1 - q3*w2 + q2*w3),
        0.5*( q3*w1 - q4*w2 - q1*w3),
        0.5*(-q2*w1 - q1*w2 + q4*w3),
        0.5*(-q1*w1 - q2*w2 - q3*w3),
    ]
    return RHS


def solver(exn,t0,tf,n,J):
    x0_vec = [1,0,0,0,0,0,0]
    t_vec = np.linspace(t0,tf,n)
    sol = sp.odeint(exn, x0_vec,t_vec,parameters=[J])

    #plot results
    i=1
    for solution_vector in sol:
        plt.fig(i)
        plt.plot(
            t_vec, solution_vector
        )
        plt.title("State{0}".format(i))
        i+=1
    plt.show()
    
    #store results
    data = {}
    i=0
    for solution_vector in sol:
        data["state{0}".format(i)] = solution_vector
        i+=1
    df = pd.DataFrame(data)
    writer = pd.ExcelWriter("State_Integration.xlsx", engine='xlswriter')
    df.to_excel(writer, index=False)
