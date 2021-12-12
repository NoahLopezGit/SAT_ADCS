import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt


#simple orbital model from which I can iteratively improve
#constants
gravity_constant = 6.67430*10**(-11.0)
mass_earth = 5.9722*10**24.0
time_steps = 50


#determine initial conditions in terms of baryonic center
def initial_conditions(mass_planet, mass_satellite, distance):
    system = np.matrix([
        [mass_planet, mass_satellite],
        [1          ,-1             ]
    ])
    equation_vector = np.matrix([[0],[distance]])
    body_positions = np.matrix([[0],[0]])
    body_positions = np.matmul( np.linalg.inv(system),
                                equation_vector)
    return body_positions


def get_orbital_period(position, mass_earth):
    orbital_period  = 2*np.pi*np.sqrt(np.linalg.norm(position)**3.0/(gravity_constant*mass_earth))
    return orbital_period


def orbital_model(X_vec,t_vec, G,M):
    x_pos,y_pos,x_vel,y_vel = X_vec
    RHS = [ #TODO this breaks when x_vel is given any non-zero value... something to do with RHS being non-lipshitz differencetiable
        x_vel,
        y_vel,
        -G*M * x_pos/(x_pos**2.0+y_pos**2.0)**(3.0/2.0),
        -G*M * y_pos/(x_pos**2.0+y_pos**2.0)**(3.0/2.0)
    ]
    return RHS


def get_plot(solution):
    plt.plot(   0.0,0.0,
                '.k',
                solution[:,0], solution[:,1],
                '.r')
    plt.title("Orbital Position (m)")
    plt.xlim(-10*10**6.0,10*10**6.0)
    plt.ylim(-10*10**6.0,10*10**6.0)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()


if __name__=="__main__":
    #print(initial_conditions(5.972*10**24,12246.994,(6371+600)*10**3))
    orbital_radius = (6371+600)*10**3.0
    orbital_radius2 = (6371+600)*10**3.0
    orbital_velocity = np.sqrt(gravity_constant*mass_earth/orbital_radius2)
    #X,Y,Vx,Vy
    states = [ orbital_radius,0.0, 0.0,orbital_velocity]
    orbital_period = get_orbital_period(states[0:2], mass_earth)
    t_vec = np.linspace(0,orbital_period,time_steps)
    
    sol = sp.odeint(orbital_model, states, t_vec,args=(gravity_constant,mass_earth))
    print(sol)
    get_plot(sol)