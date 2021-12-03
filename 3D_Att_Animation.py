#here is a function that will display attitude orientation in 3d space of precomputed attitude set (not live)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def Animate_Attitude_Set(data_y, data_x, data_z, h):

    if np.shape(data_z)[0] != 3 or np.shape(data_y)[0] != 3 or np.shape(data_x)[0] != 3:
        print("Input vectors exceed 3D")
        exit()

    if np.shape(data_z)[1] == np.shape(data_y)[1] and np.shape(data_z)[1] == np.shape(data_x)[1]:
        timesteps = len(data_y[1,:])
    else:
        print("Input vectors not same length")
        exit()

    #nested for now
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



        #fig.canvas.draw()
        #fig.canvas.flush_events()
        
    
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