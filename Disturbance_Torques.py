#Detailed Disturbance Torque Calculations
import Attitude_Kinematics as att
import numpy as np
import datetime
import csv
import matplotlib.pyplot as plt

class Orientation:#TODO make sure this isnt doing anything weird with copy/deep copy solution
    def __init__(self,  date            = datetime.datetime(1,1,1,1,1,1),
                        position        = np.matrix([ [0], [0], [0] ]).astype(float), 
                        velocity        = np.matrix([ [0], [0], [0] ]).astype(float), 
                        euler_angles    = np.array([0,0,0])):
        self.date           = date
        self.position       = position
        self.velocity       = velocity
        self.euler_angles   = euler_angles

class Plate:
    def __init__(self,  area            = 0.0,
                        normal          = np.matrix([ [1], [0], [0] ]).astype(float), 
                        lever           = np.matrix([ [1], [0], [0] ]).astype(float) , 
                        r_coef          = np.array([0,0,0])):
        self.area   = area
        self.normal = normal
        self.lever  = lever
        self.r_coef = r_coef


def sind(deg):
    return np.sin(np.deg2rad(deg))


def cosd(deg):
    return np.cos(np.deg2rad(deg))


#importing atmospheric data from csv
atmos_table = []
with open('Atmos_Density_Model.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        atmos_table.append([ float(row[0]), float(row[1]), float(row[2]) ])


#constants
PHI = 1361 #solar constant TODO units
c   = 3*10**8 #speed of light (m/s)
Wo  = np.matrix([ [0], [0], [0.000072921158553]]).astype(float) #earths rotation in rads/s
Cd  = 2.25 #typical coefficient of drag ~2-2.5
Me  = 5.97219*10**24 #earth mass
Gc  = 6.67408*10**-11 #gravitational constant
u   = Me * Gc #earth gravitational constant
#I THINK THIS MUST BE CONVERTED FROM NANO TO REG SI SO 10E-9 TODO: FIGURE OUT IF THIS IS CORRECT

m_vec = 7.77*10**(22.0-7)*np.matrix([ [sind(169.7)*cosd(108.2)], 
                                    [sind(169.7)*sind(108.2)], 
                                    [cosd(169.7)            ] ]).astype(float)



#gives sattelite-sun vector in ECI frame
def get_rsato(date,r_sattoearth):
    Tut1    = (att.Get_Julian_Datetime(date) - 2451545) / 36525
    Mo      = 357.5277233 + 35999.05034*Tut1  #all of the subsequent calculations are in terms of degrees
    Ep      = 23.439291 - 0.0130042*Tut1
    Phi_o   = 280.46 + 36000.771*Tut1
    Phi_ecl = Phi_o + 1.914666471*sind(Mo) + 0.019994643*sind(2*Mo)

    roo_unitvec = np.matrix([ [cosd(Phi_ecl)], [cosd(Ep)*sind(Phi_ecl)], [sind(Ep)*sind(Phi_ecl)] ]).astype(float)
    roo_mag     = 1.000140612 - 0.016708617*cosd(Mo) - 0.000139589*cosd(2*Mo)

    r_sattoearth    = np.multiply(1/(1.496*10**8), r_sattoearth) #r_sattoearth given in km converted to AU
    r_sato          = np.multiply(roo_mag, roo_unitvec) - r_sattoearth

    return r_sato


#angle_vec is angles arranged in the order which they are applied, order_vec is the order of axes for the transformation
def get_dcm(angle_vec, order_vec): 
    #this functions uses radians... degrees must be converted before input 
    ang = np.array([0.0,0.0,0.0]) #if these are floats then float inputs dont work?
    for i in range(3):
        ang[i] = angle_vec[order_vec[i]-1]

    z_DCM = np.matrix([
        [ np.cos(ang[0]), np.sin(ang[0]), 0 ],
        [-np.sin(ang[0]), np.cos(ang[0]), 0 ],
        [ 0             , 0             , 1 ]
    ]).astype(float)
    y_DCM = np.matrix([
        [ np.cos(ang[1]), 0,-np.sin(ang[1]) ],
        [ 0             , 1, 0              ],
        [ np.sin(ang[1]), 0, np.cos(ang[1]) ]
    ]).astype(float)
    x_DCM = np.matrix([
        [  1, 0             , 0             ],
        [  0, np.cos(ang[2]), np.sin(ang[2])],
        [  0,-np.sin(ang[2]), np.cos(ang[2])]
    ]).astype(float)
    
    DCM_array = np.array([ x_DCM, y_DCM, z_DCM ]) # x==1, y==2, z==3 for purpose of DCM description
    DCM = np.matrix([
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]
    ]).astype(float)
    for i in range(3):
        DCM = np.matmul( DCM_array[order_vec[i]-1], DCM)

    return DCM


#DCM which converts orbital to inertial frame (often you want the reverse transformation so transpose)
def get_Aio(position, velocity): 
    Aio = np.matrix([
        [0,0,0],
        [0,0,0],
        [0,0,0]
    ]).astype(float)

    Aio[:,0]    = -1/np.linalg.norm(position)*position
    h           = np.cross(position.transpose(), velocity.transpose())
    Aio[:,1]    = np.multiply( -h, 1/np.linalg.norm(h)).transpose()
    Aio[:,2]    = np.cross( Aio[:,1].transpose(), Aio[:,0].transpose() ).transpose()
    return Aio


def get_density(alt):

    #indexing atmos table to find correct altitude interval
    i=0
    while alt >= atmos_table[i][0]:
        i+=1
    
    h0      = atmos_table[i-1][0]
    rho_0   = atmos_table[i-1][1]
    H       = atmos_table[i-1][2]
    
    #calculating density from exponential model
    rho     = rho_0*np.exp(-(alt-h0)/H)
    return rho


#calculation of solar torque on plate in body frame
def get_solartorque(sat_orientation, plate_array):
    #transformation Abi
    Abo     = get_dcm(sat_orientation.euler_angles, np.array([3,2,1]))
    Aoi     = get_Aio(sat_orientation.position, sat_orientation.velocity).transpose()
    Abi     = np.matmul( Abo, Aoi)

    #sun-sat vector in body frame
    r_sato  = get_rsato(sat_orientation.date, sat_orientation.position)
    s       = np.matmul( Abi, r_sato)
    s_unit = 1/np.linalg.norm(s)*s #satellite-sun vector in body frame

    #calculation
    L_srp_total = np.matrix([ [0],[0],[0] ]).astype(float)
    for plate in plate_array:

        #incidence angle of sun-sat vector and plate normal
        cos_theta_srp = np.matmul( plate.normal.transpose(), s )[0,0]

        #calculation of solar pressure force of plate (in body frame)
        r_diff  = plate.r_coef[0]#note order
        r_spec  = plate.r_coef[1]
        F_srp   = -PHI/(c*np.linalg.norm(r_sato))*plate.area*( 2*(r_diff/3+r_spec*cos_theta_srp) + (1-r_spec)*s )*max(cos_theta_srp,0)

        #calculation of solar presure torque of plate
        L_srp = np.cross( plate.lever.transpose(), F_srp.transpose() ).transpose()
        L_srp_total = L_srp_total + L_srp
        
    if np.linalg.norm(L_srp_total) < 10**(-15):
        L_srp_total = np.matrix([
            [0],
            [0],
            [0]
        ])
    else: 
        L_srp_unit = 1/np.linalg.norm(L_srp_total)*L_srp_total

    #plotting results of disturbance torques
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    #creating 3d plot with plate vectors
    i=0
    for plate in plate_array:
        if i == 0:
            line = ax.plot( [0,plate.area*plate.normal[0]],
                            [0,plate.area*plate.normal[1]],
                            [0,plate.area*plate.normal[2]],
                            'r', label="Plate Normal x Area Vector" ) #[x],[y],[z]
        line = ax.plot( [0,plate.area*plate.normal[0]],
                            [0,plate.area*plate.normal[1]],
                            [0,plate.area*plate.normal[2]],
                            'r' ) #[x],[y],[z]
        i+=1

    #plotting relative velocity in body frame
    line = ax.plot( [0,s_unit[0]],
                    [0,s_unit[1]],
                    [0,s_unit[2]],
                    'b', label="Sun-Satellite Vector")
    #plotting resulting total torque vector in body frame
    line = ax.plot( [0,L_srp_unit[0]],
                    [0,L_srp_unit[1]],
                    [0,L_srp_unit[2]],
                    'g', label="Resultant Torque")

    #axes and legend
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.5, 1.5])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.5, 1.5])
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    return L_srp_total


def get_aerotorque(sat_orientation, plate_array):
    #getting body orientation w.r.t. ECI
    Aoi = get_Aio(sat_orientation.position, sat_orientation.velocity).transpose()
    Abo = get_dcm(sat_orientation.euler_angles, np.array([3,2,1]))
    Abi = np.matmul(Abo, Aoi)

    #calculate Vrel in body
    v_reli = sat_orientation.position + np.matrix(np.cross( Wo.transpose(), 
                                                            sat_orientation.position.transpose() )).astype(float).transpose()
    v_relb = np.matmul( Abi, v_reli )
    v_relb_unit = 1/np.linalg.norm(v_relb)*v_relb
    
    #calculation
    L_aero_total = np.matrix([ [0],[0],[0] ]).astype(float)
    for plate in plate_array:
       
        #calculation force on plate due to aerodynamic pressure F_aero
        cos_theta_aero  = 1/np.linalg.norm(v_relb)*np.matmul( plate.normal.transpose(), v_relb)[0,0]
        alt             = np.linalg.norm(sat_orientation.position) - 6371.0
        rho             = get_density(alt)
        F_aero          = -0.5*rho*Cd*np.linalg.norm(v_relb)*plate.area*max(cos_theta_aero,0)*v_relb

        #calculation of torque on plate due to aerodynamic pressure L_aero
        L_aero          = np.cross( plate.lever.transpose(), F_aero.transpose() ).transpose()
        L_aero_total    = L_aero_total + L_aero   

    if np.linalg.norm(L_aero_total) < 10**(-15):
        L_total_unit = np.matrix([
            [0],
            [0],
            [0]
        ])
    else:
        L_total_unit = 1/np.linalg.norm(L_aero_total)*L_aero_total 

    #plotting results of disturbance torques
    fig = plt.figure()
    ax  = plt.axes(projection='3d')
    #creating 3d plot with plate vectors
    i=0
    for plate in plate_array:
        if i == 0:
            line = ax.plot( [0,plate.area*plate.normal[0]],
                            [0,plate.area*plate.normal[1]],
                            [0,plate.area*plate.normal[2]],
                            'r', label="Plate Normal x Area Vector" ) #[x],[y],[z]
        line = ax.plot( [0,plate.area*plate.normal[0]],
                            [0,plate.area*plate.normal[1]],
                            [0,plate.area*plate.normal[2]],
                            'r' ) #[x],[y],[z]
        i+=1

    #plotting relative velocity in body frame
    line = ax.plot( [0,v_relb_unit[0]],
                    [0,v_relb_unit[1]],
                    [0,v_relb_unit[2]],
                    'b', label="Relative Velocity")
    #plotting resulting total torque vector in body frame
    line = ax.plot( [0,L_total_unit[0]],
                    [0,L_total_unit[1]],
                    [0,L_total_unit[2]],
                    'g', label="Resultant Torque")

    #axes and legend
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.5, 1.5])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.5, 1.5])
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    return L_aero_total


#calculated gravity torque from position, orientation and principle axis MOIs
def get_gravitytorque(sat_orientation, Jp): 
    '''
    psi     = sat_orientation.euler_angles[0]
    theta   = sat_orientation.euler_angles[1]
    '''
    #getting body orientation w.r.t. ECI
    Aoi = get_Aio(sat_orientation.position, sat_orientation.velocity).transpose()
    Abo = get_dcm(sat_orientation.euler_angles, np.array([3,2,1]))
    Abi = np.matmul(Abo, Aoi)
    
    r = sat_orientation.position*10**3.0
    Rc      = np.linalg.norm(sat_orientation.position)*10**3.0
    J_mat = np.matrix([
        [Jp[0], 0,      0       ],
        [0,     Jp[1],  0       ],
        [0,     0,      Jp[2]   ]
    ])
    '''
    vec     = np.matrix([
        [(Jp[2]-Jp[1])*cosd(theta)**2.0*cosd(psi)*sind(psi) ],
        [(Jp[2]-Jp[0])*cosd(theta)*sind(theta)*cosd(psi)    ],
        [(Jp[0]-Jp[1])*cosd(theta)*sind(theta)*sind(psi)    ]
    ]).astype(float)
    L_grav  = 3*u/Rc**3.0*vec
    '''
    L_grav = 3*u/Rc**5.0*np.cross(  np.matmul( Abi, r ).transpose(),
                                    np.matmul(J_mat, np.matmul( Abi, r ) ).transpose() ).transpose()

    if np.linalg.norm(L_grav) > 10**(-15):
        L_grav_unit = 1/np.linalg.norm(L_grav)*L_grav
    else:
        L_grav_unit = np.matrix([
            [0],
            [0],
            [0]
        ])
    
    #plotting
    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    #plotting local gravity vector in body frame
    g_local = np.matmul( get_dcm(sat_orientation.euler_angles, np.array([3,2,1])),
                         np.matrix([ [0], [0], [1] ]) ) #gravity vector in LVLH frame is [0,0,1]

    #plotting principle axes of intertia (as relative size)
    for i in range(3): #goes x,y,z
        plot = np.array([   [0, 0],
                            [0, 0],
                            [0, 0] ]).astype(float)
        plot[i,:] = np.array([-Jp[i]/np.linalg.norm(Jp), Jp[i]/np.linalg.norm(Jp)])
        
        line = ax.plot( plot[0,:],
                        plot[1,:],
                        plot[2,:],
                        'r',label="Principle MOIs")
    
    #plotting local gravity vector in body frame
    line = ax.plot( [0,g_local[0]],
                    [0,g_local[1]],
                    [0,g_local[2]],
                    'g',label="Local Gravity")

    #plotting torque due to gravity in body frame
    line = ax.plot( [0,L_grav_unit[0]],
                    [0,L_grav_unit[1]],
                    [0,L_grav_unit[2]],
                    'b',label='Gravity Torque Vector')
    
    #axes and legend
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.5, 1.5])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.5, 1.5])
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    
    return L_grav


def get_magnetictorque(sat_orientation, magnetic_residual):
    residual_unit = 1/np.linalg.norm(magnetic_residual)*magnetic_residual
    r = 10**3.0*sat_orientation.position
    Bi = ( 3*np.matmul(m_vec.transpose(), r)[0,0]*r - np.linalg.norm(r)**2.0*m_vec) / np.linalg.norm(r)**5.0
    
    #getting body orientation w.r.t. ECI
    Aoi = get_Aio(sat_orientation.position, sat_orientation.velocity).transpose()
    Abo = get_dcm(sat_orientation.euler_angles, np.array([3,2,1]))
    Abi = np.matmul(Abo, Aoi)

    #calculation of magnetic torque
    Bb = np.matmul(Abi, Bi)
    Bb_unit = 1/np.linalg.norm(Bb)*Bb
    L_magnetic = np.cross( magnetic_residual.transpose(), Bb.transpose() ).transpose()
    
    if np.linalg.norm(L_magnetic) < 10**(-15):
        L_mag_unit = np.matrix([
            [0],
            [0],
            [0]
        ])
    else:
        L_mag_unit = 1/np.linalg.norm(L_magnetic)*L_magnetic

    #plotting
    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    #lines
    line = ax.plot( [0,Bb_unit[0]],
                    [0,Bb_unit[1]], 
                    [0,Bb_unit[2]],
                    'r',label="Earth Magnetic Field")
    line = ax.plot( [0,residual_unit[0]],
                    [0,residual_unit[1]],
                    [0,residual_unit[2]],
                    'g',label="Sat Magnetic Residual")
    line = ax.plot( [0,L_mag_unit[0]],
                    [0,L_mag_unit[1]],
                    [0,L_mag_unit[2]],
                    'b',label="Magnetic Torque")
    
    #axes and legend
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylim3d([-1.5, 1.5])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-1.5, 1.5])
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

    return L_magnetic


if __name__=="__main__":

    #calculating satellite configuration values
    #orientation
    date    = datetime.datetime(2021,11,3,10,9,0)
    angles  = np.array([45,45,45]) #3-2-1 angles in order which they are applied 
    angles  = np.multiply(np.pi/180, angles)
    position = np.matrix([
        [0],
        [0],
        [6971]
    ]).astype(float)
    v_mag = np.sqrt(u/(np.linalg.norm(position)*10**3.0)) / (10**3.0)
    velocity = np.matrix([
        [0],
        [v_mag],
        [0],
    ]).astype(float)

    sat_orientation = Orientation() #this will get initialized with default values... must change to ones you want
    sat_orientation.date = date
    sat_orientation.position = position #describes position relative to center of earth in ECI frame
    sat_orientation.velocity = velocity  #these are in km and km/s units 
    sat_orientation.euler_angles = angles #in degrees

    #constructing plate(s) attributes
    #doing a rectangular prism
    R_vec = np.array([0.7,0.1,0.2]) #this is the optical coeficients for the plates

    bot         = Plate()
    bot.area    = 1.0
    bot.lever   = np.matrix([ [0],[0],[-1.0] ]).astype(float)
    bot.normal  = np.matrix([ [0],[0],[-1.0] ]).astype(float)
    bot.r_coef  = R_vec

    top         = Plate()
    top.area    = 1.0
    top.lever   = np.matrix([ [0],[0],[1.0] ]).astype(float)
    top.normal  = np.matrix([ [0],[0],[1.0] ]).astype(float)
    top.r_coef  = R_vec
    
    north           = Plate()
    north.area      = 2.0
    north.lever     = np.matrix([ [0],[0.5],[0] ]).astype(float)
    north.normal    = np.matrix([ [0],[1.0],[0] ]).astype(float)
    north.r_coef    = R_vec

    south           = Plate()
    south.area      = 2.0
    south.lever     = np.matrix([ [0],[-0.5],[0] ]).astype(float)
    south.normal    = np.matrix([ [0],[-1.0],[0] ]).astype(float)
    south.r_coef    = R_vec

    west        = Plate()
    west.area   = 2.0
    west.lever  = np.matrix([ [-0.5],[0],[0] ]).astype(float)
    west.normal = np.matrix([ [-1.0],[0],[0] ]).astype(float)
    west.r_coef = R_vec

    east        = Plate()
    east.area   = 2.0
    east.lever  = np.matrix([ [0.5],[0],[0] ]).astype(float)
    east.normal = np.matrix([ [1.0],[0],[0] ]).astype(float)
    east.r_coef = R_vec

    plate_array = [bot, top, south, west, east, north]

    print("Torque due to solar pressure on 1m^2 face cube")
    print(get_solartorque(sat_orientation,plate_array)) #I think this works

    print("Torque due to aerodynamic forces on 1m^2 face cube")
    print(get_aerotorque(sat_orientation,plate_array)) #seems right

    print("Torque due to gravity gradient")
    principle_moments = np.array([ 25000, 25000, 77217 ]) #must define principle moment of intertia for satellite
    print(get_gravitytorque(sat_orientation,principle_moments)) #seems rights

    print("Torque due to magnetic interation")
    mag_dipole = np.matrix([ [20], [20], [20] ]).astype(float) #must define a residual magnetic dipole for the satellite
    print(get_magnetictorque(sat_orientation, mag_dipole))
