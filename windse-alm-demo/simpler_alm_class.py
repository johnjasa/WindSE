from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import glob

def SimplerUpdateActuatorLineForce(problem, u_local, simTime_id, dt, turb_i, mpi_u_fluid, dfd=None, verbose=False):
    
    print(u_local.vector()[:])

    simTime = problem.simTime_list[simTime_id]

    if verbose:
        print("Current Optimization Time: "+repr(simTime)+", Turbine #"+repr(turb_i))
        sys.stdout.flush()

    def rot_x(theta):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])

        return Rx

    def rot_y(theta):
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        
        return Ry

    def rot_z(theta):
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        
        return Rz

    def build_lift_and_drag(problem, u_rel, blade_unit_vec, rdim, twist, c):

        def get_angle_between_vectors(a, b, n):
            a_x_b = np.dot(np.cross(n, a), b)

            norm_a = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
            norm_b = np.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])

            c1 = a_x_b/(norm_a*norm_b)
            c1 = np.clip(c1, -1.0, 1.0)
            aoa_1 = np.arcsin(c1)

            c2 = np.dot(a, b)/(norm_a*norm_b)
            c2 = np.clip(c2, -1.0, 1.0)
            aoa_2 = np.arccos(c2)
            
            if aoa_2 > np.pi/2.0:
                if aoa_1 < 0:
                    aoa_1 = -np.pi - aoa_1
                else:
                    aoa_1 = np.pi - aoa_1
            
            aoa_1_deg = aoa_1/np.pi*180.0
            
            return aoa_1


        # If this is the first time calling the function...
        # if problem.first_call_to_alm: # This breaks in parallel, use the version below

        if not hasattr(problem, 'interp_lift'):
            # build the lift-drag table interpolators
            rdim_all = np.linspace(0, rdim[-1], np.shape(problem.lift_table)[1])
            problem.interp_lift = interp.RectBivariateSpline(problem.interp_angles, rdim_all, problem.lift_table)
            problem.interp_drag = interp.RectBivariateSpline(problem.interp_angles, rdim_all, problem.drag_table)


        # Initialize the real cl and cd profiles
        real_cl = np.zeros(problem.num_blade_segments)
        real_cd = np.zeros(problem.num_blade_segments)

        # fp = open(problem.aoa_file, 'a')

        tip_loss = np.zeros(problem.num_blade_segments)

        for k in range(problem.num_blade_segments):
            # Get the relative wind velocity at this node
            wind_vec = u_rel[:, k]

            # Remove the component in the radial direction (along the blade span)
            wind_vec -= np.dot(wind_vec, blade_unit_vec[:, 1])*blade_unit_vec[:, 1]

            # aoa = get_angle_between_vectors(arg1, arg2, arg3)
            # arg1 = in-plane vector pointing opposite rotation (blade sweep direction)
            # arg2 = relative wind vector at node k, including blade rotation effects (wind direction)
            # arg3 = unit vector normal to plane of rotation, in this case, radially along span
            aoa = get_angle_between_vectors(-blade_unit_vec[:, 2], wind_vec, -blade_unit_vec[:, 1])

            # Compute tip-loss factor
            if rdim[k] < 1e-12:
                tip_loss[k] = 1.0
            else:
                loss_exponent = 3.0/2.0*(rdim[-1]-rdim[k])/(rdim[k]*np.sin(aoa))
                acos_arg = np.exp(-loss_exponent)
                acos_arg = np.clip(acos_arg, -1.0, 1.0)
                tip_loss[k] = 2.0/np.pi*np.arccos(acos_arg)

            # Remove the portion of the angle due to twist
            aoa -= twist[k]

            # Store the cl and cd by interpolating this (aoa, span) pair from the tables
            real_cl[k] = problem.interp_lift(aoa, rdim[k])
            real_cd[k] = problem.interp_drag(aoa, rdim[k])

            # Write the aoa to a file for future reference
            # fp.write('%.5f, ' % (aoa/np.pi*180.0))

        # fp.close()

        return real_cl, real_cd, tip_loss


    #================================================================
    # Get Mesh Properties
    #================================================================

    ndim = problem.dom.dim

    # Initialize a cumulative turbine force function (contains all turbines)
    tf = Function(problem.fs.V)
    tf.vector()[:] = 0.0

    # Initialize a cylindrical field function
    cyld = Function(problem.fs.V)

    #================================================================
    # Set Turbine and Fluid Properties
    #================================================================

    # Set the density
    rho = 1.0

    # Set the number of blades in the turbine
    num_blades = 3

    # Width of Gaussian
    # Note: this sets the gaussian width to roughly twice the minimum cell length scale
    eps = problem.gaussian_width

    # initialize numpy torque
    rotor_torque_numpy_temp = 0.0

    # Blade length (turbine radius)
    L = problem.farm.radius[turb_i]

    #================================================================
    # Set Derived Constants
    #================================================================

    # Calculate the radial position of each actuator node
    rdim = np.linspace(0.0, L, problem.num_blade_segments)

    # Calculate width of an individual blade segment
    # w = rdim[1] - rdim[0]
    w = (rdim[1] - rdim[0])*np.ones(problem.num_blade_segments)
    w[0] = w[0]/2.0
    w[-1] = w[-1]/2.0

    # Calculate an array describing the x, y, z position of each actuator node
    # Note: The basic blade is oriented along the +y-axis
    blade_pos_base = np.vstack((np.zeros(problem.num_blade_segments),
                        rdim,
                        np.zeros(problem.num_blade_segments)))

    # Calculate the blade velocity
    angular_velocity = 2.0*np.pi*problem.rpm/60.0
    tip_speed = angular_velocity*L
    # tip_speed = 9.0

    # Specify the velocity vector at each actuator node
    # Note: A blade with span oriented along the +y-axis moves in the +z direction
    blade_vel_base = np.vstack((np.zeros(problem.num_blade_segments),
                           np.zeros(problem.num_blade_segments),
                           np.linspace(0.0, tip_speed, problem.num_blade_segments)))

    # Set the spacing pf each blade
    theta_vec = np.linspace(0.0, 2.0*np.pi, num_blades, endpoint = False)

    # Create unit vectors aligned with blade geometry
    # blade_unit_vec_base[:, 0] = points along rotor shaft
    # blade_unit_vec_base[:, 1] = points along blade span axis
    # blade_unit_vec_base[:, 2] = points tangential to blade span axis (generates a torque about rotor shaft)
    blade_unit_vec_base = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

    #================================================================
    # Begin Calculating Turbine Forces
    #================================================================

    # Read cl and cd from the values specified in problem manager
    twist = np.array(problem.mtwist[turb_i], dtype = float)

    cl = np.array(problem.mcl[turb_i], dtype = float)
    cd = np.array(problem.mcd[turb_i], dtype = float)
    tip_loss = np.ones(problem.num_blade_segments)

    # Read the chord length from the values specified in the problem manager
    # c = L/20.0
    c = np.array(problem.mchord[turb_i], dtype = float)


    # Initialze arrays depending on what this function will be returning
    if dfd is None:
        tf_vec = np.zeros(np.size(problem.coords))
        tf_vec_for_power = np.zeros(np.size(problem.coords))
        lift_force = np.zeros((np.shape(problem.coords)[0], ndim))
        drag_force = np.zeros((np.shape(problem.coords)[0], ndim))

    # Calculate the blade position based on current simTime and turbine RPM
    period = 60.0/problem.rpm
    # theta_offset = simTime/period*2.0*np.pi
    theta_offset = (simTime+0.5*dt)/period*2.0*np.pi
    # theta_offset = 0.0


    # Treat each blade separately
    for blade_ct, theta_0 in enumerate(theta_vec):
        # If the minimum distance between this mesh and the turbine is >2*RD,
        # don't need to account for this turbine
        if problem.min_dist[turb_i] > 2.0*(2.0*L):
            break

        theta = theta_0 + theta_offset

        # Generate a rotation matrix for this turbine blade
        Rx = rot_x(theta)
        Rz = rot_z(float(problem.farm.myaw[turb_i]))

        # Rotate the blade velocity in the global x, y, z, coordinate system
        # Note: blade_vel_base is negative since we seek the velocity of the fluid relative to a stationary blade
        # and blade_vel_base is defined based on the movement of the blade
        blade_vel = np.dot(Rz, np.dot(Rx, -blade_vel_base))

        # Rotate the blade unit vectors to be pointing in the rotated positions
        blade_unit_vec = np.dot(Rz, np.dot(Rx, blade_unit_vec_base))

        # Rotate the entire [x; y; z] matrix using this matrix, then shift to the hub location
        blade_pos = np.dot(Rz, np.dot(Rx, blade_pos_base))
        blade_pos[0, :] += problem.farm.x[turb_i]
        blade_pos[1, :] += problem.farm.y[turb_i]
        blade_pos[2, :] += problem.farm.z[turb_i]

        time_offset = 1
        if simTime_id < time_offset:
            theta_behind = theta_0 + 0.5*(problem.simTime_list[simTime_id]+simTime)/period*2.0*np.pi
        else:
            theta_behind = theta_0 + 0.5*(problem.simTime_list[simTime_id-time_offset]+simTime)/period*2.0*np.pi
            # if blade_ct == 0:
            #     print('SimTime = %f, using %f' % (simTime, problem.simTime_list[-time_offset]))

        Rx_alt = rot_x(theta_behind)
        blade_pos_alt = np.dot(Rz, np.dot(Rx_alt, blade_pos_base))
        blade_pos_alt[0, :] += problem.farm.x[turb_i]
        blade_pos_alt[1, :] += problem.farm.y[turb_i]
        blade_pos_alt[2, :] += problem.farm.z[turb_i]

        # Initialize space to hold the fluid velocity at each actuator node
        u_fluid = np.zeros((3, problem.num_blade_segments))

        # Generate the fluid velocity from the actual node locations in the flow
        for k in range(problem.num_blade_segments):

            u_fluid[:, k] = u_local(blade_pos_alt[0, k],
                     blade_pos_alt[1, k],
                     blade_pos_alt[2, k])

            u_fluid[:, k] -= np.dot(u_fluid[:, k], blade_unit_vec[:, 1])*blade_unit_vec[:, 1]

#         # Read the fluid velocity for this blade from mpi_u_fluid
#         start_pt = blade_ct*3*problem.num_blade_segments
#         end_pt = start_pt + 3*problem.num_blade_segments

#         u_fluid = mpi_u_fluid[turb_i, start_pt:end_pt]
#         u_fluid = np.reshape(u_fluid, (3, -1), 'F')

        for k in range(problem.num_blade_segments):
            u_fluid[:, k] -= np.dot(u_fluid[:, k], blade_unit_vec[:, 1])*blade_unit_vec[:, 1]


#         problem.blade_pos_previous[blade_ct] = blade_pos
                        
        # Form the total relative velocity vector (including velocity from rotating blade)
        u_rel = u_fluid + blade_vel
        
        u_rel_mag = np.linalg.norm(u_rel, axis=0)
        u_rel_mag[u_rel_mag < 1e-6] = 1e-6
        u_unit_vec = u_rel/u_rel_mag
        
        cl, cd, tip_loss = build_lift_and_drag(problem, u_rel, blade_unit_vec, rdim, twist, c)
                
        # Calculate the lift and drag forces using the relative velocity magnitude
        lift = tip_loss*(0.5*cl*rho*c*w*u_rel_mag**2)
        drag = tip_loss*(0.5*cd*rho*c*w*u_rel_mag**2)

        # Tile the blade coordinates for every mesh point, [numGridPts*ndim x problem.num_blade_segments]
        blade_pos_full = np.tile(blade_pos, (np.shape(problem.coords)[0], 1))

        # Subtract and square to get the dx^2 values in the x, y, and z directions
        dx_full = (problem.coordsLinear - blade_pos_full)**2

        # Add together to get |x^2 + y^2 + z^2|^2
        dist2 = dx_full[0::ndim] + dx_full[1::ndim] + dx_full[2::ndim]

        # Calculate the force magnitude at every mesh point due to every node [numGridPts x NumActuators]
        nodal_lift = lift*np.exp(-dist2/eps**2)/(eps**3 * np.pi**1.5)
        nodal_drag = drag*np.exp(-dist2/eps**2)/(eps**3 * np.pi**1.5)

        for k in range(problem.num_blade_segments):
            # The drag unit simply points opposite the relative velocity unit vector
            drag_unit_vec = -np.copy(u_unit_vec[:, k])
            
            # The lift is normal to the plane generated by the blade and relative velocity
            lift_unit_vec = np.cross(drag_unit_vec, blade_unit_vec[:, 1])

            # All force magnitudes get multiplied by the correctly-oriented unit vector
            vector_nodal_lift = np.outer(nodal_lift[:, k], lift_unit_vec)
            vector_nodal_drag = np.outer(nodal_drag[:, k], drag_unit_vec)

            if dfd == None:
                lift_force += vector_nodal_lift
                drag_force += vector_nodal_drag


    if dfd == None:
        # The total turbine force is the sum of lift and drag effects
        turbine_force = drag_force + lift_force
        turbine_force_for_power = -drag_force + lift_force

        # Riffle-shuffle the x-, y-, and z-column force components
        for k in range(ndim):
            tf_vec[k::ndim] = turbine_force[:, k]
            tf_vec_for_power[k::ndim] = turbine_force_for_power[:, k]

        # Remove near-zero values
        tf_vec[np.abs(tf_vec) < 1e-12] = 0.0

        # Add to the cumulative turbine force
        tf.vector()[:] += tf_vec

    problem.first_call_to_alm = False

    tf.vector().update_ghost_values()

    if dfd == None:

        return tf
