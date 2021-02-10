from fenics import *
import numpy as np
import scipy.interpolate as interp
import openmdao.api as om

num_blades = 3


def rot_x(theta):
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )

    return Rx

def rot_z(theta):
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    return Rz

def build_lift_and_drag(problem, rdim):
    if not hasattr(problem, "interp_lift"):
        # build the lift-drag table interpolators
        rdim_all = np.linspace(0, rdim[-1], np.shape(problem.lift_table)[1])
        problem.interp_lift = interp.RectBivariateSpline(
            problem.interp_angles, rdim_all, problem.lift_table
        )
        problem.interp_drag = interp.RectBivariateSpline(
            problem.interp_angles, rdim_all, problem.drag_table
        )

def call_lift_and_drag(problem, u_rel, blade_unit_vec, rdim, twist, c):
    def get_angle_between_vectors(a, b, n):
        a_x_b = np.dot(np.cross(n, a), b)

        norm_a = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        norm_b = np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])

        c1 = a_x_b / (norm_a * norm_b)
        c1 = np.clip(c1, -1.0, 1.0)
        aoa_1 = np.arcsin(c1)

        c2 = np.dot(a, b) / (norm_a * norm_b)
        c2 = np.clip(c2, -1.0, 1.0)
        aoa_2 = np.arccos(c2)

        if aoa_2 > np.pi / 2.0:
            if aoa_1 < 0:
                aoa_1 = -np.pi - aoa_1
            else:
                aoa_1 = np.pi - aoa_1

        aoa_1_deg = aoa_1 / np.pi * 180.0

        return aoa_1

    # Initialize the real cl and cd profiles
    real_cl = np.zeros(problem.num_blade_segments)
    real_cd = np.zeros(problem.num_blade_segments)

    tip_loss = np.zeros(problem.num_blade_segments)

    for k in range(problem.num_blade_segments):
        # Get the relative wind velocity at this node
        wind_vec = u_rel[:, k]

        # Remove the component in the radial direction (along the blade span)
        wind_vec -= np.dot(wind_vec, blade_unit_vec[:, 1]) * blade_unit_vec[:, 1]

        # aoa = get_angle_between_vectors(arg1, arg2, arg3)
        # arg1 = in-plane vector pointing opposite rotation (blade sweep direction)
        # arg2 = relative wind vector at node k, including blade rotation effects (wind direction)
        # arg3 = unit vector normal to plane of rotation, in this case, radially along span
        aoa = get_angle_between_vectors(
            -blade_unit_vec[:, 2], wind_vec, -blade_unit_vec[:, 1]
        )

        # Compute tip-loss factor
        if rdim[k] < 1e-12:
            tip_loss[k] = 1.0
        else:
            loss_exponent = (
                3.0 / 2.0 * (rdim[-1] - rdim[k]) / (rdim[k] * np.sin(aoa))
            )
            acos_arg = np.exp(-loss_exponent)
            acos_arg = np.clip(acos_arg, -1.0, 1.0)
            tip_loss[k] = 2.0 / np.pi * np.arccos(acos_arg)

        # Remove the portion of the angle due to twist
        aoa -= twist[k]

        # Store the cl and cd by interpolating this (aoa, span) pair from the tables
        real_cl[k] = problem.interp_lift(aoa, rdim[k])
        real_cd[k] = problem.interp_drag(aoa, rdim[k])

    return real_cl, real_cd, tip_loss


class Preprocess(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        self.simTime_id = self.options['simTime_id']
        self.dt = self.options['dt']
        self.turb_i = self.options['turb_i']
        
        self.add_output('theta_vec', shape=num_blades)
        self.add_output('width', shape=self.problem.num_blade_segments)
        self.add_output('rdim', shape=self.problem.num_blade_segments)
        self.add_output('blade_pos_base', shape=(3, self.problem.num_blade_segments))
        self.add_output('blade_vel_base', shape=(3, self.problem.num_blade_segments))
        
    def compute(self, inputs, outputs):
        dfd = None
        problem = self.problem
        simTime_id = self.simTime_id
        dt = self.dt
        turb_i = self.turb_i
            
        simTime = problem.simTime_list[simTime_id]
        
        # ================================================================
        # Get Mesh Properties
        # ================================================================

        ndim = problem.dom.dim

        # ================================================================
        # Set Turbine and Fluid Properties
        # ================================================================

        # Set the number of blades in the turbine
        num_blades = 3

        # Width of Gaussian
        # Note: this sets the gaussian width to roughly twice the minimum cell length scale
        eps = problem.gaussian_width

        # Blade length (turbine radius)
        blade_length = problem.farm.radius[turb_i]

        # ================================================================
        # Set Derived Constants
        # ================================================================

        # Calculate the radial position of each actuator node
        rdim = np.linspace(0.0, blade_length, self.problem.num_blade_segments)
        outputs['rdim'] = rdim

        # Calculate width of an individual blade segment
        # width = rdim[1] - rdim[0]
        width = (rdim[1] - rdim[0]) * np.ones(problem.num_blade_segments)
        width[0] = width[0] / 2.0
        width[-1] = width[-1] / 2.0
        outputs['width'] = width

        # Calculate an array describing the x, y, z position of each actuator node
        # Note: The basic blade is oriented along the +y-axis
        outputs['blade_pos_base'] = np.vstack(
            (
                np.zeros(problem.num_blade_segments),
                rdim,
                np.zeros(problem.num_blade_segments),
            )
        )

        # Calculate the blade velocity
        angular_velocity = 2.0 * np.pi * problem.rpm / 60.0
        tip_speed = angular_velocity * blade_length

        # Specify the velocity vector at each actuator node
        # Note: A blade with span oriented along the +y-axis moves in the +z direction
        outputs['blade_vel_base'] = np.vstack(
            (
                np.zeros(problem.num_blade_segments),
                np.zeros(problem.num_blade_segments),
                np.linspace(0.0, tip_speed, self.problem.num_blade_segments),
            )
        )

        outputs['theta_vec'] = np.linspace(0.0, 2.0 * np.pi, num_blades, endpoint=False)


class ComputeRotationMatrices(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        self.simTime_id = self.options['simTime_id']
        self.dt = self.options['dt']
        
        self.add_input('yaw', val=0.)
        self.add_input('theta', val=0.)
        self.add_output('Rx', shape=(3, 3))
        self.add_output('Rz', shape=(3, 3))
        
        self.declare_partials('Rx', 'theta')
        self.declare_partials('Rz', 'yaw')
        
        
    def compute(self, inputs, outputs):
        problem = self.problem
        simTime_id = self.simTime_id
        dt = self.dt
            
        simTime = problem.simTime_list[simTime_id]
        
        # Calculate the blade position based on current simTime and turbine RPM
        period = 60.0 / problem.rpm
        theta_offset = (simTime + 0.5 * dt) / period * 2.0 * np.pi
        
        theta = inputs['theta'] + theta_offset

        # Generate a rotation matrix for this turbine blade
        outputs['Rx'] = rot_x(float(theta))
        outputs['Rz'] = rot_z(float(inputs['yaw']))
        
    def compute_partials(self, inputs, partials):
        problem = self.problem
        simTime_id = self.simTime_id
        dt = self.dt
            
        simTime = problem.simTime_list[simTime_id]
        
        # Calculate the blade position based on current simTime and turbine RPM
        period = 60.0 / problem.rpm
        theta_offset = (simTime + 0.5 * dt) / period * 2.0 * np.pi
        
        theta = inputs['theta'] + theta_offset
        
        dRx = np.array(
            [
                [0., 0, 0],
                [0, -np.sin(theta), -np.cos(theta)],
                [0, np.cos(theta), -np.sin(theta)],
            ]
        )
        partials['Rx', 'theta'] = dRx
        
        yaw = inputs['yaw']
        dRz = np.array(
            [
                [-np.sin(yaw), -np.cos(yaw), 0],
                [np.cos(yaw), -np.sin(yaw), 0],
                [0, 0, 0.],
            ]
        )
        partials['Rz', 'yaw'] = dRz
        

class ComputeBladeVel(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        self.simTime_id = self.options['simTime_id']
        self.dt = self.options['dt']
        self.turb_i = self.options['turb_i']
        
        self.add_input('Rx', shape=(3, 3))
        self.add_input('Rz', shape=(3, 3))
        self.add_input('blade_vel_base', shape=(3, self.problem.num_blade_segments))
        self.add_output('blade_vel', shape=(3, self.problem.num_blade_segments))
        self.add_output('blade_unit_vec', shape=(3, 3))
        
        self.declare_partials('blade_unit_vec', 'Rx')
        self.declare_partials('blade_unit_vec', 'Rz')
        self.declare_partials('blade_vel', 'Rx')
        self.declare_partials('blade_vel', 'Rz')
        self.declare_partials('blade_vel', 'blade_vel_base')
        
        
    def compute(self, inputs, outputs):
        Rx = inputs['Rx']
        Rz = inputs['Rz']
        blade_vel_base = inputs['blade_vel_base']
            
        # Rotate the blade velocity in the global x, y, z, coordinate system
        # Note: blade_vel_base is negative since we seek the velocity of the fluid relative to a stationary blade
        # and blade_vel_base is defined based on the movement of the blade
        blade_vel = np.dot(Rz, np.dot(Rx, -blade_vel_base))
        
        # Create unit vectors aligned with blade geometry
        # blade_unit_vec_base[:, 0] = points along rotor shaft
        # blade_unit_vec_base[:, 1] = points along blade span axis
        # blade_unit_vec_base[:, 2] = points tangential to blade span axis (generates a torque about rotor shaft)
        blade_unit_vec_base = np.eye((3))

        # Rotate the blade unit vectors to be pointing in the rotated positions
        blade_unit_vec = np.dot(Rz, np.dot(Rx, blade_unit_vec_base))
        
        outputs['blade_vel'] = blade_vel
        outputs['blade_unit_vec'] = blade_unit_vec
        
    def compute_partials(self, inputs, partials):
        Rx = inputs['Rx']
        Rz = inputs['Rz']
        blade_vel_base = inputs['blade_vel_base']
        num_blade_segments = self.problem.num_blade_segments
            
        partials['blade_unit_vec', 'Rx'] = np.einsum('ik, jl', Rz, np.eye((3)))
        partials['blade_unit_vec', 'Rz'] = np.einsum('ik, jl', np.eye((3)), (Rx).dot(np.eye((3))).T)
        
        
        T_0 = (Rz).dot(Rx)
        partials['blade_vel', 'blade_vel_base'] = -np.einsum('ik, jl', T_0, np.eye(num_blade_segments))
        
        partials['blade_vel', 'Rz'] = -np.einsum('ik, jl', np.eye(3), (Rx).dot(blade_vel_base).T)
        partials['blade_vel', 'Rx'] = -np.einsum('ik, jl', Rz, blade_vel_base.T)


class ComputeURel(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        
        self.add_input('blade_vel', shape=(3, self.problem.num_blade_segments))
        self.add_input('blade_unit_vec', shape=(3, 3))
        self.add_input('u_local', shape=3)
        self.add_output('u_rel', shape=(3, self.problem.num_blade_segments))
        
        arange = np.arange(3*self.problem.num_blade_segments)
        self.declare_partials('u_rel', 'blade_vel', rows=arange, cols=arange, val=1.)
        
        rows = np.arange(3*self.problem.num_blade_segments)
        cols = np.repeat([0, 1, 2], self.problem.num_blade_segments)
        self.declare_partials('u_rel', 'u_local', rows=rows, cols=cols, val=1.)
        self.declare_partials('u_rel', 'blade_unit_vec', method='cs')
        
    def compute(self, inputs, outputs):
        problem = self.problem
        blade_unit_vec = inputs['blade_unit_vec']
        blade_vel = inputs['blade_vel']
        
        # Initialize space to hold the fluid velocity at each actuator node
        u_fluid = np.zeros((3, problem.num_blade_segments))

        # Generate the fluid velocity from the actual node locations in the flow
        for k in range(problem.num_blade_segments):
        
            u_fluid[:, k] = inputs['u_local'] - (
                np.dot(u_fluid[:, k], blade_unit_vec[:, 1]) * blade_unit_vec[:, 1]
            )
            
        # Form the total relative velocity vector (including velocity from rotating blade)
        outputs['u_rel'] = u_fluid + blade_vel
        
        
class ComputeUUnit(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        
    def setup(self):
        self.problem = self.options['problem']
        num_blade_segments = self.problem.num_blade_segments
        
        self.add_input('u_rel', shape=(3, num_blade_segments))
        self.add_output('u_unit_vec', shape=(3, num_blade_segments))
        self.add_output('u_rel_mag', shape=num_blade_segments)
        
        rows = np.repeat(np.arange(num_blade_segments), 3)
        cols = []
        for i in range(num_blade_segments):
            cols.append(np.arange(3)*num_blade_segments + i)
        cols = np.array(cols).flatten()
        
        self.declare_partials('u_rel_mag', 'u_rel', rows=rows, cols=cols)
        self.declare_partials('u_unit_vec', 'u_rel')
        
        
    def compute(self, inputs, outputs):
        u_rel_mag = np.linalg.norm(inputs['u_rel'], axis=0)
        u_rel_mag[u_rel_mag < 1e-6] = 1e-6
        outputs['u_rel_mag'] = u_rel_mag
        outputs['u_unit_vec'] = inputs['u_rel'] / u_rel_mag
        
    def compute_partials(self, inputs, partials):
        u_rel = inputs['u_rel']
        t_0 = np.linalg.norm(inputs['u_rel'], axis=0)
        partials['u_rel_mag', 'u_rel'] = ((1 / t_0) * u_rel).flatten(order='F')
        

        

class ComputeCLCD(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        self.simTime_id = self.options['simTime_id']
        self.dt = self.options['dt']
        self.turb_i = self.options['turb_i']
        
        self.add_input('u_rel', shape=(3, self.problem.num_blade_segments))
        self.add_input('blade_unit_vec', shape=(3, 3))
        self.add_input('rdim', shape=self.problem.num_blade_segments)
        self.add_output('cl', shape=self.problem.num_blade_segments)
        self.add_output('cd', shape=self.problem.num_blade_segments)
        self.add_output('tip_loss', shape=self.problem.num_blade_segments)
        
    def compute(self, inputs, outputs):
        u_rel = inputs['u_rel']
        blade_unit_vec = inputs['blade_unit_vec']
        rdim = inputs['rdim']
        twist = np.array(self.problem.mtwist[self.turb_i], dtype=float)
        c = np.array(self.problem.mchord[self.turb_i], dtype=float)
        
        build_lift_and_drag(self.problem, rdim)
        cl, cd, tip_loss = call_lift_and_drag(
            self.problem, u_rel, blade_unit_vec, rdim, twist, c
        )
        
        outputs['cl'] = cl
        outputs['cd'] = cd
        outputs['tip_loss'] = tip_loss
        
        
class ComputeLiftDrag(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        self.simTime_id = self.options['simTime_id']
        self.dt = self.options['dt']
        self.turb_i = self.options['turb_i']
        
        self.add_input('cl', shape=self.problem.num_blade_segments)
        self.add_input('cd', shape=self.problem.num_blade_segments)
        self.add_input('tip_loss', shape=self.problem.num_blade_segments)
        self.add_input('width', shape=self.problem.num_blade_segments)
        self.add_input('u_rel_mag', shape=self.problem.num_blade_segments)
        self.add_output('lift', shape=self.problem.num_blade_segments)
        self.add_output('drag', shape=self.problem.num_blade_segments)
        
        arange = np.arange(self.problem.num_blade_segments)
        
        self.declare_partials('lift', 'cl', rows=arange, cols=arange)
        self.declare_partials('drag', 'cd', rows=arange, cols=arange)
        self.declare_partials('lift', 'tip_loss', rows=arange, cols=arange)
        self.declare_partials('drag', 'tip_loss', rows=arange, cols=arange)
        self.declare_partials('lift', 'width', rows=arange, cols=arange)
        self.declare_partials('drag', 'width', rows=arange, cols=arange)
        self.declare_partials('lift', 'u_rel_mag', rows=arange, cols=arange)
        self.declare_partials('drag', 'u_rel_mag', rows=arange, cols=arange)
        
    def compute(self, inputs, outputs):
        # Set the density
        rho = 1.0
        c = np.array(self.problem.mchord[self.turb_i], dtype=float)
        
        cl = inputs['cl']
        cd = inputs['cd']
        tip_loss = inputs['tip_loss']
        width = inputs['width']
        u_rel_mag = inputs['u_rel_mag']
        
        # Calculate the lift and drag forces using the relative velocity magnitude
        outputs['lift'] = tip_loss * (0.5 * cl * rho * c * width * u_rel_mag ** 2)
        outputs['drag'] = tip_loss * (0.5 * cd * rho * c * width * u_rel_mag ** 2)
        
    def compute_partials(self, inputs, partials):
        rho = 1.0
        c = np.array(self.problem.mchord[self.turb_i], dtype=float)
        cl = inputs['cl']
        cd = inputs['cd']
        tip_loss = inputs['tip_loss']
        width = inputs['width']
        u_rel_mag = inputs['u_rel_mag']

        partials['lift', 'tip_loss'] = 0.5 * cl * rho * c * width * u_rel_mag ** 2
        partials['drag', 'tip_loss'] = 0.5 * cd * rho * c * width * u_rel_mag ** 2

        partials['lift', 'cl'] = 0.5 * tip_loss * rho * c * width * u_rel_mag ** 2
        partials['drag', 'cd'] = 0.5 * tip_loss * rho * c * width * u_rel_mag ** 2

        partials['lift', 'width'] = tip_loss * (0.5 * cl * rho * c * u_rel_mag ** 2)
        partials['drag', 'width'] = tip_loss * (0.5 * cd * rho * c * u_rel_mag ** 2)

        partials['lift', 'u_rel_mag'] = tip_loss * cl * rho * c * width * u_rel_mag
        partials['drag', 'u_rel_mag'] = tip_loss * cd * rho * c * width * u_rel_mag
        

class ComputeNodalLiftDrag(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        self.simTime_id = self.options['simTime_id']
        self.dt = self.options['dt']
        self.turb_i = self.options['turb_i']
        
        self.add_input('blade_pos_base', shape=(3, self.problem.num_blade_segments))
        self.add_input('Rx', shape=(3, 3))
        self.add_input('Rz', shape=(3, 3))
        self.add_input('lift', shape=self.problem.num_blade_segments)
        self.add_input('drag', shape=self.problem.num_blade_segments)
        self.add_output('nodal_lift', shape=(125, self.problem.num_blade_segments))
        self.add_output('nodal_drag', shape=(125, self.problem.num_blade_segments))
        
    def compute(self, inputs, outputs):
        dfd = None
        problem = self.problem
        simTime_id = self.simTime_id
        dt = self.dt
        turb_i = self.turb_i
            
        simTime = problem.simTime_list[simTime_id]
        ndim = problem.dom.dim
        eps = problem.gaussian_width
        
        Rx = inputs['Rx']
        Rz = inputs['Rz']
        blade_pos_base = inputs['blade_pos_base']
        
        # Rotate the entire [x; y; z] matrix using this matrix, then shift to the hub location
        blade_pos = np.dot(Rz, np.dot(Rx, blade_pos_base))
        blade_pos[0, :] += problem.farm.x[turb_i]
        blade_pos[1, :] += problem.farm.y[turb_i]
        blade_pos[2, :] += problem.farm.z[turb_i]

        # Tile the blade coordinates for every mesh point, [numGridPts*ndim x problem.num_blade_segments]
        blade_pos_full = np.tile(blade_pos, (np.shape(problem.coords)[0], 1))

        # Subtract and square to get the dx^2 values in the x, y, and z directions
        dx_full = (problem.coordsLinear - blade_pos_full) ** 2

        # Add together to get |x^2 + y^2 + z^2|^2
        dist2 = dx_full[0::ndim] + dx_full[1::ndim] + dx_full[2::ndim]

        # Calculate the force magnitude at every mesh point due to every node [numGridPts x NumActuators]
        outputs['nodal_lift'] = inputs['lift'] * np.exp(-dist2 / eps ** 2) / (eps ** 3 * np.pi ** 1.5)
        outputs['nodal_drag'] = inputs['drag'] * np.exp(-dist2 / eps ** 2) / (eps ** 3 * np.pi ** 1.5)
    

class ComputeLiftDragForces(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('simTime_id', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('turb_i', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        self.simTime_id = self.options['simTime_id']
        self.dt = self.options['dt']
        self.turb_i = self.options['turb_i']
        ndim = self.problem.dom.dim
        
        self.add_input('u_unit_vec', shape=(3, self.problem.num_blade_segments))
        self.add_input('blade_unit_vec', shape=(3, 3))
        self.add_input('nodal_lift', shape=(125, self.problem.num_blade_segments))
        self.add_input('nodal_drag', shape=(125, self.problem.num_blade_segments))
        self.add_output('lift_force', shape=(125, ndim))
        self.add_output('drag_force', shape=(125, ndim))
        
    def compute(self, inputs, outputs):
        problem = self.problem
        u_unit_vec = inputs['u_unit_vec']
        blade_unit_vec = inputs['blade_unit_vec']
        nodal_lift = inputs['nodal_lift']
        nodal_drag = inputs['nodal_drag']
        
        outputs['lift_force'][:] = 0.
        outputs['drag_force'][:] = 0.
        
        for k in range(problem.num_blade_segments):
            # The drag unit simply points opposite the relative velocity unit vector
            drag_unit_vec = -np.copy(u_unit_vec[:, k])

            # The lift is normal to the plane generated by the blade and relative velocity
            lift_unit_vec = np.cross(drag_unit_vec, blade_unit_vec[:, 1])

            # All force magnitudes get multiplied by the correctly-oriented unit vector
            vector_nodal_lift = np.outer(nodal_lift[:, k], lift_unit_vec)
            vector_nodal_drag = np.outer(nodal_drag[:, k], drag_unit_vec)

            outputs['lift_force'] += vector_nodal_lift
            outputs['drag_force'] += vector_nodal_drag
    

class ComputeTurbineForce(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('problem', types=object)
        self.options.declare('num_blades', types=int)
        
    def setup(self):
        self.problem = self.options['problem']
        ndim = self.problem.dom.dim
        
        arange = np.arange(375)
        
        for i_blade in range(self.options['num_blades']):
            self.add_input(f'lift_force_{i_blade}', shape=(125, ndim))
            self.add_input(f'drag_force_{i_blade}', shape=(125, ndim))
            
            self.declare_partials('turbine_forces', f'lift_force_{i_blade}', rows=arange, cols=arange, val=1.)
            self.declare_partials('turbine_forces', f'drag_force_{i_blade}', rows=arange, cols=arange, val=1.)
            
        self.add_output('turbine_forces', shape=375)
        
    def compute(self, inputs, outputs):
        problem = self.problem
        ndim = self.problem.dom.dim
        turbine_force = np.zeros(inputs['drag_force_0'].shape)
        
        for i_blade in range(self.options['num_blades']):
            # The total turbine force is the sum of lift and drag effects
            turbine_force += inputs[f'drag_force_{i_blade}'] + inputs[f'lift_force_{i_blade}']

        tf_vec = np.zeros(np.size(self.problem.coords))
        
        # Riffle-shuffle the x-, y-, and z-column force components
        for k in range(ndim):
            tf_vec[k::ndim] = turbine_force[:, k]

        # Remove near-zero values
        tf_vec[np.abs(tf_vec) < 1e-12] = 0.0

        problem.first_call_to_alm = False

        outputs['turbine_forces'] = tf_vec