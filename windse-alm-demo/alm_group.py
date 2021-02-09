from fenics import *
import numpy as np
import scipy.interpolate as interp
import openmdao.api as om
from alm_components import (
    Preprocess,
    ComputeRotationMatrices,
    ComputeBladeVel,
    ComputeURel,
    ComputeUUnit,
    ComputeCLCD,
    ComputeLiftDrag,
    ComputeNodalLiftDrag,
    ComputeLiftDragForces,
    ComputeTurbineForce,
)


class ALMBlade(om.Group):
    def initialize(self):
        self.options.declare("problem", types=object)
        self.options.declare("simTime_id", types=int)
        self.options.declare("dt", types=float)
        self.options.declare("turb_i", types=int)

    def setup(self):
        self.problem = self.options["problem"]
        self.simTime_id = self.options["simTime_id"]
        self.dt = self.options["dt"]
        self.turb_i = self.options["turb_i"]

        self.add_subsystem(
            "ComputeRotationMatrices",
            ComputeRotationMatrices(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "ComputeBladeVel",
            ComputeBladeVel(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "ComputeURel",
            ComputeURel(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "ComputeUUnit",
            ComputeUUnit(
                problem=self.problem,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "ComputeCLCD",
            ComputeCLCD(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "ComputeLiftDrag",
            ComputeLiftDrag(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "ComputeNodalLiftDrag",
            ComputeNodalLiftDrag(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "ComputeLiftDragForces",
            ComputeLiftDragForces(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )


class ALMGroup(om.Group):
    def initialize(self):
        self.options.declare("problem", types=object)
        self.options.declare("simTime_id", types=int)
        self.options.declare("dt", types=float)
        self.options.declare("turb_i", types=int)
        self.options.declare("num_blades", types=int)

    def setup(self):
        self.problem = self.options["problem"]
        self.simTime_id = self.options["simTime_id"]
        self.dt = self.options["dt"]
        self.turb_i = self.options["turb_i"]

        self.add_subsystem(
            "Preprocess",
            Preprocess(
                problem=self.problem,
                simTime_id=self.simTime_id,
                dt=self.dt,
                turb_i=self.turb_i,
            ),
            promotes=["*"],
        )

        for i_blade in range(self.options["num_blades"]):
            self.add_subsystem(
                f"ALMBlade_{i_blade}",
                ALMBlade(
                    problem=self.problem,
                    simTime_id=self.simTime_id,
                    dt=self.dt,
                    turb_i=self.turb_i,
                ),
                promotes_inputs=[
                    "width",
                    "rdim",
                    "blade_pos_base",
                    "blade_vel_base",
                    "u_local",
                    "yaw",
                ],
            )

            self.connect("theta_vec", f"ALMBlade_{i_blade}.theta", src_indices=[i_blade])
            self.connect(f"ALMBlade_{i_blade}.lift_force", f"lift_force_{i_blade}")
            self.connect(f"ALMBlade_{i_blade}.drag_force", f"drag_force_{i_blade}")


        self.add_subsystem(
            "CombineTurbineForce",
            ComputeTurbineForce(
                problem=self.problem,
                num_blades=self.options['num_blades'],
            ),
            promotes=["*"],
        )
