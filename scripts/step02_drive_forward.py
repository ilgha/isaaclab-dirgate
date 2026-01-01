from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import Usd, UsdPhysics

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation

NOVA_CARTER_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)
ROBOT_PRIM_PATH = "/Nova_Carter"
WHEEL_SPEED = 6.0  # rad/s


def find_articulation_root(parent_prim_path: str) -> str:
    stage = omni.usd.get_context().get_stage()
    parent = stage.GetPrimAtPath(parent_prim_path)
    if not parent.IsValid():
        return ""
    for prim in Usd.PrimRange(parent):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    return ""


def main():
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    print(f"[STEP02] Spawning Nova Carter -> {ROBOT_PRIM_PATH}")
    add_reference_to_stage(NOVA_CARTER_USD, ROBOT_PRIM_PATH)
    for _ in range(3):
        simulation_app.update()

    art_root = find_articulation_root(ROBOT_PRIM_PATH)
    if not art_root:
        raise RuntimeError("No articulation root found under /Nova_Carter.")
    print(f"[STEP02] Articulation root: {art_root}")

    robot = SingleArticulation(prim_path=art_root, name="nova_carter")
    world.scene.add(robot)

    world.reset()
    print("[STEP02] World reset complete")

    dof_names = list(robot.dof_names)
    left_idx = dof_names.index("joint_wheel_left")
    right_idx = dof_names.index("joint_wheel_right")
    print(f"[STEP02] wheel indices: left={left_idx}, right={right_idx}")

    vel = [0.0] * len(dof_names)
    vel[left_idx] = WHEEL_SPEED
    vel[right_idx] = WHEEL_SPEED

    print("[STEP02] Driving forward...")
    while simulation_app.is_running():
        robot.set_joint_velocities(vel)
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
