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

    print(f"[STEP01] Spawning Nova Carter -> {ROBOT_PRIM_PATH}")
    add_reference_to_stage(NOVA_CARTER_USD, ROBOT_PRIM_PATH)
    for _ in range(3):
        simulation_app.update()

    art_root = find_articulation_root(ROBOT_PRIM_PATH)
    if not art_root:
        raise RuntimeError("No articulation root found under /Nova_Carter.")
    print(f"[STEP01] Articulation root: {art_root}")

    robot = SingleArticulation(prim_path=art_root, name="nova_carter")
    world.scene.add(robot)

    world.reset()
    print("[STEP01] World reset complete")

    dof_names = list(robot.dof_names)
    print("\n=== Nova Carter DOFs ===")
    print("num_dofs:", len(dof_names))
    for i, n in enumerate(dof_names):
        print(f"{i:02d}  {n}")
    print("========================\n")

    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
