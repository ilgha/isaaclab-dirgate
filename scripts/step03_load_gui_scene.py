from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom, Gf

# Update if you rename the file
USD_PATH = "usd/dirgate_arena_base.usd"

# Prim paths (from your GUI naming)
PRIMS = {
    "arena": "/World/arena",
    "gate": "/World/gate",
    "dirlight": "/World/arena/dirlight",
    "robot": "/World/Nova_Carter",
    # Optional anchors (recommended to create in GUI)
    "robot_spawn": "/World/RobotSpawn",
    "gate_entrance": "/World/GateEntrance",
}


def get_world_pose(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    cache = UsdGeom.XformCache()
    xf: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)

    t = xf.ExtractTranslation()
    r = xf.ExtractRotationMatrix()  # 3x3

    # yaw (around Z) for quick debugging
    # yaw = atan2(r10, r00) assuming Z-up
    yaw = float(__import__("math").atan2(r[1][0], r[0][0]))

    return {
        "translation": (float(t[0]), float(t[1]), float(t[2])),
        "yaw_rad": yaw,
    }


def main():
    ctx = omni.usd.get_context()
    print(f"[STEP03] Opening stage: {USD_PATH}")
    ctx.open_stage(USD_PATH)

    # Let Kit finish loading
    for _ in range(5):
        simulation_app.update()

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Stage failed to load.")

    print("\n[STEP03] Prim contract check:")
    for k, p in PRIMS.items():
        prim = stage.GetPrimAtPath(p)
        print(f"  {k:12s} {p:28s} exists={prim.IsValid()} type={prim.GetTypeName() if prim.IsValid() else '-'}")

    print("\n[STEP03] World poses (if prim exists):")
    for k, p in PRIMS.items():
        pose = get_world_pose(stage, p)
        if pose is None:
            continue
        print(f"  {k:12s} pos={pose['translation']} yaw(rad)={pose['yaw_rad']:.3f}")

    # Keep sim open
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
