import argparse

from isaaclab.app import AppLauncher

# Define collision approximation choices (must be defined before parser)
_valid_collision_approx = [
    "convexDecomposition",
    "convexHull",
    "triangleMesh",
    "meshSimplification",
    "sdf",
    "boundingCube",
    "boundingSphere",
    "none",
]

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("--input", type=str, help="The path to the input mesh file.")
parser.add_argument("--output", type=str, help="The path to store the USD file.")
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=_valid_collision_approx,
    help="The method used for approximating the collision mesh. Set to 'none' to disable collision mesh generation.",
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",    
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import glob
from tqdm import tqdm
import random

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict

collision_approximation_map = {
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "triangleMesh": schemas_cfg.TriangleMeshPropertiesCfg,
    "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    "sdf": schemas_cfg.SDFMeshPropertiesCfg,
    "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
    "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
    "none": None,
}


def main():
    # check valid file path
    mesh_dir = args_cli.input

    OBJ_PATH = os.path.join(mesh_dir, '*.obj')
    obj_list = glob.glob(OBJ_PATH, recursive=True)
    print(obj_list)

    dest_dir = args_cli.output

    for mesh_path in tqdm(obj_list, total=len(obj_list)):
        mesh_path = os.path.abspath(mesh_path)

        dest_path = os.path.abspath(os.path.join(dest_dir, (os.path.basename(mesh_path)).replace('.obj', '.usd')))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        rand_mass = random.uniform(0.05, 0.15)
        mass_props = schemas_cfg.MassPropertiesCfg(mass=rand_mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()

        # Collision properties
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=args_cli.collision_approximation != "none")

        # Create Mesh converter config
        cfg_class = collision_approximation_map.get(args_cli.collision_approximation)
        if cfg_class is None and args_cli.collision_approximation != "none":
            valid_keys = ", ".join(sorted(collision_approximation_map.keys()))
            raise ValueError(
                f"Invalid collision approximation type '{args_cli.collision_approximation}'. "
                f"Valid options are: {valid_keys}."
            )
        
        collision_cfg = cfg_class() if cfg_class is not None else None

        mesh_converter_cfg = MeshConverterCfg(
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            asset_path=mesh_path,
            force_usd_conversion=True,
            usd_dir=os.path.dirname(dest_path),
            usd_file_name=os.path.basename(dest_path),
            make_instanceable=args_cli.make_instanceable,
            mesh_collision_props=collision_cfg,
        )

        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input Mesh file: {mesh_path}")
        print("Mesh importer config:")
        print_dict(mesh_converter_cfg.to_dict(), nesting=0)
        print("-" * 80)
        print("-" * 80)

        # Create Mesh converter and import the file
        mesh_converter = MeshConverter(mesh_converter_cfg)
        # print output
        print("Mesh importer output:")
        print(f"Generated USD file: {mesh_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

        # Determine if there is a GUI to update:
        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # read flag for whether a local GUI is enabled
        local_gui = carb_settings_iface.get("/app/window/enabled")
        # read flag for whether livestreaming GUI is enabled
        livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

        # Simulate scene (if not headless)
        if local_gui or livestream_gui:
            # Open the stage with USD
            stage_utils.open_stage(mesh_converter.usd_path)
            # Reinitialize the simulation
            app = omni.kit.app.get_app_interface()
            # Run simulation
            with contextlib.suppress(KeyboardInterrupt):
                while app.is_running():
                    # perform step
                    app.update()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
