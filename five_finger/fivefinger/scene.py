"""Build a derived workcell while preserving the downloaded model unchanged."""
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
from .paths import VENDOR, output_dir


def build_scene():
    source = VENDOR / "dexrobot_mujoco/models/dexhand021_right_jaka_zu7.xml"
    root = ET.parse(source).getroot()
    root.set("model", "five_finger_reference_workcell")
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", (source.parent / mesh.get("file")).resolve().as_posix())
    root.find("option").set("timestep", "0.002")
    world = root.find("worldbody")
    world.find("body[@name='r_a_base']").set("pos", "-0.35 0 0")
    ET.SubElement(world, "light", pos="0 -0.8 2", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "light", pos="-1 1 1.5", dir="1 -1 -1", diffuse="0.4 0.4 0.4")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1", pos="0 0 -0.76", rgba="0.16 0.18 0.22 1")
    ET.SubElement(world, "geom", name="table", type="box", size="0.65 0.45 0.025", pos="0.15 0 -0.025", rgba="0.45 0.42 0.37 1")
    ET.SubElement(world, "geom", name="paper", type="box", size="0.1485 0.21 0.001", pos="0.25 0 0.001", rgba="0.96 0.93 0.86 1", friction="0.7 0.01 0.001")
    # A generic test mark, not an official dragon image or an automatic scorer.
    ET.SubElement(world, "geom", name="dot_target", type="cylinder", size="0.007 0.0002", pos="0.25 0 0.0022", rgba="0.25 0.5 0.6 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="inkstone", type="box", size="0.055 0.045 0.009", pos="0 0.28 0.009", rgba="0.08 0.08 0.09 1")
    ET.SubElement(world, "geom", name="ink_surface_visual", type="box", size="0.04 0.03 0.0002", pos="0 0.28 0.0183", rgba="0.01 0.015 0.02 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="rack", type="box", size="0.05 0.025 0.025", pos="0.05 -0.29 0.025", rgba="0.24 0.16 0.10 1")
    brush = ET.SubElement(world, "body", name="brush", pos="0.05 -0.29 0.10", quat="0.70710678 0 0.70710678 0")
    ET.SubElement(brush, "freejoint", name="brush_free")
    ET.SubElement(brush, "geom", name="brush_barrel", type="capsule", size="0.004 0.085", mass="0.018", rgba="0.36 0.18 0.06 1", friction="0.8 0.01 0.001")
    ET.SubElement(brush, "geom", name="brush_tip_approximation", type="capsule", size="0.003 0.009", pos="0 0 -0.095", mass="0.002", rgba="0.03 0.03 0.03 1", solref="0.03 1", friction="0.6 0.01 0.001")
    ET.SubElement(brush, "site", name="brush_tip", pos="0 0 -0.107", size="0.001", rgba="1 0.3 0 1")
    palm = root.find(".//body[@name='right_hand_base']")
    ET.SubElement(palm, "site", name="palm_control", pos="0 0 0.1", size="0.006", rgba="0 0.7 0.9 0.6")
    ET.SubElement(world, "camera", name="overhead", pos="0.15 0 1.8", quat="1 0 0 0", fovy="45")
    # Lower finger servo gains for development; these are NOT hardware force limits.
    for act in root.findall("./actuator/position"):
        if "_f_" in act.get("joint", ""):
            act.set("kp", "12"); act.set("kv", "0.4")
            act.set("forcerange", "-0.4 0.4")
    out = output_dir() / "workcell.xml"
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out


def load_scene():
    # MuJoCo's Windows file API may reject non-ASCII paths. Python reads the
    # original files, then MuJoCo receives an in-memory XML and mesh filesystem.
    root = ET.parse(build_scene()).getroot()
    assets = {}
    for index, mesh in enumerate(root.findall("./asset/mesh")):
        source = Path(mesh.get("file"))
        key = f"mesh_{index}{source.suffix}"
        assets[key] = source.read_bytes()
        mesh.set("file", key)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)
