#!/usr/bin/env python3
"""Create the MuJoCo Warp training scene from the controller/viewer MJCF.

MuJoCo Warp does not support MuJoCo body plugins. The controller/viewer scene
uses ``mujoco.elasticity.cable`` plugin instances on the cable bodies, so this
script creates a separate Warp scene for headless training. It keeps bodies,
joints, actuators, collision geoms, welds, and task geometry, but removes:

* top-level ``<extension>`` declarations
* body-level ``<plugin .../>`` elements
* visual-only geoms and visual assets
* cameras, lights, and sensors

The resulting files are:

* ``mjcf/aic_robot_warp.xml``
* ``mjcf/aic_world_warp.xml``
* ``mjcf/scene_warp.xml``

Run from a new ``aic_eval`` distrobox terminal:

cd /home/rmalhan/Software/ws_aic/src/aic
pixi shell
export PYTHONNOUSERSITE=1
python3 aic_utils/aic_mujoco/scripts/prepare_warp_scene.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from xml.etree import ElementTree as ET

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MJCF_DIR = PACKAGE_ROOT / "mjcf"


def remove_tags(root: ET.Element, tags: set[str]) -> None:
    """Remove XML elements with matching tags while preserving the rest."""

    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in tags:
                parent.remove(child)


def strip_visual_geoms(root: ET.Element) -> None:
    """Remove geoms that are visual-only in MJCF."""

    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag != "geom":
                continue
            contype = child.attrib.get("contype")
            conaffinity = child.attrib.get("conaffinity")
            group = child.attrib.get("group")
            name = child.attrib.get("name", "")
            if contype == "0" and conaffinity == "0":
                parent.remove(child)
            elif group in {"0", "1", "2"} and "collision" not in name:
                parent.remove(child)


def strip_visual_assets(root: ET.Element) -> None:
    """Keep only mesh assets still referenced by remaining collision geoms."""

    used_meshes = {
        elem.attrib["mesh"]
        for elem in root.iter()
        if elem.tag == "geom" and "mesh" in elem.attrib
    }
    remove_tags(root, {"visual", "sensor"})
    for asset in list(root.findall("asset")):
        for child in list(asset):
            if child.tag != "mesh" or child.attrib.get("name") not in used_meshes:
                asset.remove(child)
        if len(list(asset)) == 0:
            root.remove(asset)


def find_body(root: ET.Element, name: str) -> ET.Element:
    """Find a body by exact MuJoCo body name."""

    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    raise RuntimeError(f"Body not found: {name}")


def remove_body(root: ET.Element, name: str) -> None:
    """Remove a body subtree by exact MuJoCo body name."""

    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.attrib.get("name") == name:
                parent.remove(child)
                return
    raise RuntimeError(f"Body not found for removal: {name}")


def remove_cable_equalities(root: ET.Element) -> None:
    """Remove cable/gripper welds that are replaced by direct parenting."""

    for equality in list(root.findall("equality")):
        for child in list(equality):
            if child.attrib.get("body2") == "lc_plug_link":
                equality.remove(child)
        if len(list(equality)) == 0:
            root.remove(equality)


def remove_cable_contacts(root: ET.Element) -> None:
    """Remove contact exclusions that refer to the deleted cable chain."""

    for contact in list(root.findall("contact")):
        for child in list(contact):
            if "cable" in child.attrib.get("body1", "") or "cable" in child.attrib.get(
                "body2", ""
            ):
                contact.remove(child)
            elif child.attrib.get("body1", "").startswith("link_") or child.attrib.get(
                "body2", ""
            ).startswith("link_"):
                contact.remove(child)
        if len(list(contact)) == 0:
            root.remove(contact)


def attach_plug_to_robot(robot_root: ET.Element, world_root: ET.Element) -> None:
    """Attach the LC/SFP plug subtree directly under ``ati/tool_link``."""

    plug = copy.deepcopy(find_body(world_root, "lc_plug_link"))
    for elem in plug.iter():
        if elem.attrib.get("class") == "world_default":
            elem.attrib["class"] = "robot_unused"
    plug.attrib["pos"] = "-0.000711 0.001759 0.168213"
    plug.attrib["quat"] = "0.577301 0.816105 -0.021418 -0.015395"
    find_body(robot_root, "ati/tool_link").append(plug)
    remove_body(world_root, "cable_end_0")
    remove_cable_equalities(world_root)
    remove_cable_contacts(world_root)


def main() -> int:
    robot_in = MJCF_DIR / "aic_robot.xml"
    robot_out = MJCF_DIR / "aic_robot_warp.xml"
    world_in = MJCF_DIR / "aic_world.xml"
    world_out = MJCF_DIR / "aic_world_warp.xml"
    scene_out = MJCF_DIR / "scene_warp.xml"

    robot_tree = ET.parse(robot_in)
    robot_root = robot_tree.getroot()
    source_world_root = ET.parse(world_in).getroot()
    attach_plug_to_robot(robot_root, source_world_root)
    remove_tags(robot_root, {"camera", "light"})
    strip_visual_geoms(robot_root)
    strip_visual_assets(robot_root)
    robot_tree.write(robot_out, encoding="unicode", xml_declaration=False)

    tree = ET.ElementTree(source_world_root)
    world_root = source_world_root
    remove_tags(world_root, {"extension", "plugin", "camera", "light"})
    strip_visual_geoms(world_root)
    strip_visual_assets(world_root)
    tree.write(world_out, encoding="unicode", xml_declaration=False)

    scene_out.write_text(
        '<mujoco model="SceneWarp">\n'
        '  <option integrator="implicitfast" timestep="0.002" solver="Newton" '
        'iterations="200" tolerance="1e-10"/>\n'
        '  <include file="aic_robot_warp.xml"/>\n'
        '  <include file="aic_world_warp.xml"/>\n'
        "</mujoco>\n",
        encoding="utf-8",
    )

    print(f"Wrote {robot_out}")
    print(f"Wrote {world_out}")
    print(f"Wrote {scene_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
