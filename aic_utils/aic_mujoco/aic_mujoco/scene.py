"""Build the reduced AIC SFP/NIC MJCF consumed by MuJoCo-Warp."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from aic_mujoco.utils.mujoco_math import (
    compose_pose,
    normalized_quaternion,
    rotate_vector,
)


def required_element(root: ET.Element, query: str, description: str) -> ET.Element:
    """Return one required XML element.

    Args:
        root: XML subtree to search.
        query: ElementTree query relative to ``root``.
        description: Human-readable element description for errors.

    Returns:
        The matched XML element.

    Raises:
        ValueError: If the query does not match an element.
    """

    element = root.find(query)
    if element is None:
        raise ValueError(f"Source MJCF is missing {description}: {query}")
    return element


def named_element(root: ET.Element, tag: str, name: str) -> ET.Element:
    """Return the only XML element with a tag and name.

    Args:
        root: XML subtree to search recursively.
        tag: XML element tag.
        name: Required ``name`` attribute.

    Returns:
        The unique matching element.

    Raises:
        ValueError: If zero or multiple elements match.
    """

    matches = [element for element in root.iter(tag) if element.get("name") == name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one <{tag} name='{name}'>, found {len(matches)}"
        )
    return matches[0]


def parse_vector(text: str | None, length: int, description: str) -> list[float]:
    """Parse and validate a fixed-length MJCF numeric attribute.

    Args:
        text: Space-separated MJCF attribute text.
        length: Required number of values.
        description: Human-readable attribute description for errors.

    Returns:
        Parsed finite floating-point values.

    Raises:
        ValueError: If the attribute is missing, malformed, or non-finite.
    """

    if text is None:
        raise ValueError(f"Missing {description}")
    values = [float(value) for value in text.split()]
    if len(values) != length:
        raise ValueError(f"{description} must contain {length} numbers")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{description} must contain only finite numbers")
    return values


def format_vector(values: list[float] | np.ndarray) -> str:
    """Format a numeric vector as a stable MJCF attribute.

    Args:
        values: Numeric values to serialize.

    Returns:
        Space-separated values with stable precision.
    """

    return " ".join(f"{value:.16g}" for value in values)


def remove_descendant(parent: ET.Element, target: ET.Element) -> bool:
    """Remove an XML descendant while preserving the remaining subtree.

    Args:
        parent: Subtree root to search recursively.
        target: Exact element instance to remove.

    Returns:
        ``True`` when the element was found and removed.
    """

    for child in parent:
        if child is target:
            parent.remove(child)
            return True
        if remove_descendant(child, target):
            return True
    return False


def copy_defaults(robot: ET.Element, world: ET.Element) -> ET.Element:
    """Copy only default classes required by the reduced scene.

    Args:
        robot: Source robot MJCF root.
        world: Source world MJCF root.

    Returns:
        Reduced ``default`` XML element.

    Raises:
        ValueError: If a required class is absent.
    """

    output = ET.Element("default")
    permitted = {"robot_unused", "world_default", "nic_card_default"}
    for root in (robot, world):
        source = root.find("default")
        if source is None:
            continue
        for child in source:
            if child.get("class") in permitted:
                output.append(copy.deepcopy(child))
    found = {child.get("class") for child in output}
    if found != permitted:
        raise ValueError(f"Required default classes missing: {sorted(permitted - found)}")
    return output


def copy_assets(
    robot: ET.Element,
    world: ET.Element,
    selected: list[ET.Element],
    output_directory: Path,
) -> ET.Element:
    """Copy the transitive asset closure used by selected scene elements.

    Args:
        robot: Source robot MJCF root.
        world: Source world MJCF root.
        selected: Scene subtrees retained in the reduced model.
        output_directory: Directory that must contain referenced asset files.

    Returns:
        Reduced ``asset`` XML element.

    Raises:
        FileNotFoundError: If a referenced asset file is absent.
        ValueError: If asset names conflict or a reference is unresolved.
    """

    assets_by_name: dict[str, ET.Element] = {}
    for root in (robot, world):
        asset = root.find("asset")
        if asset is None:
            continue
        for item in asset:
            name = item.get("name")
            if not name:
                continue
            if name in assets_by_name:
                raise ValueError(f"Duplicate asset name across source MJCFs: {name}")
            assets_by_name[name] = item

    required: set[str] = set()
    for element in selected:
        for descendant in element.iter():
            for attribute in ("mesh", "material", "texture", "hfield", "skin"):
                name = descendant.get(attribute)
                if name:
                    required.add(name)

    copied: dict[str, ET.Element] = {}
    pending = list(required)
    while pending:
        name = pending.pop()
        if name in copied:
            continue
        source = assets_by_name.get(name)
        if source is None:
            raise ValueError(f"Selected MJCF geometry references missing asset: {name}")
        item = copy.deepcopy(source)
        copied[name] = item
        for attribute in ("material", "texture", "hfield"):
            dependency = item.get(attribute)
            if dependency:
                pending.append(dependency)

    output = ET.Element("asset")
    order = {"texture": 0, "material": 1, "mesh": 2, "hfield": 3, "skin": 4}
    for name, item in sorted(
        copied.items(), key=lambda pair: (order.get(pair[1].tag, 99), pair[0])
    ):
        filename = item.get("file")
        if filename and not (output_directory / filename).is_file():
            raise FileNotFoundError(f"Asset file for {name} does not exist: {filename}")
        output.append(item)
    return output


def validate_model(model: mujoco.MjModel, config: dict[str, Any]) -> None:
    """Validate the compiled reduced model against its strict contract.

    Args:
        model: Compiled reduced MuJoCo model.
        config: Strict merged runtime configuration.

    Raises:
        ValueError: If dimensions or required named elements differ.
    """

    names = config["scene"]["names"]
    expected = {
        "nq": 6,
        "nv": 6,
        "nu": 6,
        "ncam": 3,
        "nsensor": 2,
        "nsensordata": 6,
        "nmocap": 2,
    }
    for field, value in expected.items():
        actual = int(getattr(model, field))
        if actual != value:
            raise ValueError(f"Generated scene {field}={actual}; expected {value}")
    for joint in names["joints"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) < 0:
            raise ValueError(f"Generated scene is missing joint: {joint}")
    for actuator in names["actuators"]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) < 0:
            raise ValueError(f"Generated scene is missing actuator: {actuator}")
    for camera in names["cameras"].values():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f"Generated scene is missing camera: {camera}")
    for sensor in names["sensors"].values():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor) < 0:
            raise ValueError(f"Generated scene is missing sensor: {sensor}")
    for body in (names["board_body"], names["nic_body"]):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0 or model.body_mocapid[body_id] < 0:
            raise ValueError(f"Generated scene body is not mocap-controlled: {body}")


def prepare_scene(config: dict[str, Any]) -> Path:
    """Generate and compile-check the deterministic reduced scene."""

    scene = config["scene"]
    names = scene["names"]
    robot_path = Path(scene["source_robot"])
    world_path = Path(scene["source_world"])
    output_path = Path(scene["output"])
    if not robot_path.is_file() or not world_path.is_file():
        raise FileNotFoundError("Both source MJCF files must exist before scene preparation")
    if len({robot_path.parent, world_path.parent, output_path.parent}) != 1:
        raise ValueError("Source and reduced MJCF files must share one asset directory")
    if output_path in (robot_path, world_path):
        raise ValueError("The reduced scene output cannot overwrite a source MJCF")

    robot = ET.parse(robot_path).getroot()
    world = ET.parse(world_path).getroot()

    robot_body = copy.deepcopy(named_element(robot, "body", names["robot_root_body"]))
    tool = named_element(robot_body, "body", names["tool_body"])
    fixed_position = float(scene["gripper_fixed_position"])
    finger_names = (
        (names["left_finger_body"], names["left_finger_joint"]),
        (names["right_finger_body"], names["right_finger_joint"]),
    )
    for finger_name, joint_name in finger_names:
        finger = named_element(robot_body, "body", finger_name)
        joint = named_element(finger, "joint", joint_name)
        if joint.get("type") != "slide":
            raise ValueError(f"Fixed gripper joint is not a slide joint: {joint_name}")
        if joint.get("limited") != "true":
            raise ValueError(f"Fixed gripper joint is not explicitly limited: {joint_name}")
        joint_range = parse_vector(joint.get("range"), 2, f"{joint_name} range")
        if fixed_position < joint_range[0] or fixed_position > joint_range[1]:
            raise ValueError(
                f"scene.gripper_fixed_position is outside {joint_name} range"
            )
        finger_position = parse_vector(finger.get("pos"), 3, f"{finger_name} position")
        finger_quaternion = normalized_quaternion(
            parse_vector(finger.get("quat") or "1 0 0 0", 4, "finger quaternion")
        )
        axis = parse_vector(joint.get("axis"), 3, f"{joint_name} axis")
        displacement = rotate_vector(
            finger_quaternion, [fixed_position * value for value in axis]
        )
        finger.set(
            "pos",
            format_vector(
                [
                    position + offset
                    for position, offset in zip(
                        finger_position, displacement, strict=True
                    )
                ]
            ),
        )
        finger.set("quat", format_vector(finger_quaternion))
        finger.remove(joint)
        for geom in finger.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    sfp_source = named_element(world, "body", names["sfp_source_body"])
    lc_source = named_element(world, "body", names["lc_source_body"])
    welds = [
        weld
        for weld in world.iter("weld")
        if weld.get("body1") == names["tool_body"]
        and weld.get("body2") == names["lc_source_body"]
    ]
    if len(welds) != 1:
        raise ValueError("Expected one source weld from the tool body to the LC plug")
    weld_pose = parse_vector(welds[0].get("relpose"), 7, "tool-to-LC weld relpose")
    sfp_local_pos = parse_vector(sfp_source.get("pos"), 3, "SFP local position")
    sfp_local_quat = parse_vector(sfp_source.get("quat"), 4, "SFP local quaternion")
    sfp_pos, sfp_quat = compose_pose(
        weld_pose[:3], weld_pose[3:], sfp_local_pos, sfp_local_quat
    )
    sfp = copy.deepcopy(sfp_source)
    sfp.set("pos", format_vector(sfp_pos))
    sfp.set("quat", format_vector(sfp_quat))
    tool.append(sfp)

    board = copy.deepcopy(named_element(world, "body", names["board_source_body"]))
    nested_nic = named_element(board, "body", names["nic_source_body"])
    if not remove_descendant(board, nested_nic):
        raise ValueError("NIC source body is not a descendant of the task board")
    board.set("name", names["board_body"])
    board.set("mocap", "true")

    nic_source = named_element(world, "body", names["nic_source_body"])
    nic = copy.deepcopy(nic_source)
    nic.set("name", names["nic_body"])
    nic.set("mocap", "true")
    board_source = named_element(world, "body", names["board_source_body"])
    initial_nic_pos, initial_nic_quat = compose_pose(
        parse_vector(board_source.get("pos"), 3, "board position"),
        parse_vector(board_source.get("quat"), 4, "board quaternion"),
        parse_vector(nic_source.get("pos"), 3, "NIC position"),
        parse_vector(nic_source.get("quat") or "1 0 0 0", 4, "NIC quaternion"),
    )
    nic.set("pos", format_vector(initial_nic_pos))
    nic.set("quat", format_vector(initial_nic_quat))

    light = copy.deepcopy(named_element(world, "light", names["light"]))
    for camera_name in names["cameras"].values():
        camera = named_element(robot_body, "camera", camera_name)
        camera.set("resolution", f'{config["cameras"]["width"]} {config["cameras"]["height"]}')

    output = ET.Element("mujoco", {"model": scene["model_name"]})
    output.append(copy.deepcopy(required_element(robot, "compiler", "compiler settings")))
    physics = config["physics"]
    output.append(
        ET.Element(
            "option",
            {
                "gravity": format_vector([float(x) for x in physics["gravity"]]),
                "timestep": f'{physics["timestep"]:.16g}',
                "integrator": physics["integrator"],
                "solver": physics["solver"],
                "iterations": str(physics["iterations"]),
                "tolerance": f'{physics["tolerance"]:.16g}',
            },
        )
    )
    visual = copy.deepcopy(required_element(robot, "visual", "visual settings"))
    global_visual = required_element(visual, "global", "global visual settings")
    global_visual.set("offwidth", str(config["cameras"]["width"]))
    global_visual.set("offheight", str(config["cameras"]["height"]))
    output.append(visual)
    output.append(copy_defaults(robot, world))
    selected = [robot_body, board, nic]
    output.append(copy_assets(robot, world, selected, output_path.parent))

    worldbody = ET.SubElement(output, "worldbody")
    worldbody.extend([light, robot_body, board, nic])
    contact = ET.SubElement(output, "contact")
    source_contact = required_element(robot, "contact", "robot contact exclusions")
    for exclusion in source_contact:
        contact.append(copy.deepcopy(exclusion))
    ET.SubElement(
        contact,
        "exclude",
        {
            "name": "task_board_target_nic",
            "body1": names["board_body"],
            "body2": names["nic_body"],
        },
    )

    source_actuator = required_element(robot, "actuator", "actuator section")
    actuator = ET.SubElement(output, "actuator")
    torque_limits = config["control"]["torque_limits"]
    for actuator_name, limit in zip(names["actuators"], torque_limits, strict=True):
        source = named_element(source_actuator, "general", actuator_name)
        item = copy.deepcopy(source)
        item.set("ctrllimited", "true")
        item.set("ctrlrange", f"{-float(limit):.16g} {float(limit):.16g}")
        actuator.append(item)

    source_sensor = required_element(robot, "sensor", "sensor section")
    sensor = ET.SubElement(output, "sensor")
    for tag, sensor_name in names["sensors"].items():
        sensor.append(copy.deepcopy(named_element(source_sensor, tag, sensor_name)))

    ET.indent(output, space="  ")
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(output, encoding="unicode")
        + "\n"
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary_path.write_text(xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(temporary_path))
        validate_model(model, config)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
