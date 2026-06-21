"""Core MuJoCo utilities for AIC insertion R&D.

The package is intentionally split into small layers:

``commands``
    Joint target containers used by controllers.
``joints``
    MuJoCo joint/actuator mapping and passive joint handling.
``controllers``
    Joint impedance control that converts targets into torques.
``utils``
    Frame transforms and reset-time pre-insertion IK helpers.
``mjlab``
    Local prototype reset/step/observation/reward utilities.
``warp``
    Direct vector-env, RSL-RL, and MuJoCo Warp preflight code.
"""
