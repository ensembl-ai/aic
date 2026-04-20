#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_setup="${repo_root}/install/setup.bash"

# Re-apply the local colcon overlay inside Pixi so ROS package discovery
# continues to work for packages built into this workspace.
if [[ -f "${workspace_setup}" ]]; then
  # shellcheck disable=SC1090
  source "${workspace_setup}"
fi
