#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaaclab_root="$(cd "${repo_root}/../IsaacLab" 2>/dev/null && pwd || true)"

if [[ -n "${isaaclab_root}" ]]; then
  for path in \
    "${isaaclab_root}/source/isaaclab" \
    "${isaaclab_root}/source/isaaclab_assets" \
    "${isaaclab_root}/source/isaaclab_tasks" \
    "${isaaclab_root}/source/isaaclab_rl" \
    "${isaaclab_root}/source/isaaclab_visualizers" \
    "${repo_root}/aic_utils/aic_isaac/aic_isaaclab/source/aic_task"; do
    if [[ -d "${path}" && ":${PYTHONPATH:-}:" != *":${path}:"* ]]; then
      export PYTHONPATH="${path}:${PYTHONPATH:-}"
    fi
  done
fi

# Re-apply only the non-Python package overlays needed for resource lookup.
# Sourcing install/setup.bash pulls every built package into PYTHONPATH, which
# can shadow Pixi-installed Python packages with stale colcon install copies.
for pkg in aic_assets aic_description; do
  package_setup="${repo_root}/install/${pkg}/share/${pkg}/package.bash"
  if [[ -f "${package_setup}" ]]; then
    # shellcheck disable=SC1090
    source "${package_setup}"
  fi
done
