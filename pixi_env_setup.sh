#!/usr/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
