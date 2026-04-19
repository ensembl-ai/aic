#!/usr/bin/env bash
set -euo pipefail

HTTP_PORT="${1:-${AIC_BROWSER_DESKTOP_HTTP_PORT:-6080}}"
DISPLAY_NUM="${AIC_BROWSER_DESKTOP_DISPLAY_NUM:-1}"
DISPLAY_VALUE=":${DISPLAY_NUM}"
VNC_PORT="${AIC_BROWSER_DESKTOP_VNC_PORT:-5901}"
SCREEN_WIDTH="${AIC_BROWSER_DESKTOP_WIDTH:-1920}"
SCREEN_HEIGHT="${AIC_BROWSER_DESKTOP_HEIGHT:-1080}"
SCREEN_DEPTH="${AIC_BROWSER_DESKTOP_DEPTH:-24}"
XORG_LOG="/var/log/aic-browser-desktop-xorg.log"
NOVNC_LOG="/var/log/aic-browser-desktop-novnc.log"
X11VNC_LOG="/var/log/aic-browser-desktop-x11vnc.log"
OPENBOX_LOG="/var/log/aic-browser-desktop-openbox.log"
GLXINFO_LOG="/var/log/aic-browser-desktop-glxinfo.log"
XORG_CONFIG_FILE="/etc/X11/aic-browser-desktop-xorg.conf"
ENV_FILE="/etc/profile.d/aic_browser_desktop_env.sh"
NVIDIA_XORG_MODULE_DIR=""
NVIDIA_EGL_VENDOR_FILE=""
NVIDIA_DEVICE_BUS_ID=""
NVIDIA_RUNTIME_RESOLVED=0

have_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1
}

resolve_existing_path() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

find_nvidia_xorg_module_dir() {
  local driver_path

  driver_path="$(
    resolve_existing_path \
      /usr/lib/x86_64-linux-gnu/nvidia/xorg/drivers/nvidia_drv.so \
      /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so \
      /usr/lib/aarch64-linux-gnu/nvidia/xorg/drivers/nvidia_drv.so \
      /usr/lib/aarch64-linux-gnu/nvidia/xorg/nvidia_drv.so \
      /usr/lib64/nvidia/xorg/drivers/nvidia_drv.so \
      /usr/lib64/nvidia/xorg/nvidia_drv.so \
      /usr/lib/nvidia/xorg/drivers/nvidia_drv.so \
      /usr/lib/nvidia/xorg/nvidia_drv.so \
      /usr/lib64/xorg/modules/drivers/nvidia_drv.so \
      /usr/lib/xorg/modules/drivers/nvidia_drv.so
  )" || return 1

  case "${driver_path}" in
    */nvidia/xorg/drivers/nvidia_drv.so)
      dirname "$(dirname "${driver_path}")"
      ;;
    */xorg/modules/drivers/nvidia_drv.so)
      dirname "$(dirname "${driver_path}")"
      ;;
    *)
      dirname "${driver_path}"
      ;;
  esac
}

find_nvidia_egl_vendor_file() {
  local env_path

  if [[ -n "${AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE:-}" ]] && [[ -f "${AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE}" ]]; then
    echo "${AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE}"
    return 0
  fi

  if [[ -n "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]]; then
    IFS=':' read -r -a env_paths <<< "${__EGL_VENDOR_LIBRARY_FILENAMES}"
    for env_path in "${env_paths[@]}"; do
      if [[ -f "${env_path}" ]]; then
        echo "${env_path}"
        return 0
      fi
    done
  fi

  resolve_existing_path \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    /etc/glvnd/egl_vendor.d/10_nvidia.json
}

normalize_xorg_bus_id() {
  local raw_bus_id="${1^^}"

  if [[ "${raw_bus_id}" =~ ^PCI:[0-9]+:[0-9]+:[0-9]+$ ]]; then
    echo "${raw_bus_id}"
    return 0
  fi

  if [[ "${raw_bus_id}" =~ ^(([0-9A-F]{4}|[0-9A-F]{8}):)?([0-9A-F]{2}):([0-9A-F]{2})\.([0-9]+)$ ]]; then
    printf 'PCI:%d:%d:%d\n' \
      "$((16#${BASH_REMATCH[3]}))" \
      "$((16#${BASH_REMATCH[4]}))" \
      "${BASH_REMATCH[5]}"
    return 0
  fi

  return 1
}

detect_nvidia_bus_id() {
  local raw_bus_id=""
  local drm_path=""

  if [[ -n "${AIC_BROWSER_DESKTOP_NVIDIA_BUS_ID:-}" ]]; then
    raw_bus_id="${AIC_BROWSER_DESKTOP_NVIDIA_BUS_ID}"
  elif [[ -d /dev/dri/by-path ]]; then
    drm_path="$(find /dev/dri/by-path -maxdepth 1 -type l -name 'pci-*-card' | sort | head -n 1 || true)"
    if [[ -n "${drm_path}" ]]; then
      raw_bus_id="$(basename "${drm_path}")"
      raw_bus_id="${raw_bus_id#pci-}"
      raw_bus_id="${raw_bus_id%-card}"
    fi
  else
    raw_bus_id="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  fi

  [[ -n "${raw_bus_id}" ]] || return 1

  normalize_xorg_bus_id "${raw_bus_id}"
}

resolve_nvidia_runtime() {
  if [[ "${NVIDIA_RUNTIME_RESOLVED}" -eq 1 ]]; then
    return 0
  fi

  NVIDIA_XORG_MODULE_DIR="$(find_nvidia_xorg_module_dir || true)"
  NVIDIA_EGL_VENDOR_FILE="$(find_nvidia_egl_vendor_file || true)"
  NVIDIA_DEVICE_BUS_ID="$(detect_nvidia_bus_id || true)"
  NVIDIA_RUNTIME_RESOLVED=1
}

install_packages() {
  local packages=(
    dbus-x11
    mesa-utils
    net-tools
    novnc
    openbox
    python3-websockify
    x11-utils
    x11-xserver-utils
    x11vnc
    xauth
    xserver-xorg-core
  )

  apt-get update

  DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
}

write_env_file() {
  {
    echo "export DISPLAY=${DISPLAY_VALUE}"
    echo "export QT_X11_NO_MITSHM=1"
    echo "export XAUTHORITY=/root/.Xauthority"
    echo "export __GLX_VENDOR_LIBRARY_NAME=nvidia"
    if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
      echo "export __EGL_VENDOR_LIBRARY_FILENAMES=${NVIDIA_EGL_VENDOR_FILE}"
    fi
    if [[ -n "${NVIDIA_DEVICE_BUS_ID}" ]]; then
      echo "export AIC_BROWSER_DESKTOP_NVIDIA_BUS_ID=${NVIDIA_DEVICE_BUS_ID}"
    fi
  } > "${ENV_FILE}"
}

write_nvidia_xorg_config() {
  local mode="${1}"

  mkdir -p "$(dirname "${XORG_CONFIG_FILE}")"

  {
    echo 'Section "Files"'
    if [[ -n "${NVIDIA_XORG_MODULE_DIR}" ]]; then
      echo "    ModulePath \"${NVIDIA_XORG_MODULE_DIR}\""
    fi
    echo '    ModulePath "/usr/lib/xorg/modules"'
    echo 'EndSection'
    echo
    echo 'Section "ServerLayout"'
    echo '    Identifier "layout"'
    echo '    Screen 0 "screen"'
    echo '    Option "AllowNVIDIAGPUScreens" "true"'
    echo 'EndSection'
    echo
    echo 'Section "ServerFlags"'
    echo '    Option "AutoAddGPU" "false"'
    echo 'EndSection'
    echo
    echo 'Section "Monitor"'
    echo '    Identifier "monitor"'
    echo '    HorizSync 28.0-80.0'
    echo '    VertRefresh 48.0-75.0'
    echo 'EndSection'
    echo
    echo 'Section "Device"'
    echo '    Identifier "device"'
    echo '    Driver "nvidia"'
    echo '    VendorName "NVIDIA Corporation"'
    if [[ -n "${NVIDIA_DEVICE_BUS_ID}" ]]; then
      echo "    BusID \"${NVIDIA_DEVICE_BUS_ID}\""
    fi
    echo '    Option "AllowEmptyInitialConfiguration" "true"'
    echo '    Option "AllowExternalGpus" "true"'
    echo '    Option "ProbeAllGpus" "false"'
    case "${mode}" in
      none)
        echo '    Option "UseDisplayDevice" "none"'
        ;;
      dfp)
        echo '    Option "ConnectedMonitor" "DFP"'
        echo '    Option "UseDisplayDevice" "DFP"'
        ;;
    esac
    echo 'EndSection'
    echo
    echo 'Section "Screen"'
    echo '    Identifier "screen"'
    echo '    Device "device"'
    echo '    Monitor "monitor"'
    echo "    DefaultDepth ${SCREEN_DEPTH}"
    echo '    SubSection "Display"'
    echo "        Depth ${SCREEN_DEPTH}"
    echo "        Virtual ${SCREEN_WIDTH} ${SCREEN_HEIGHT}"
    echo '    EndSubSection'
    echo 'EndSection'
  } > "${XORG_CONFIG_FILE}"
}

run_glxinfo() {
  local -a env_args=(
    DISPLAY="${DISPLAY_VALUE}"
    XAUTHORITY=/root/.Xauthority
    __GLX_VENDOR_LIBRARY_NAME=nvidia
  )

  if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
    env_args+=("__EGL_VENDOR_LIBRARY_FILENAMES=${NVIDIA_EGL_VENDOR_FILE}")
  fi

  env "${env_args[@]}" glxinfo -B
}

append_log_banner() {
  local message="$1"

  printf '\n===== %s (%s) =====\n' "${message}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "${XORG_LOG}"
}

display_uses_nvidia_glx() {
  local glxinfo_output

  ensure_xauthority

  if ! glxinfo_output="$(run_glxinfo 2>&1)"; then
    printf '%s\n' "${glxinfo_output}" > "${GLXINFO_LOG}"
    return 1
  fi

  printf '%s\n' "${glxinfo_output}" > "${GLXINFO_LOG}"

  [[ "${glxinfo_output}" == *"OpenGL vendor string: NVIDIA Corporation"* ]] || \
  [[ "${glxinfo_output}" == *"OpenGL renderer string: NVIDIA"* ]]
}

stop_process_from_pidfile() {
  local pidfile="$1"
  local pid

  [[ -f "${pidfile}" ]] || return 0

  pid="$(<"${pidfile}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi

  rm -f "${pidfile}"
}

stop_desktop_stack() {
  stop_process_from_pidfile /tmp/aic-browser-desktop-novnc.pid
  stop_process_from_pidfile /tmp/aic-browser-desktop-x11vnc.pid
  stop_process_from_pidfile /tmp/aic-browser-desktop-openbox.pid
  stop_process_from_pidfile /tmp/aic-browser-desktop-xorg.pid

  pkill -f "Xorg ${DISPLAY_VALUE}" >/dev/null 2>&1 || true

  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
  rm -rf "/tmp/.X11-unix/X${DISPLAY_NUM}"
}

start_detached_process() {
  local pidfile="$1"
  local logfile="$2"
  shift
  shift

  nohup "$@" >> "${logfile}" 2>&1 </dev/null &
  echo "$!" > "${pidfile}"
}

fail_nvidia_display() {
  echo "$1" >&2
  echo "Gazebo needs a real NVIDIA-backed GLX display on ${DISPLAY_VALUE}." >&2
  echo "See ${XORG_LOG} and ${GLXINFO_LOG} for details." >&2
  return 1
}

validate_nvidia_runtime() {
  if ! have_nvidia_gpu; then
    fail_nvidia_display "No NVIDIA GPU was detected in this environment."
  fi

  resolve_nvidia_runtime

  if [[ -z "${NVIDIA_XORG_MODULE_DIR}" ]]; then
    echo "Missing NVIDIA Xorg driver files inside the container." >&2
    echo "Looked for nvidia_drv.so in the common NVIDIA runtime locations." >&2
    echo "This environment should provide NVIDIA userspace libraries via the container runtime, not by installing drivers in the container." >&2
    fail_nvidia_display "NVIDIA container runtime is incomplete."
  fi

  if [[ -z "${NVIDIA_DEVICE_BUS_ID}" ]]; then
    echo "Warning: could not auto-detect a PCI BusID from nvidia-smi; Xorg will rely on auto-selection." >&2
  fi
}

start_xorg() {
  local mode="$1"
  local -a xorg_args

  stop_process_from_pidfile /tmp/aic-browser-desktop-xorg.pid
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
  rm -rf "/tmp/.X11-unix/X${DISPLAY_NUM}"

  write_nvidia_xorg_config "${mode}"
  append_log_banner "Starting Xorg on ${DISPLAY_VALUE} with mode=${mode}"

  xorg_args=(
    "${DISPLAY_VALUE}"
    -config "${XORG_CONFIG_FILE}"
    -logfile "${XORG_LOG}"
    -noreset
    +extension GLX
    +extension RANDR
    +extension RENDER
  )

  if [[ -n "${NVIDIA_DEVICE_BUS_ID}" ]]; then
    xorg_args+=(-isolateDevice "${NVIDIA_DEVICE_BUS_ID}")
  fi

  nohup Xorg "${xorg_args[@]}" >> "${XORG_LOG}" 2>&1 </dev/null &
  local xorg_pid=$!

  for _ in $(seq 1 20); do
    if [[ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]]; then
      echo "${xorg_pid}" > "/tmp/aic-browser-desktop-xorg.pid"
      return 0
    fi

    if ! kill -0 "${xorg_pid}" >/dev/null 2>&1; then
      wait "${xorg_pid}" >/dev/null 2>&1 || true
      return 1
    fi

    sleep 1
  done

  kill "${xorg_pid}" >/dev/null 2>&1 || true
  wait "${xorg_pid}" >/dev/null 2>&1 || true
  return 1
}

ensure_xauthority() {
  touch /root/.Xauthority
  xauth remove "${DISPLAY_VALUE}" >/dev/null 2>&1 || true
  xauth add "${DISPLAY_VALUE}" . "$(mcookie)"
}

ensure_xorg() {
  local mode
  local -a modes

  validate_nvidia_runtime

  if [[ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]] && display_uses_nvidia_glx; then
    return 0
  fi

  stop_desktop_stack
  : > "${XORG_LOG}"

  if [[ -n "${AIC_BROWSER_DESKTOP_XORG_MODES:-}" ]]; then
    read -r -a modes <<< "${AIC_BROWSER_DESKTOP_XORG_MODES}"
  else
    modes=(none auto dfp)
  fi

  for mode in "${modes[@]}"; do
    if start_xorg "${mode}" && display_uses_nvidia_glx; then
      return 0
    fi

    stop_desktop_stack
  done

  fail_nvidia_display "Failed to start NVIDIA-backed Xorg on ${DISPLAY_VALUE}."
}

start_window_manager() {
  if pgrep -f "openbox.*${DISPLAY_VALUE}" >/dev/null 2>&1; then
    return 0
  fi

  start_detached_process /tmp/aic-browser-desktop-openbox.pid \
    "${OPENBOX_LOG}" \
    env DISPLAY="${DISPLAY_VALUE}" XAUTHORITY=/root/.Xauthority \
    dbus-launch --exit-with-session openbox
}

start_x11vnc() {
  if pgrep -f "x11vnc.*-rfbport ${VNC_PORT}" >/dev/null 2>&1; then
    return 0
  fi

  mkdir -p /root/.vnc

  local args=(
    -display "${DISPLAY_VALUE}"
    -rfbport "${VNC_PORT}"
    -forever
    -shared
    -localhost
    -nopw
    -xkb
    -noxdamage
    -o "${X11VNC_LOG}"
  )

  if [[ -n "${AIC_BROWSER_DESKTOP_PASSWORD:-}" ]]; then
    x11vnc -storepasswd "${AIC_BROWSER_DESKTOP_PASSWORD}" /root/.vnc/passwd >/dev/null
    args=(
      -display "${DISPLAY_VALUE}"
      -rfbport "${VNC_PORT}"
      -forever
      -shared
      -localhost
      -rfbauth /root/.vnc/passwd
      -xkb
      -noxdamage
      -o "${X11VNC_LOG}"
    )
  fi

  start_detached_process /tmp/aic-browser-desktop-x11vnc.pid \
    "${X11VNC_LOG}" \
    x11vnc "${args[@]}"
}

find_novnc_proxy() {
  local candidates=(
    "/usr/share/novnc/utils/novnc_proxy"
    "/usr/share/novnc/utils/launch.sh"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

start_novnc() {
  local novnc_proxy

  if pgrep -f "novnc.*${HTTP_PORT}" >/dev/null 2>&1 || pgrep -f "websockify.*${HTTP_PORT}" >/dev/null 2>&1; then
    return 0
  fi

  novnc_proxy="$(find_novnc_proxy)" || {
    echo "Unable to locate the noVNC proxy executable." >&2
    return 1
  }

  start_detached_process /tmp/aic-browser-desktop-novnc.pid \
    "${NOVNC_LOG}" \
    "${novnc_proxy}" --listen "${HTTP_PORT}" --vnc "localhost:${VNC_PORT}"
}

print_summary() {
  cat <<ENVEOF
Browser desktop is ready.
  HTTP: http://127.0.0.1:${HTTP_PORT}/vnc.html
  DISPLAY: ${DISPLAY_VALUE}
  VNC port: ${VNC_PORT} (localhost only)

Use this display for Gazebo/RViz:
  export DISPLAY=${DISPLAY_VALUE}
  export QT_X11_NO_MITSHM=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
ENVEOF

  if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
    echo "  export __EGL_VENDOR_LIBRARY_FILENAMES=${NVIDIA_EGL_VENDOR_FILE}"
  fi

  cat <<ENVEOF

To verify GPU rendering inside the same environment:
  glxinfo -B
  nvidia-smi
ENVEOF
}

main() {
  install_packages
  validate_nvidia_runtime
  write_env_file

  export DISPLAY="${DISPLAY_VALUE}"
  export QT_X11_NO_MITSHM=1
  export XAUTHORITY=/root/.Xauthority
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_FILE}"
  fi

  ensure_xorg

  xhost +local: >/dev/null 2>&1 || true
  DISPLAY="${DISPLAY_VALUE}" xsetroot -solid "#20252b" >/dev/null 2>&1 || true

  start_window_manager
  start_x11vnc
  start_novnc
  print_summary
}

main "$@"
