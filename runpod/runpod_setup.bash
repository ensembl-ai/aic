#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# AIC RunPod setup script
#
# NVIDIA ONLY.
# NO Xvfb.
# NO backend fallback.
# NO fake root password.
# NO automatic destructive restart unless --restart-desktop.
#
# First setup:
#   ./aic_runpod_setup.bash --first
#
# Successive safe runs:
#   ./aic_runpod_setup.bash
#
# Explicit desktop restart:
#   ./aic_runpod_setup.bash --restart-desktop
#
# Root password default is root. Override explicitly:
#   ./aic_runpod_setup.bash --first --root-password "your-password"
#
# DO NOT put this script in ~/.bashrc.
# ============================================================

FIRST=0
RESTART_DESKTOP=0
START_DESKTOP=1
ROOT_PASSWORD="root"

ROS_SETUP="${ROS_SETUP:-/opt/ros/kilted/setup.bash}"

PIXI_BIN_DIR="/root/.pixi/bin"
PIXIBIN="${PIXI_BIN_DIR}/pixi"

PROFILE_ENV_FILE="/etc/profile.d/aic_runpod_env.sh"
DESKTOP_ENV_FILE="/etc/profile.d/aic_browser_desktop_env.sh"

DISPLAY_NUM="${AIC_BROWSER_DESKTOP_DISPLAY_NUM:-1}"
DISPLAY_VALUE=":${DISPLAY_NUM}"

HTTP_PORT="${AIC_BROWSER_DESKTOP_HTTP_PORT:-6080}"
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

NVIDIA_XORG_MODULE_DIR=""
NVIDIA_EGL_VENDOR_FILE=""
NVIDIA_DEVICE_BUS_ID=""

log() {
  echo "[aic-setup] $*"
}

die() {
  echo "[aic-setup:ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  ./aic_runpod_setup.bash --first
  ./aic_runpod_setup.bash
  ./aic_runpod_setup.bash --restart-desktop
  ./aic_runpod_setup.bash --no-desktop

Flags:
  --first                 One-time install/config setup.
  --restart-desktop       Explicitly stop and restart NVIDIA Xorg/noVNC/x11vnc/openbox.
  --no-desktop            Do not start desktop stack.
  --root-password VALUE   Set root password. Default is root unless overridden.
  --http-port PORT        noVNC HTTP port. Default: 6080.
  --vnc-port PORT         local VNC port. Default: 5901.
  --display-num NUM       X display number. Default: 1.
  --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --first)
      FIRST=1
      shift
      ;;
    --restart-desktop)
      RESTART_DESKTOP=1
      shift
      ;;
    --no-desktop)
      START_DESKTOP=0
      shift
      ;;
    --root-password)
      ROOT_PASSWORD="${2:?Missing value for --root-password}"
      shift 2
      ;;
    --http-port)
      HTTP_PORT="${2:?Missing value for --http-port}"
      shift 2
      ;;
    --vnc-port)
      VNC_PORT="${2:?Missing value for --vnc-port}"
      shift 2
      ;;
    --display-num)
      DISPLAY_NUM="${2:?Missing value for --display-num}"
      DISPLAY_VALUE=":${DISPLAY_NUM}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  die "Run this as root inside the RunPod container."
fi

exec 9>/tmp/aic_runpod_setup.lock
if ! flock -n 9; then
  die "Another setup run is already active. Refusing to run concurrently."
fi

source_ros_if_present() {
  if [[ ! -f "${ROS_SETUP}" ]]; then
    die "ROS setup file not found: ${ROS_SETUP}"
  fi

  local aic_had_nounset=0
  case "$-" in
    *u*)
      aic_had_nounset=1
      set +u
      ;;
  esac

  # shellcheck source=/dev/null
  source "${ROS_SETUP}"

  if [[ "${aic_had_nounset}" -eq 1 ]]; then
    set -u
  fi
}

write_base_env_file() {
  cat > "${PROFILE_ENV_FILE}" <<EOF
export PATH="${PIXI_BIN_DIR}:\$PATH"
export QT_X11_NO_MITSHM=1
export XAUTHORITY=/root/.Xauthority
export AIC_ENABLE_BROWSER_DESKTOP=1
export AIC_BROWSER_DESKTOP_HTTP_PORT="${HTTP_PORT}"
export AIC_BROWSER_DESKTOP_VNC_PORT="${VNC_PORT}"
export AIC_BROWSER_DESKTOP_DISPLAY_NUM="${DISPLAY_NUM}"
export ROS_SETUP="${ROS_SETUP}"

if [ -f "\${ROS_SETUP}" ]; then
  _aic_profile_had_nounset=0
  case "\$-" in
    *u*)
      _aic_profile_had_nounset=1
      set +u
      ;;
  esac

  source "\${ROS_SETUP}"

  if [ "\${_aic_profile_had_nounset}" = "1" ]; then
    set -u
  fi

  unset _aic_profile_had_nounset
fi
EOF
}

write_desktop_env_file() {
  {
    echo "export DISPLAY=${DISPLAY_VALUE}"
    echo "export QT_X11_NO_MITSHM=1"
    echo "export XAUTHORITY=/root/.Xauthority"
    echo "export __GLX_VENDOR_LIBRARY_NAME=nvidia"
    echo "export AIC_BROWSER_DESKTOP_HTTP_PORT=${HTTP_PORT}"
    echo "export AIC_BROWSER_DESKTOP_VNC_PORT=${VNC_PORT}"
    echo "export AIC_BROWSER_DESKTOP_DISPLAY_NUM=${DISPLAY_NUM}"

    if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
      echo "export __EGL_VENDOR_LIBRARY_FILENAMES=${NVIDIA_EGL_VENDOR_FILE}"
    fi

    if [[ -n "${NVIDIA_DEVICE_BUS_ID}" ]]; then
      echo "export AIC_BROWSER_DESKTOP_NVIDIA_BUS_ID=${NVIDIA_DEVICE_BUS_ID}"
    fi
  } > "${DESKTOP_ENV_FILE}"
}

install_first_run_packages() {
  export DEBIAN_FRONTEND=noninteractive

  log "Running first-time apt install."

  apt-get update

  apt-get install -y \
    ca-certificates \
    curl \
    dbus-x11 \
    git \
    iproute2 \
    mesa-utils \
    net-tools \
    novnc \
    openbox \
    openssh-server \
    python3-websockify \
    vim \
    x11-utils \
    x11-xserver-utils \
    x11vnc \
    xauth \
    xserver-xorg-core

  log "First-time apt install complete."
}

install_pixi_if_needed() {
  if [[ ! -x "${PIXIBIN}" ]]; then
    log "Installing Pixi."
    curl -fsSL https://pixi.sh/install.sh | sh
  else
    log "Pixi already exists."
  fi

  export PATH="${PIXI_BIN_DIR}:$PATH"
}

install_node_and_codex_if_needed() {
  local node_major="0"

  if command -v node >/dev/null 2>&1; then
    node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
  fi

  if [[ "${node_major}" -lt 22 ]] || ! command -v npm >/dev/null 2>&1; then
    log "Installing Node.js 22 from NodeSource because node>=22 and npm are required."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
  else
    log "Node.js >=22 and npm already exist."
  fi

  command -v node >/dev/null 2>&1 || die "node still missing after Node.js install."
  command -v npm >/dev/null 2>&1 || die "npm still missing after Node.js install."

  local final_node_major
  final_node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"

  if [[ "${final_node_major}" -lt 22 ]]; then
    die "Node.js major version is ${final_node_major}; expected >=22."
  fi

  log "Node version: $(node -v)"
  log "npm version: $(npm -v)"

  if ! command -v codex >/dev/null 2>&1; then
    log "Installing Codex CLI."
    npm i -g @openai/codex
  else
    log "Codex CLI already exists."
  fi
}
set_sshd_option() {
  local key="$1"
  local value="$2"
  local file="/etc/ssh/sshd_config"

  [[ -f "${file}" ]] || die "Missing ${file}. Run with --first."

  if grep -Eq "^[#[:space:]]*${key}[[:space:]]+" "${file}"; then
    sed -ri "s|^[#[:space:]]*${key}[[:space:]].*|${key} ${value}|g" "${file}"
  else
    echo "${key} ${value}" >> "${file}"
  fi
}

configure_sshd() {
  command -v sshd >/dev/null 2>&1 || die "sshd not installed. Run with --first."

  mkdir -p /var/run/sshd
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh

  if [[ -n "${ROOT_PASSWORD}" ]]; then
    echo "root:${ROOT_PASSWORD}" | chpasswd
    log "Root password set from explicit --root-password argument."
  else
    log "No root password set. No dummy password used."
  fi

  set_sshd_option "PermitRootLogin" "yes"
  set_sshd_option "PasswordAuthentication" "yes"

  grep -q '^ListenAddress 0.0.0.0$' /etc/ssh/sshd_config || \
    echo "ListenAddress 0.0.0.0" >> /etc/ssh/sshd_config

  /usr/sbin/sshd -t
}

ensure_sshd_running() {
  if ss -ltnp 2>/dev/null | grep -qE 'LISTEN.+:22[[:space:]]'; then
    log "sshd already listening on port 22."
    return 0
  fi

  log "Starting sshd."
  /usr/sbin/sshd
}

setup_git_and_github_ssh() {
  git config --global user.name "rishimalhan"
  git config --global user.email "rmalhan0112@gmail.com"
  git config --global init.defaultBranch main

  mkdir -p /root/.ssh
  chmod 700 /root/.ssh

  if [[ ! -f /root/.ssh/id_ed25519 ]]; then
    log "Generating GitHub SSH key."
    ssh-keygen -t ed25519 -C "rmalhan0112@gmail.com" -f /root/.ssh/id_ed25519 -N ""
  else
    log "GitHub SSH key already exists."
  fi

  ssh-keygen -F github.com >/dev/null 2>&1 || \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null

  chmod 644 /root/.ssh/known_hosts
  git config --global core.sshCommand "ssh -i /root/.ssh/id_ed25519"

  echo
  echo "GitHub public key:"
  cat /root/.ssh/id_ed25519.pub
  echo
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
  if [[ -n "${AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE:-}" ]]; then
    [[ -f "${AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE}" ]] || \
      die "AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE is set but file does not exist: ${AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE}"

    echo "${AIC_BROWSER_DESKTOP_EGL_VENDOR_FILE}"
    return 0
  fi

  if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    echo /usr/share/glvnd/egl_vendor.d/10_nvidia.json
    return 0
  fi

  if [[ -f /etc/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    echo /etc/glvnd/egl_vendor.d/10_nvidia.json
    return 0
  fi

  return 1
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

  if [[ -n "${AIC_BROWSER_DESKTOP_NVIDIA_BUS_ID:-}" ]]; then
    raw_bus_id="${AIC_BROWSER_DESKTOP_NVIDIA_BUS_ID}"
  else
    raw_bus_id="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  fi

  [[ -n "${raw_bus_id}" ]] || die "Could not detect NVIDIA PCI bus ID from nvidia-smi."

  normalize_xorg_bus_id "${raw_bus_id}" || die "Could not normalize NVIDIA PCI bus ID: ${raw_bus_id}"
}

resolve_nvidia_runtime() {
  NVIDIA_XORG_MODULE_DIR="$(find_nvidia_xorg_module_dir)" || die "Missing NVIDIA Xorg driver file nvidia_drv.so inside container."

  NVIDIA_EGL_VENDOR_FILE="$(find_nvidia_egl_vendor_file || true)"

  NVIDIA_DEVICE_BUS_ID="$(detect_nvidia_bus_id)"

  log "NVIDIA Xorg module dir: ${NVIDIA_XORG_MODULE_DIR}"
  log "NVIDIA BusID: ${NVIDIA_DEVICE_BUS_ID}"

  if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
    log "NVIDIA EGL vendor file: ${NVIDIA_EGL_VENDOR_FILE}"
  else
    log "No NVIDIA EGL vendor file found. Not setting __EGL_VENDOR_LIBRARY_FILENAMES."
  fi
}

validate_nvidia_runtime() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. NVIDIA runtime is not visible."

  nvidia-smi >/dev/null 2>&1 || die "nvidia-smi exists but failed. NVIDIA runtime is broken."

  command -v Xorg >/dev/null 2>&1 || die "Xorg missing. Run with --first."
  command -v glxinfo >/dev/null 2>&1 || die "glxinfo missing. Run with --first."
  command -v xauth >/dev/null 2>&1 || die "xauth missing. Run with --first."
  command -v xdpyinfo >/dev/null 2>&1 || die "xdpyinfo missing. Run with --first."

  resolve_nvidia_runtime
}

write_nvidia_xorg_config() {
  mkdir -p "$(dirname "${XORG_CONFIG_FILE}")"

  cat > "${XORG_CONFIG_FILE}" <<EOF
Section "Files"
    ModulePath "${NVIDIA_XORG_MODULE_DIR}"
    ModulePath "/usr/lib/xorg/modules"
EndSection

Section "ServerLayout"
    Identifier "layout"
    Screen 0 "screen"
    Option "AllowNVIDIAGPUScreens" "true"
EndSection

Section "ServerFlags"
    Option "AutoAddGPU" "false"
EndSection

Section "Monitor"
    Identifier "monitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
EndSection

Section "Device"
    Identifier "device"
    Driver "nvidia"
    VendorName "NVIDIA Corporation"
    BusID "${NVIDIA_DEVICE_BUS_ID}"
    Option "AllowEmptyInitialConfiguration" "true"
    Option "AllowExternalGpus" "true"
    Option "ProbeAllGpus" "false"
    Option "UseDisplayDevice" "none"
EndSection

Section "Screen"
    Identifier "screen"
    Device "device"
    Monitor "monitor"
    DefaultDepth ${SCREEN_DEPTH}
    SubSection "Display"
        Depth ${SCREEN_DEPTH}
        Virtual ${SCREEN_WIDTH} ${SCREEN_HEIGHT}
    EndSubSection
EndSection
EOF
}

ensure_xauthority() {
  touch /root/.Xauthority
  xauth remove "${DISPLAY_VALUE}" >/dev/null 2>&1 || true
  xauth add "${DISPLAY_VALUE}" . "$(mcookie)" >/dev/null
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

display_responds() {
  DISPLAY="${DISPLAY_VALUE}" XAUTHORITY=/root/.Xauthority xdpyinfo >/dev/null 2>&1
}

display_uses_nvidia_glx() {
  local glxinfo_output

  if ! glxinfo_output="$(run_glxinfo 2>&1)"; then
    printf '%s\n' "${glxinfo_output}" > "${GLXINFO_LOG}"
    return 1
  fi

  printf '%s\n' "${glxinfo_output}" > "${GLXINFO_LOG}"

  [[ "${glxinfo_output}" == *"OpenGL vendor string: NVIDIA Corporation"* ]] || \
  [[ "${glxinfo_output}" == *"OpenGL renderer string: NVIDIA"* ]]
}

start_detached_process() {
  local pidfile="$1"
  local logfile="$2"
  shift 2

  nohup "$@" >> "${logfile}" 2>&1 </dev/null &
  echo "$!" > "${pidfile}"
}

stop_process_from_pidfile() {
  local pidfile="$1"

  [[ -f "${pidfile}" ]] || return 0

  local pid
  pid="$(cat "${pidfile}" 2>/dev/null || true)"

  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi

  rm -f "${pidfile}"
}

stop_desktop_stack() {
  log "Stopping desktop stack because --restart-desktop was explicitly requested."

  stop_process_from_pidfile /tmp/aic-browser-desktop-novnc.pid
  stop_process_from_pidfile /tmp/aic-browser-desktop-x11vnc.pid
  stop_process_from_pidfile /tmp/aic-browser-desktop-openbox.pid
  stop_process_from_pidfile /tmp/aic-browser-desktop-xorg.pid

  pkill -f "Xorg ${DISPLAY_VALUE}" >/dev/null 2>&1 || true
  pkill -f "x11vnc.*-rfbport ${VNC_PORT}" >/dev/null 2>&1 || true
  pkill -f "novnc.*${HTTP_PORT}" >/dev/null 2>&1 || true
  pkill -f "websockify.*${HTTP_PORT}" >/dev/null 2>&1 || true
  pkill -f "openbox" >/dev/null 2>&1 || true

  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
  rm -rf "/tmp/.X11-unix/X${DISPLAY_NUM}"
}

start_nvidia_xorg() {
  validate_nvidia_runtime
  ensure_xauthority
  write_nvidia_xorg_config

  if display_responds; then
    if display_uses_nvidia_glx; then
      log "Existing display ${DISPLAY_VALUE} is already NVIDIA-backed. Not restarting."
      return 0
    fi

    die "Display ${DISPLAY_VALUE} exists but is not NVIDIA GLX. Use --restart-desktop if you want to replace it."
  fi

  : > "${XORG_LOG}"

  local -a xorg_args=(
    "${DISPLAY_VALUE}"
    -config "${XORG_CONFIG_FILE}"
    -logfile "${XORG_LOG}"
    -noreset
    +extension GLX
    +extension RANDR
    +extension RENDER
    -isolateDevice "${NVIDIA_DEVICE_BUS_ID}"
  )

  log "Starting NVIDIA Xorg on ${DISPLAY_VALUE}."
  log "Xorg config: ${XORG_CONFIG_FILE}"
  log "Xorg log: ${XORG_LOG}"

  start_detached_process /tmp/aic-browser-desktop-xorg.pid \
    "${XORG_LOG}" \
    Xorg "${xorg_args[@]}"

  for _ in $(seq 1 20); do
    if display_responds; then
      if display_uses_nvidia_glx; then
        log "NVIDIA Xorg is ready."
        return 0
      fi
    fi
    sleep 1
  done

  die "NVIDIA Xorg started but did not validate as NVIDIA GLX. Check ${XORG_LOG} and ${GLXINFO_LOG}."
}

find_novnc_proxy() {
  if [[ -x /usr/share/novnc/utils/novnc_proxy ]]; then
    echo /usr/share/novnc/utils/novnc_proxy
    return 0
  fi

  if [[ -x /usr/share/novnc/utils/launch.sh ]]; then
    echo /usr/share/novnc/utils/launch.sh
    return 0
  fi

  return 1
}

start_window_manager() {
  if pgrep -f "openbox" >/dev/null 2>&1; then
    log "openbox already running."
    return 0
  fi

  log "Starting openbox."

  start_detached_process /tmp/aic-browser-desktop-openbox.pid \
    "${OPENBOX_LOG}" \
    env DISPLAY="${DISPLAY_VALUE}" XAUTHORITY=/root/.Xauthority \
    dbus-launch --exit-with-session openbox
}

start_x11vnc() {
  command -v x11vnc >/dev/null 2>&1 || die "x11vnc missing. Run with --first."

  if ss -ltnp 2>/dev/null | grep -qE ":${VNC_PORT}[[:space:]]"; then
    if pgrep -f "x11vnc.*-rfbport ${VNC_PORT}" >/dev/null 2>&1; then
      log "x11vnc already listening on ${VNC_PORT}."
      return 0
    fi

    die "Port ${VNC_PORT} is already occupied, but not by expected x11vnc."
  fi

  mkdir -p /root/.vnc

  local -a args=(
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

  log "Starting x11vnc on localhost:${VNC_PORT}."

  start_detached_process /tmp/aic-browser-desktop-x11vnc.pid \
    "${X11VNC_LOG}" \
    x11vnc "${args[@]}"
}

start_novnc() {
  local novnc_proxy

  novnc_proxy="$(find_novnc_proxy)" || die "Unable to locate noVNC proxy."

  if ss -ltnp 2>/dev/null | grep -qE ":${HTTP_PORT}[[:space:]]"; then
    if pgrep -f "novnc.*${HTTP_PORT}" >/dev/null 2>&1 || \
       pgrep -f "websockify.*${HTTP_PORT}" >/dev/null 2>&1; then
      log "noVNC already listening on ${HTTP_PORT}."
      return 0
    fi

    die "Port ${HTTP_PORT} is already occupied, but not by expected noVNC/websockify."
  fi

  log "Starting noVNC on port ${HTTP_PORT}."

  start_detached_process /tmp/aic-browser-desktop-novnc.pid \
    "${NOVNC_LOG}" \
    "${novnc_proxy}" --listen "${HTTP_PORT}" --vnc "localhost:${VNC_PORT}"
}

ensure_desktop_stack() {
  command -v openbox >/dev/null 2>&1 || die "openbox missing. Run with --first."
  command -v dbus-launch >/dev/null 2>&1 || die "dbus-launch missing. Run with --first."
  find_novnc_proxy >/dev/null 2>&1 || die "noVNC missing. Run with --first."

  if [[ "${RESTART_DESKTOP}" -eq 1 ]]; then
    stop_desktop_stack
  fi

  start_nvidia_xorg
  write_desktop_env_file

  export DISPLAY="${DISPLAY_VALUE}"
  export QT_X11_NO_MITSHM=1
  export XAUTHORITY=/root/.Xauthority
  export __GLX_VENDOR_LIBRARY_NAME=nvidia

  if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_FILE}"
  fi

  xhost +local: >/dev/null 2>&1 || true
  DISPLAY="${DISPLAY_VALUE}" xsetroot -solid "#20252b" >/dev/null 2>&1 || true

  start_window_manager
  start_x11vnc
  start_novnc

  cat <<EOF

Browser desktop is ready.

  HTTP:   http://127.0.0.1:${HTTP_PORT}/vnc.html
  DISPLAY: ${DISPLAY_VALUE}
  VNC:    localhost:${VNC_PORT}
  Backend: NVIDIA Xorg only

Use this in terminals:

  source ${PROFILE_ENV_FILE}
  source ${DESKTOP_ENV_FILE}

Verify:

  glxinfo -B
  nvidia-smi

Logs:

  Xorg:   ${XORG_LOG}
  GLX:    ${GLXINFO_LOG}
  noVNC:  ${NOVNC_LOG}
  x11vnc: ${X11VNC_LOG}

EOF
}

main() {
  source_ros_if_present
  write_base_env_file

  if [[ "${FIRST}" -eq 1 ]]; then
    install_first_run_packages
    install_pixi_if_needed
    install_node_and_codex_if_needed
    configure_sshd
    ensure_sshd_running
    setup_git_and_github_ssh
  else
    configure_sshd
    ensure_sshd_running
  fi

  if [[ "${START_DESKTOP}" -eq 1 ]]; then
    ensure_desktop_stack
  else
    log "Desktop startup skipped because --no-desktop was provided."
  fi

  log "Done."

  cat <<EOF

Safe ~/.bashrc lines only:

  source ${PROFILE_ENV_FILE} 2>/dev/null || true
  source ${DESKTOP_ENV_FILE} 2>/dev/null || true

Do NOT put this in ~/.bashrc:

  ./aic_runpod_setup.bash
  ./aic_runpod_setup.bash --first

EOF
}

main "$@"
