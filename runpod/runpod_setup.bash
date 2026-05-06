#!/usr/bin/env bash
# aic_runpod_session.bash
#
# Intended usage:
#
#   # 1) First-time setup on a fresh container.
#   # Installs apt packages, Pixi, Node.js/Codex, configures key-only SSH,
#   # sets Git/GitHub SSH key, and sources ROS/Pixi into this terminal.
#   source ./aic_runpod_session.bash --first
#
#   # 2) Activate this SSH terminal/session only.
#   # Sources ROS and adds Pixi/Codex to PATH. Does not start display.
#   source ./aic_runpod_session.bash
#
#   # 3) Start or reuse remote display for this terminal.
#   # Starts NVIDIA Xorg + openbox + x11vnc + noVNC.
#   # Exports DISPLAY/XAUTHORITY in this terminal.
#   source ./aic_runpod_session.bash --display
#
#   # 4) Restart remote display cleanly for this terminal.
#   # Use when Gazebo/RViz/noVNC/display becomes stale or weird.
#   source ./aic_runpod_session.bash --restart-display
#
#   # 5) Stop remote display stack.
#   # Stops Xorg/openbox/x11vnc/noVNC. Does not need source.
#   ./aic_runpod_session.bash --stop-display
#
#   # 6) Use alternate noVNC browser port if 6080 is occupied.
#   source ./aic_runpod_session.bash --display --http-port 6081
#
#   # 7) Use alternate VNC backend port if 5901 is occupied.
#   source ./aic_runpod_session.bash --display --vnc-port 5902
#
#   # 8) Use a different X display number if :1 is occupied/stale.
#   source ./aic_runpod_session.bash --display --display-num 2
#
#   # 9) First-time setup but skip Codex installation.
#   source ./aic_runpod_session.bash --first --no-codex
#
#   # 10) Show available arguments.
#   ./aic_runpod_session.bash --help
#
# Notes:
#   - Use `source` when you want ROS/Pixi/DISPLAY variables to persist
#     in the current SSH terminal.
#   - `./script` runs in a child process, so exported environment variables
#     do not persist in your current shell.
#   - Display is not started unless --display or --restart-display is passed.
#   - No ~/.bashrc or /etc/profile.d shell hooks are written.
#   - No login shell is automatically opened.
#
# Browser desktop access after --display:
#   ssh -L 6080:127.0.0.1:6080 root@YOUR_RUNPOD_HOST -p YOUR_SSH_PORT
#   open http://127.0.0.1:6080/vnc.html
#
# No ~/.bashrc edits.
# No /etc/profile.d writes.
# No exec bash -l.
# ROS and Pixi/Codex PATH are activated in the current shell when sourced.
# This script is often sourced to activate ROS/Pixi/DISPLAY.
# Therefore, never leak strict shell options into the user's interactive shell.
AIC_SCRIPT_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  AIC_SCRIPT_SOURCED=1
fi

if [[ "${AIC_SCRIPT_SOURCED}" -eq 0 ]]; then
  set -Eeuo pipefail
else
  set +e
  set +u
  set +o pipefail 2>/dev/null || true
fi

FIRST=0
DISPLAY_ENABLE=0
RESTART_DISPLAY=0
STOP_DISPLAY=0
INSTALL_CODEX=1

ROS_SETUP="${ROS_SETUP:-/opt/ros/kilted/setup.bash}"

PIXI_BIN_DIR="/root/.pixi/bin"
PIXIBIN="${PIXI_BIN_DIR}/pixi"

DISPLAY_NUM="${AIC_BROWSER_DESKTOP_DISPLAY_NUM:-1}"
DISPLAY_VALUE=":${DISPLAY_NUM}"
HTTP_PORT="${AIC_BROWSER_DESKTOP_HTTP_PORT:-6080}"
VNC_PORT="${AIC_BROWSER_DESKTOP_VNC_PORT:-5901}"

SCREEN_WIDTH="${AIC_BROWSER_DESKTOP_WIDTH:-1920}"
SCREEN_HEIGHT="${AIC_BROWSER_DESKTOP_HEIGHT:-1080}"
SCREEN_DEPTH="${AIC_BROWSER_DESKTOP_DEPTH:-24}"

XAUTHORITY_FILE="/tmp/aic-Xauthority-${DISPLAY_NUM}"
XORG_CONFIG_FILE="/tmp/aic-browser-desktop-xorg-${DISPLAY_NUM}.conf"

NVIDIA_XORG_MODULE_DIR=""
NVIDIA_EGL_VENDOR_FILE=""
NVIDIA_DEVICE_BUS_ID=""

log() { echo "[aic-session] $*" >&2; }
die() { echo "[aic-session:ERROR] $*" >&2; return 1 2>/dev/null || exit 1; }

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

usage() {
  cat >&2 <<EOF
Usage:
  source ./aic_runpod_session.bash --first
  source ./aic_runpod_session.bash
  source ./aic_runpod_session.bash --display
  source ./aic_runpod_session.bash --restart-display
  ./aic_runpod_session.bash --stop-display

Flags:
  --first              One-time install/config setup.
  --display            Start/reuse NVIDIA display and activate DISPLAY in this shell.
  --restart-display    Restart NVIDIA display and activate DISPLAY in this shell.
  --stop-display       Stop display stack.
  --no-codex           Skip Codex install during --first.
  --http-port PORT     noVNC HTTP port. Default: 6080.
  --vnc-port PORT      VNC port. Default: 5901.
  --display-num NUM    X display number. Default: 1.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --first) FIRST=1; shift ;;
    --display) DISPLAY_ENABLE=1; shift ;;
    --restart-display) DISPLAY_ENABLE=1; RESTART_DISPLAY=1; shift ;;
    --stop-display) STOP_DISPLAY=1; shift ;;
    --no-codex) INSTALL_CODEX=0; shift ;;
    --http-port) HTTP_PORT="${2:?Missing value}"; shift 2 ;;
    --vnc-port) VNC_PORT="${2:?Missing value}"; shift 2 ;;
    --display-num)
      DISPLAY_NUM="${2:?Missing value}"
      DISPLAY_VALUE=":${DISPLAY_NUM}"
      XAUTHORITY_FILE="/tmp/aic-Xauthority-${DISPLAY_NUM}"
      XORG_CONFIG_FILE="/tmp/aic-browser-desktop-xorg-${DISPLAY_NUM}.conf"
      shift 2
      ;;
    --help|-h) usage; return 0 2>/dev/null || exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

require_root() {
  [[ "$(id -u)" -eq 0 ]] || die "Run/source this as root inside the RunPod container."
}

activate_session_env() {
  export PATH="${PIXI_BIN_DIR}:$PATH"
  export QT_X11_NO_MITSHM=1

  if [[ -f "${ROS_SETUP}" ]]; then
    local had_nounset=0
    case "$-" in
      *u*) had_nounset=1; set +u ;;
    esac

    # shellcheck source=/dev/null
    source "${ROS_SETUP}"

    if [[ "${had_nounset}" -eq 1 ]]; then
      set -u
    fi

    log "ROS sourced in this shell: ${ROS_SETUP}"
  else
    log "ROS setup not found, skipped: ${ROS_SETUP}"
  fi

  if command -v codex >/dev/null 2>&1; then
    log "Codex available: $(command -v codex)"
  fi
}

install_first_run_packages() {
  export DEBIAN_FRONTEND=noninteractive
  log "Installing base packages."

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

  log "Base packages installed."
}

install_pixi_if_needed() {
  if [[ ! -x "${PIXIBIN}" ]]; then
    log "Installing Pixi."
    curl -fsSL https://pixi.sh/install.sh | sh
  else
    log "Pixi already installed."
  fi

  export PATH="${PIXI_BIN_DIR}:$PATH"
}

install_node_and_codex_if_needed() {
  [[ "${INSTALL_CODEX}" -eq 1 ]] || { log "Codex install skipped."; return 0; }

  local node_major="0"

  if command -v node >/dev/null 2>&1; then
    node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
  fi

  if [[ "${node_major}" -lt 22 ]] || ! command -v npm >/dev/null 2>&1; then
    log "Installing Node.js 22."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
  else
    log "Node.js >=22 already installed."
  fi

  if ! command -v codex >/dev/null 2>&1; then
    log "Installing Codex CLI."
    npm i -g @openai/codex
  else
    log "Codex already installed."
  fi
}

set_sshd_option() {
  local key="$1"
  local value="$2"
  local file="/etc/ssh/sshd_config"

  [[ -f "${file}" ]] || die "Missing ${file}. Run --first."

  if grep -Eq "^[#[:space:]]*${key}[[:space:]]+" "${file}"; then
    sed -ri "s|^[#[:space:]]*${key}[[:space:]].*|${key} ${value}|g" "${file}"
  else
    echo "${key} ${value}" >> "${file}"
  fi
}

configure_sshd_key_only() {
  mkdir -p /var/run/sshd /root/.ssh
  chmod 700 /root/.ssh

  set_sshd_option "PermitRootLogin" "prohibit-password"
  set_sshd_option "PasswordAuthentication" "no"
  set_sshd_option "PubkeyAuthentication" "yes"
  set_sshd_option "KbdInteractiveAuthentication" "no"
  set_sshd_option "ChallengeResponseAuthentication" "no"

  grep -q '^ListenAddress 0.0.0.0$' /etc/ssh/sshd_config || \
    echo "ListenAddress 0.0.0.0" >> /etc/ssh/sshd_config

  /usr/sbin/sshd -t
  log "sshd configured key-only."
}

ensure_sshd_running() {
  if ss -ltnp 2>/dev/null | grep -qE 'LISTEN.+:22[[:space:]]'; then
    log "sshd already running."
  else
    log "Starting sshd."
    /usr/sbin/sshd
  fi
}

setup_git_and_github_ssh() {
  git config --global user.name "rishimalhan"
  git config --global user.email "rmalhan0112@gmail.com"
  git config --global init.defaultBranch main

  mkdir -p /root/.ssh
  chmod 700 /root/.ssh

  if [[ ! -f /root/.ssh/id_ed25519 ]]; then
    ssh-keygen -t ed25519 -C "rmalhan0112@gmail.com" -f /root/.ssh/id_ed25519 -N ""
  fi

  ssh-keygen -F github.com >/dev/null 2>&1 || \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null

  chmod 644 /root/.ssh/known_hosts
  git config --global core.sshCommand "ssh -i /root/.ssh/id_ed25519"

  log "GitHub public key:"
  cat /root/.ssh/id_ed25519.pub >&2
}

resolve_existing_path() {
  local candidate
  for candidate in "$@"; do
    [[ -e "${candidate}" ]] && { echo "${candidate}"; return 0; }
  done
  return 1
}

find_nvidia_xorg_module_dir() {
  local driver_path
  driver_path="$(
    resolve_existing_path \
      /usr/lib/x86_64-linux-gnu/nvidia/xorg/drivers/nvidia_drv.so \
      /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so \
      /usr/lib64/xorg/modules/drivers/nvidia_drv.so \
      /usr/lib/xorg/modules/drivers/nvidia_drv.so
  )" || return 1

  case "${driver_path}" in
    */nvidia/xorg/drivers/nvidia_drv.so) dirname "$(dirname "${driver_path}")" ;;
    */xorg/modules/drivers/nvidia_drv.so) dirname "$(dirname "${driver_path}")" ;;
    *) dirname "${driver_path}" ;;
  esac
}

find_nvidia_egl_vendor_file() {
  [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && {
    echo /usr/share/glvnd/egl_vendor.d/10_nvidia.json
    return 0
  }

  [[ -f /etc/glvnd/egl_vendor.d/10_nvidia.json ]] && {
    echo /etc/glvnd/egl_vendor.d/10_nvidia.json
    return 0
  }

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
  local raw_bus_id
  raw_bus_id="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  [[ -n "${raw_bus_id}" ]] || die "Could not detect NVIDIA PCI bus ID."
  normalize_xorg_bus_id "${raw_bus_id}"
}

resolve_nvidia_runtime() {
  NVIDIA_XORG_MODULE_DIR="$(find_nvidia_xorg_module_dir)" || die "Missing nvidia_drv.so."
  NVIDIA_EGL_VENDOR_FILE="$(find_nvidia_egl_vendor_file || true)"
  NVIDIA_DEVICE_BUS_ID="$(detect_nvidia_bus_id)"

  log "NVIDIA Xorg module dir: ${NVIDIA_XORG_MODULE_DIR}"
  log "NVIDIA BusID: ${NVIDIA_DEVICE_BUS_ID}"
}

validate_nvidia_runtime() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi missing."
  nvidia-smi >/dev/null 2>&1 || die "nvidia-smi failed."

  command -v Xorg >/dev/null 2>&1 || die "Xorg missing. Run --first."
  command -v glxinfo >/dev/null 2>&1 || die "glxinfo missing. Run --first."
  command -v xauth >/dev/null 2>&1 || die "xauth missing. Run --first."
  command -v xdpyinfo >/dev/null 2>&1 || die "xdpyinfo missing. Run --first."

  resolve_nvidia_runtime
}

write_nvidia_xorg_config() {
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
  touch "${XAUTHORITY_FILE}"
  chmod 600 "${XAUTHORITY_FILE}"

  XAUTHORITY="${XAUTHORITY_FILE}" xauth remove "${DISPLAY_VALUE}" >/dev/null 2>&1 || true
  XAUTHORITY="${XAUTHORITY_FILE}" xauth add "${DISPLAY_VALUE}" . "$(mcookie)" >/dev/null
}

run_glxinfo() {
  local -a env_args=(
    DISPLAY="${DISPLAY_VALUE}"
    XAUTHORITY="${XAUTHORITY_FILE}"
    __GLX_VENDOR_LIBRARY_NAME=nvidia
  )

  [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]] && \
    env_args+=("__EGL_VENDOR_LIBRARY_FILENAMES=${NVIDIA_EGL_VENDOR_FILE}")

  env "${env_args[@]}" glxinfo -B
}

display_responds() {
  DISPLAY="${DISPLAY_VALUE}" XAUTHORITY="${XAUTHORITY_FILE}" xdpyinfo >/dev/null 2>&1
}

display_uses_nvidia_glx() {
  local out
  out="$(run_glxinfo 2>&1)" || return 1
  [[ "${out}" == *"OpenGL vendor string: NVIDIA Corporation"* ]] || \
  [[ "${out}" == *"OpenGL renderer string: NVIDIA"* ]]
}

start_detached_process() {
  local pidfile="$1"
  shift
  nohup "$@" >/dev/null 2>&1 </dev/null &
  echo "$!" > "${pidfile}"
}

stop_display_stack() {
  log "Stopping display stack."

  pkill -f "Xorg ${DISPLAY_VALUE}" >/dev/null 2>&1 || true
  pkill -f "x11vnc.*-rfbport ${VNC_PORT}" >/dev/null 2>&1 || true
  pkill -f "novnc.*${HTTP_PORT}" >/dev/null 2>&1 || true
  pkill -f "websockify.*${HTTP_PORT}" >/dev/null 2>&1 || true
  pkill -f "openbox" >/dev/null 2>&1 || true

  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
  rm -rf "/tmp/.X11-unix/X${DISPLAY_NUM}"
  rm -f "${XORG_CONFIG_FILE}"

  log "Display stack stopped."
}

start_nvidia_xorg() {
  validate_nvidia_runtime
  ensure_xauthority
  write_nvidia_xorg_config

  if display_responds; then
    if display_uses_nvidia_glx; then
      log "Existing display ${DISPLAY_VALUE} is NVIDIA-backed."
      return 0
    fi
    die "Display ${DISPLAY_VALUE} exists but is not NVIDIA GLX. Use --restart-display."
  fi

  log "Starting NVIDIA Xorg on ${DISPLAY_VALUE}."

  start_detached_process "/tmp/aic-xorg-${DISPLAY_NUM}.pid" \
    Xorg \
      "${DISPLAY_VALUE}" \
      -config "${XORG_CONFIG_FILE}" \
      -logfile /dev/null \
      -noreset \
      +extension GLX \
      +extension RANDR \
      +extension RENDER \
      -isolateDevice "${NVIDIA_DEVICE_BUS_ID}"

  for _ in $(seq 1 20); do
    if display_responds && display_uses_nvidia_glx; then
      log "NVIDIA Xorg ready."
      return 0
    fi
    sleep 1
  done

  die "NVIDIA Xorg did not validate."
}

find_novnc_proxy() {
  [[ -x /usr/share/novnc/utils/novnc_proxy ]] && {
    echo /usr/share/novnc/utils/novnc_proxy
    return 0
  }

  [[ -x /usr/share/novnc/utils/launch.sh ]] && {
    echo /usr/share/novnc/utils/launch.sh
    return 0
  }

  return 1
}

start_window_manager() {
  if pgrep -f "openbox" >/dev/null 2>&1; then
    log "openbox already running."
    return 0
  fi

  start_detached_process "/tmp/aic-openbox-${DISPLAY_NUM}.pid" \
    env DISPLAY="${DISPLAY_VALUE}" XAUTHORITY="${XAUTHORITY_FILE}" \
    dbus-launch --exit-with-session openbox
}

start_x11vnc() {
  if pgrep -f "x11vnc.*-rfbport ${VNC_PORT}" >/dev/null 2>&1; then
    log "x11vnc already running."
    return 0
  fi

  start_detached_process "/tmp/aic-x11vnc-${VNC_PORT}.pid" \
    x11vnc \
      -display "${DISPLAY_VALUE}" \
      -auth "${XAUTHORITY_FILE}" \
      -rfbport "${VNC_PORT}" \
      -forever \
      -shared \
      -localhost \
      -nopw \
      -xkb \
      -noxdamage \
      -quiet
}

start_novnc() {
  local novnc_proxy
  novnc_proxy="$(find_novnc_proxy)" || die "noVNC proxy not found."

  if pgrep -f "websockify.*${HTTP_PORT}" >/dev/null 2>&1 || \
     pgrep -f "novnc.*${HTTP_PORT}" >/dev/null 2>&1; then
    log "noVNC already running."
    return 0
  fi

  start_detached_process "/tmp/aic-novnc-${HTTP_PORT}.pid" \
    "${novnc_proxy}" \
      --listen "127.0.0.1:${HTTP_PORT}" \
      --vnc "127.0.0.1:${VNC_PORT}"
}

activate_display_env() {
  export DISPLAY="${DISPLAY_VALUE}"
  export XAUTHORITY="${XAUTHORITY_FILE}"
  export QT_X11_NO_MITSHM=1
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export AIC_BROWSER_DESKTOP_HTTP_PORT="${HTTP_PORT}"
  export AIC_BROWSER_DESKTOP_VNC_PORT="${VNC_PORT}"
  export AIC_BROWSER_DESKTOP_DISPLAY_NUM="${DISPLAY_NUM}"

  if [[ -n "${NVIDIA_EGL_VENDOR_FILE}" ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_FILE}"
  fi

  log "DISPLAY activated in this shell: ${DISPLAY}"
}

start_display_stack() {
  [[ "${RESTART_DISPLAY}" -eq 1 ]] && stop_display_stack

  start_nvidia_xorg
  activate_display_env

  xhost +SI:localuser:root >/dev/null 2>&1 || true
  xsetroot -solid "#20252b" >/dev/null 2>&1 || true

  start_window_manager
  start_x11vnc
  start_novnc

  log "Display ready."
  log "Tunnel: ssh -L ${HTTP_PORT}:127.0.0.1:${HTTP_PORT} root@HOST -p PORT"
  log "Open: http://127.0.0.1:${HTTP_PORT}/vnc.html"
}

main() {
  require_root

  # This is now done every time the script is run/sourced.
  # That means Codex PATH and ROS are available in the current session.
  activate_session_env

  if [[ "${FIRST}" -eq 1 ]]; then
    install_first_run_packages
    install_pixi_if_needed
    install_node_and_codex_if_needed
    configure_sshd_key_only
    ensure_sshd_running
    setup_git_and_github_ssh

    # Re-activate after install so pixi/codex are available immediately.
    activate_session_env

    log "First-time setup complete."
  else
    if command -v sshd >/dev/null 2>&1; then
      configure_sshd_key_only
      ensure_sshd_running
    fi
  fi

  if [[ "${STOP_DISPLAY}" -eq 1 ]]; then
    stop_display_stack
    return 0 2>/dev/null || exit 0
  fi

  if [[ "${DISPLAY_ENABLE}" -eq 1 ]]; then
    start_display_stack
  fi

  log "Done."
}

main "$@"
