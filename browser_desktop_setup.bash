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
XORG_CONFIG_DIR="/etc/X11/xorg.conf.d"
NVIDIA_XORG_CONFIG="${XORG_CONFIG_DIR}/99-aic-nvidia-headless.conf"
DUMMY_XORG_CONFIG="${XORG_CONFIG_DIR}/99-aic-dummy-headless.conf"
ENV_FILE="/etc/profile.d/aic_browser_desktop_env.sh"

install_packages() {
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    dbus-x11 \
    mesa-utils \
    net-tools \
    novnc \
    openbox \
    python3-websockify \
    x11-utils \
    x11-xserver-utils \
    x11vnc \
    xauth \
    xserver-xorg-core \
    xserver-xorg-video-dummy
}

write_env_file() {
  cat > "${ENV_FILE}" <<ENVEOF
export DISPLAY=${DISPLAY_VALUE}
export QT_X11_NO_MITSHM=1
export XAUTHORITY=/root/.Xauthority
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENVEOF
}

write_nvidia_xorg_config() {
  mkdir -p "${XORG_CONFIG_DIR}"
  cat > "${NVIDIA_XORG_CONFIG}" <<ENVEOF
Section "ServerLayout"
    Identifier "layout"
    Screen 0 "screen"
EndSection

Section "Device"
    Identifier "device"
    Driver "nvidia"
    VendorName "NVIDIA Corporation"
    Option "AllowEmptyInitialConfiguration" "true"
    Option "UseDisplayDevice" "none"
EndSection

Section "Screen"
    Identifier "screen"
    Device "device"
    DefaultDepth ${SCREEN_DEPTH}
    SubSection "Display"
        Depth ${SCREEN_DEPTH}
        Virtual ${SCREEN_WIDTH} ${SCREEN_HEIGHT}
    EndSubSection
EndSection
ENVEOF
}

write_dummy_xorg_config() {
  mkdir -p "${XORG_CONFIG_DIR}"
  cat > "${DUMMY_XORG_CONFIG}" <<ENVEOF
Section "Monitor"
    Identifier "monitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
    Modeline "1920x1080" 172.80 1920 2040 2248 2576 1080 1081 1084 1118
EndSection

Section "Device"
    Identifier "dummy"
    Driver "dummy"
    VideoRam 256000
EndSection

Section "Screen"
    Identifier "screen"
    Device "dummy"
    Monitor "monitor"
    DefaultDepth ${SCREEN_DEPTH}
    SubSection "Display"
        Depth ${SCREEN_DEPTH}
        Modes "1920x1080"
        Virtual ${SCREEN_WIDTH} ${SCREEN_HEIGHT}
    EndSubSection
EndSection
ENVEOF
}

remove_xorg_configs() {
  rm -f "${NVIDIA_XORG_CONFIG}" "${DUMMY_XORG_CONFIG}"
}

start_xorg() {
  local mode="$1"

  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
  rm -rf "/tmp/.X11-unix/X${DISPLAY_NUM}"

  if [[ "${mode}" == "nvidia" ]]; then
    remove_xorg_configs
    write_nvidia_xorg_config
  else
    remove_xorg_configs
    write_dummy_xorg_config
  fi

  Xorg "${DISPLAY_VALUE}" -configdir "${XORG_CONFIG_DIR}" -noreset +extension GLX +extension RANDR +extension RENDER > "${XORG_LOG}" 2>&1 &
  local xorg_pid=$!

  for _ in $(seq 1 20); do
    if [[ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]]; then
      echo "${xorg_pid}" > "/tmp/aic-browser-desktop-xorg.pid"
      return 0
    fi
    sleep 1
  done

  kill "${xorg_pid}" >/dev/null 2>&1 || true
  wait "${xorg_pid}" >/dev/null 2>&1 || true
  return 1
}

ensure_xorg() {
  if [[ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]]; then
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    if start_xorg "nvidia"; then
      return 0
    fi
  fi

  start_xorg "dummy"
}

ensure_xauthority() {
  touch /root/.Xauthority
  xauth add "${DISPLAY_VALUE}" . "$(mcookie)"
}

start_window_manager() {
  if pgrep -f "openbox.*${DISPLAY_VALUE}" >/dev/null 2>&1; then
    return 0
  fi

  DISPLAY="${DISPLAY_VALUE}" XAUTHORITY=/root/.Xauthority dbus-launch --exit-with-session openbox > "${OPENBOX_LOG}" 2>&1 &
  echo "$!" > /tmp/aic-browser-desktop-openbox.pid
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

  x11vnc "${args[@]}" >/dev/null 2>&1 &
  echo "$!" > /tmp/aic-browser-desktop-x11vnc.pid
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
  if pgrep -f "novnc.*${HTTP_PORT}" >/dev/null 2>&1 || pgrep -f "websockify.*${HTTP_PORT}" >/dev/null 2>&1; then
    return 0
  fi

  local novnc_proxy
  novnc_proxy="$(find_novnc_proxy)"
  "${novnc_proxy}" --listen "${HTTP_PORT}" --vnc "localhost:${VNC_PORT}" > "${NOVNC_LOG}" 2>&1 &
  echo "$!" > /tmp/aic-browser-desktop-novnc.pid
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
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

If you start a container manually, pass:
  -e DISPLAY=${DISPLAY_VALUE}
  -e QT_X11_NO_MITSHM=1
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  -v /tmp/.X11-unix:/tmp/.X11-unix

To verify GPU rendering inside the same environment:
  glxinfo -B
  nvidia-smi
ENVEOF
}

main() {
  install_packages
  write_env_file

  export DISPLAY="${DISPLAY_VALUE}"
  export QT_X11_NO_MITSHM=1
  export XAUTHORITY=/root/.Xauthority
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

  ensure_xorg
  ensure_xauthority

  xhost +local: >/dev/null 2>&1 || true
  DISPLAY="${DISPLAY_VALUE}" xsetroot -solid "#20252b" >/dev/null 2>&1 || true

  start_window_manager
  start_x11vnc
  start_novnc
  print_summary
}

main "$@"
