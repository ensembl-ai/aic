#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_ENV_FILE="/etc/profile.d/aic_base_setup_env.sh"
PIXI_BIN_DIR="/root/.pixi/bin"
PIXIBIN="${PIXI_BIN_DIR}/pixi"

export DEBIAN_FRONTEND=noninteractive

persist_env() {
  cat > "${PROFILE_ENV_FILE}" <<ENVEOF
export PATH="${PIXI_BIN_DIR}:\$PATH"
export QT_X11_NO_MITSHM=1
export XAUTHORITY=/root/.Xauthority
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export AIC_ENABLE_BROWSER_DESKTOP=1
export AIC_BROWSER_DESKTOP_HTTP_PORT=6080
export AIC_BROWSER_DESKTOP_NVIDIA_BUS_ID="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n1)"
ENVEOF
}

# Source ROS for the current setup process.
source /opt/ros/kilted/setup.bash

set -u

# Pixi
if [[ ! -x "${PIXIBIN}" ]]; then
  curl -fsSL https://pixi.sh/install.sh | sh
fi
export PATH="${PIXI_BIN_DIR}:$PATH"

# Install once (will skip if already installed)
apt-get update
apt-get install -y iproute2 net-tools vim openssh-server

# Codex
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
npm i -g @openai/codex

# SSH related items
mkdir -p /var/run/sshd
echo "root:root" | chpasswd
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
grep -q '^ListenAddress 0.0.0.0$' /etc/ssh/sshd_config || echo "ListenAddress 0.0.0.0" >> /etc/ssh/sshd_config
/usr/sbin/sshd

persist_env

if [[ "${AIC_ENABLE_BROWSER_DESKTOP:-1}" != "0" && "${AIC_ENABLE_BROWSER_DESKTOP:-1}" != "false" ]]; then
  BROWSER_DESKTOP_HTTP_PORT="${AIC_BROWSER_DESKTOP_HTTP_PORT:-6080}"
  "${SCRIPT_DIR}/browser_desktop_setup.bash" "${BROWSER_DESKTOP_HTTP_PORT}"
fi

git config --global user.name "rishimalhan"
git config --global user.email "rmalhan0112@gmail.com"
git config --global init.defaultBranch main

mkdir -p ~/.ssh
chmod 700 ~/.ssh

if [[ -f ~/.ssh/id_ed25519 ]]; then
  echo "SSH key ~/.ssh/id_ed25519 already exists; reusing it."
else
  ssh-keygen -t ed25519 -C "rmalhan0112@gmail.com" -f ~/.ssh/id_ed25519 -N ""
  echo "New SSH key created. Add this to GitHub:"
fi
cat ~/.ssh/id_ed25519.pub

ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
chmod 644 ~/.ssh/known_hosts

git config --global core.sshCommand "ssh -i ~/.ssh/id_ed25519"

# Running a script cannot modify the parent shell, so replace this process
# with a fresh login shell that picks up /etc/profile.d/*.sh and keeps the
# current SSH session interactive.
if [[ -t 0 && -t 1 && "${AIC_SKIP_LOGIN_SHELL:-0}" != "1" ]]; then
  echo "Opening a fresh login shell with the configured environment..."
  exec bash -l
fi
