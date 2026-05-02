FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y \
    git \
    python3-colcon-common-extensions \
    ros-kilted-ament-cmake \
  && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY aic_assets /ws_aic/src/aic/aic_assets
COPY aic_description /ws_aic/src/aic/aic_description
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# Add other dependencies

SHELL ["/bin/bash", "-c"]
RUN cd /ws_aic/src/aic \
    && source /opt/ros/kilted/setup.bash \
    && colcon build \
        --packages-select aic_assets aic_description \
        --packages-ignore aic_gazebo aic_engine_interfaces \
        --executor sequential

RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --locked

RUN cd /ws_aic/src/aic \
    && pixi run python -m pip install --no-cache-dir tesseract-robotics==0.5.1

WORKDIR /ws_aic/src/aic
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# start the model
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.CheatCode", "-p", "use_sim_time:=true"]
