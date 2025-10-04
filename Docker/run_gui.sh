#!/usr/bin/env bash

ARGS=("$@")

REPOSITORY="wilbur1240/isaacsim"
TAG="5.0.0-ros2-humble"

IMG="${REPOSITORY}:${TAG}"

USER_NAME="arg"
REPO_NAME="isaacsim-arg"
ISAAC_LOCAL_DIR="$HOME/isaacsim-5.0.0"
CONTAINER_NAME="isaac-5.0.0-ros2-humble-gui"

CONTAINER_ID=$(docker ps -aqf "ancestor=${IMG}")
if [ $CONTAINER_ID ]; then
  echo "Attach to docker container $CONTAINER_ID"
  xhost +
  docker exec --privileged -e DISPLAY=${DISPLAY} -e LINES="$(tput lines)" -it ${CONTAINER_ID} bash
  xhost -
  return
fi

xhost +local:root  # allow X11

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
  xauth_list=$(xauth nlist $DISPLAY)
  xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
  if [ ! -z "$xauth_list" ]; then
    echo "$xauth_list" | xauth -f $XAUTH nmerge -
  else
    touch $XAUTH
  fi
  chmod a+r $XAUTH
fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
  echo "[$XAUTH] was not properly created. Exiting..."
  exit 1
fi

docker run \
    -it \
    --rm \
    --user "root:root" \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    --privileged \
    --security-opt seccomp=unconfined \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -e REPO_NAME=$REPO_NAME \
    -e HOME=/home/${USER_NAME} \
    -e XDG_RUNTIME_DIR=/tmp/runtime-${USER_NAME} \
    -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    -v /usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /tmp/runtime-${USER_NAME}:/tmp/runtime-${USER_NAME} \
    -v "$XAUTH:$XAUTH" \
    -v "/home/${USER}/${REPO_NAME}:/home/${USER_NAME}/${REPO_NAME}" \
    -v "${ISAAC_LOCAL_DIR}":/home/${USER_NAME}/isaac-sim:rw \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    --workdir "/home/${USER_NAME}/${REPO_NAME}" \
    --name "${CONTAINER_NAME}" \
    "${IMG}" \
    bash
