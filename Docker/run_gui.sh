#!/usr/bin/env bash

ARGS=("$@")

REPOSITORY="wilbur1240/isaacsim"
TAG="5.0.0-ros2-humble"

USER_NAME="arg"
REPO_NAME="isaacsim"
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
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -e REPO_NAME=$REPO_NAME \
    -e HOME=/home/${USER_NAME} \
    -e OPENAI_API_KEY=$OPENAI_API_KEY\
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e XDG_RUNTIME_DIR=/tmp/runtime-${USER_NAME} \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.cache/ov:/root/.cache/ov \
    -v /tmp/runtime-${USER_NAME}:/tmp/runtime-${USER_NAME} \
    -v "$XAUTH:$XAUTH" \
    -v "/home/${USER}/${REPO_NAME}:/home/${USER_NAME}/${REPO_NAME}" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev:/dev" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v "/usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json" \
    --name "${CONTAINER_NAME}" \
    "${IMG}" \
    bash
