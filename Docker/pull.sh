#!/usr/bin/env bash

REPOSITORY="wilbur1240/isaacsim"
TAG="5.0.0-ros2-humble"

IMG="${REPOSITORY}:${TAG}"

docker pull "${IMG}"