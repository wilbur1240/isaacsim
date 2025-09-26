# Choose a recent CUDA+cudnn devel image on Ubuntu 22.04 (glibc 2.35+)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video

# Accept Omniverse/Isaac Sim EULA non-interactively
# (you can remove this and accept at first run if you prefer)
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV OMNI_KIT_ALLOW_ROOT=1

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    NB_USER=arg \
    NB_UID=1001

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime

###################################### user #####################################
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} 

RUN echo "root:root" | chpasswd \
    && echo "${NB_USER}:111111" | chpasswd

ENV HOME=/home/${NB_USER}

###################################### basic #####################################
RUN rm -f /etc/apt/sources.list.d/*sury*.list \
    && apt-get update \
    && apt-get install -y libbrotli-dev

RUN apt-get update && \
    apt-get install -q -y --no-install-recommends \
    ca-certificates \
    ccache \
    g++ \
    gcc \
    gdb \
    gawk \
    make \
    ninja-build \
    python3-numpy \
    python3-dev \
    python3-pyparsing \
    python3-serial \
    python-is-python3 \
    libpython3-stdlib \
    libtool-bin \
    zip \
    default-jre \
    socat \
    tzdata \
    pip \
    dirmngr \
    gnupg2 \
    git \
    gitk git-gui \
    vim \
    less \
    curl \
    wget \
    sudo \
    cmake \
    swig \
    net-tools \
    lsb-release \
    gnupg \
    locales \
    build-essential \
    libcgal-dev \
    libfftw3-dev \
    rapidjson-dev \
    pkg-config \
    libboost-system-dev \
    libopencv-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    software-properties-common \
    libfreetype6-dev \
    libfontconfig1-dev \
    libvulkan1 vulkan-tools \
    libx11-6 libxext6 libxrandr2 libxinerama1 libxi6 libxxf86vm1 \
    libxfixes3 libxkbcommon0 libxrender1 libxtst6 libxcursor1 \
    libasound2 libpulse0 libdbus-1-3 \
    libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

###################################### python pkg###############################
ENV CONDA_DIR=/opt/CONDA_DIR
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm -f /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -y -n base -c defaults conda && \
    conda create -y -n isaac python=3.11 && \
    conda clean -afy
ENV CONDA_DEFAULT_ENV=isaac
ENV PATH=$CONDA_DIR/envs/isaac/bin:$PATH

RUN apt-get update && apt-get upgrade pip -y && \
    pip install --upgrade pip && \
    pip install \
    numpy \
    scipy \
    matplotlib \
    scikit-learn \
    scikit-image \
    pandas \
    opencv-python \
    opencv-python-headless \
    pyyaml \
    pyquaternion \
    pyserial \
    pexpect \
    Pillow \
    tensorboard \
    tensorflow \
    future

# Install Isaac Sim via pip (bundles: all, extscache, rl, ros2)
# Note: first run may still download some Omniverse extensions if not in cache.
RUN python -m pip install --upgrade pip && \
    pip install "isaacsim[all,extscache,rl,ros2]==5.0.0" --extra-index-url https://pypi.nvidia.com

RUN python - <<'PY'
import site, os
try:
    sitepkgs = site.getsitepackages()[0]
except Exception:
    sitepkgs = os.path.dirname(site.__file__)
sc = os.path.join(sitepkgs, "sitecustomize.py")
with open(sc, "w") as f:
    f.write(r'''
# sitecustomize: compatibility shim for prebundled packages expecting np.cumproduct
try:
    import numpy as _np
    if not hasattr(_np, "cumproduct"):
        setattr(_np, "cumproduct", getattr(_np, "cumprod", None))
except Exception:
    pass
''')
print("wrote", sc)
PY

################################## torch gymnasium sb3 #####################################
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --ignore-installed PyYAML && \
    python3 -m pip install \
    torch \
    torchaudio \
    torchvision \
    gymnasium \
    gymnasium[box2d] \
    stable-baselines3 \
    openai \
    trajectopy \
    huggingface-sb3 \
    huggingface_hub \
    transformers \
    datasets \
    hf_transfer \
    opencv-python-headless \
    timm

# Add ROS2 apt repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-ros2-control \
    ros-${ROS_DISTRO}-ros2-controllers \
    ros-${ROS_DISTRO}-joy* \
    ros-${ROS_DISTRO}-joint-state-publisher \
    ros-${ROS_DISTRO}-mavros \
    ros-${ROS_DISTRO}-mavros-extras \
    ros-${ROS_DISTRO}-mavros-msgs \
    ros-${ROS_DISTRO}-rosbridge-server \
    ros-${ROS_DISTRO}-gps-msgs \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-ament-cmake \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-tf2-msgs \
    ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-rviz-rendering \
    ros-${ROS_DISTRO}-rcl-interfaces \
    ros-${ROS_DISTRO}-urdf \
    ros-${ROS_DISTRO}-geographic-msgs \
    ros-${ROS_DISTRO}-topic-tools \
    ros-${ROS_DISTRO}-launch-pytest \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    python3-rosdep \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-argcomplete && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Init rosdep
RUN rosdep init && rosdep update || true

# Auto-source ROS2
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

# Isaac Sim ↔ ROS bridge deps (handy for messaging)
RUN apt-get update && apt-get install -y \
    python3-pip \
    && pip install --no-cache-dir \
    transforms3d \
    numpy \
    rosbags \
    && rm -rf /var/lib/apt/lists/*

