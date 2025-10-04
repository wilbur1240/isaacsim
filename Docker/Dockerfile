# ------------------------------------
# Base image: Ubuntu 22.04
# ------------------------------------
    FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

    ENV DEBIAN_FRONTEND=noninteractive
    # ------------------------------------
    # Environment variables
    # ------------------------------------
    ENV ROS_DISTRO=humble
    ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    ENV NVIDIA_VISIBLE_DEVICES=all
    ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

    ENV OMNI_KIT_ACCEPT_EULA=YES
    ENV OMNI_KIT_ALLOW_ROOT=1

    # ------------------------------------
    # User
    # ------------------------------------
    ENV NB_USER=arg \
        NB_UID=1000
    
    RUN echo 'Etc/UTC' > /etc/timezone && \
        ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime

    RUN adduser --disabled-password \
        --gecos "Default user" \
        --uid ${NB_UID} \
        ${NB_USER} 
    
    RUN echo "root:root" | chpasswd \
        && echo "${NB_USER}:111111" | chpasswd
    
    ENV HOME=/home/${NB_USER}
    
    # ------------------------------------
    # Install system dependencies and GCC 12 
    # ------------------------------------
    RUN apt-get update && \
        apt-get install -q -y --no-install-recommends \
        # --- Security & Repo Management ---
        ca-certificates \               
        dirmngr \                        
        gnupg2 \                       
        gnupg \                         
        lsb-release \                    
        software-properties-common \     
        curl \                           
        wget \                           
        locales \                        
    
        # --- Compiler / Build Toolchain ---
        gcc-12 g++-12 libgcc-12-dev \    
        gdb \                           
        make \                     
        ninja-build \               
        cmake \                        
        build-essential \             
        swig \                        
        libtool-bin \              
        pkg-config \                    
        ccache \                        
        rapidjson-dev \                 
        zip \                          
    
        # --- Python & Development ---
        python3-numpy \                 
        python3-dev \                  
        python3-pyparsing \            
        python3-serial \                
        python-is-python3 \      
        libpython3-stdlib \        
        pip \                           
        git \                       
        vim less \                   
    
        # --- Networking / System Tools ---
        socat \                         
        net-tools \                     
        sudo \                         
        tzdata \                    
    
        # --- Graphics / Rendering (Isaac Sim GUI) ---
        libvulkan1 vulkan-tools \ 
        libx11-6 libxext6 libxrandr2 \   
        libxinerama1 libxi6 \            
        libxxf86vm1 libxfixes3 \       
        libxkbcommon0 libxrender1 \     
        libxtst6 libxcursor1 \          
        libegl1 libgl1 \                 
        libasound2 libpulse0 \       
        libdbus-1-3 \                    
        libfreetype6-dev libfontconfig1-dev \ 
    
        # --- Vision & Sensor Dependencies ---
        libopencv-dev \                  
        gstreamer1.0-plugins-bad \       
        gstreamer1.0-libav \             
        gstreamer1.0-gl \              
        libcgal-dev \                    
        libfftw3-dev && \ 
        rm -rf /var/lib/apt/lists/*
    
    # ------------------------------------
    # Install ROS 2 Humble
    # ------------------------------------
    RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    
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
    
    # Initialize rosdep
    RUN rosdep init && rosdep update && \
        bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash"
    
    # ------------------------------------
    # Conda env
    # ------------------------------------
    ENV CONDA_DIR=/opt/CONDA_DIR
    RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
        bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm -f /tmp/miniconda.sh
    ENV PATH=$CONDA_DIR/bin:$PATH

    RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
        conda update -y -n base -c defaults conda && \
        conda create -y -n isaac python=3.11 && \
        conda clean -afy

    # RUN source $CONDA_DIR/bin/activate isaac

    # ------------------------------------
    # Install Isaac Sim environment via pip inside conda env
    # ------------------------------------
    RUN ${CONDA_DIR}/bin/conda run -n isaac pip install --upgrade pip setuptools wheel && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install \
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
        future \
        "isaacsim[all,extscache,rl,ros2]==5.0.0" --extra-index-url https://pypi.nvidia.com
    
    # ------------------------------------
    # Reinforcement Learning / ML dependencies
    # ------------------------------------
    RUN ${CONDA_DIR}/bin/conda run -n isaac pip install --upgrade pip && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install \ 
        torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

    RUN ${CONDA_DIR}/bin/conda run -n isaac pip install --upgrade pip && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install --no-cache-dir --ignore-installed PyYAML && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install \
        torchaudio \
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

    RUN ${CONDA_DIR}/bin/conda run -n isaac pip install --no-cache-dir \ 
        numpy==1.26.4 --force-reinstall
    
    # ------------------------------------
    # IsaacLab Installation
    # ------------------------------------
    RUN git clone https://github.com/isaac-sim/IsaacLab.git ${HOME}/IsaacLab && \ 
        cd ${HOME}/IsaacLab && \ 
        git checkout v2.2.0

    RUN ${CONDA_DIR}/bin/conda run -n isaac pip install --upgrade pip && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install -e ${HOME}/IsaacLab/source/isaaclab && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install -e ${HOME}/IsaacLab/source/isaaclab_tasks && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install -e ${HOME}/IsaacLab/source/isaaclab_assets && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install -e "${HOME}/IsaacLab/source/isaaclab_rl[all]" && \
        ${CONDA_DIR}/bin/conda run -n isaac pip install -e "${HOME}/IsaacLab/source/isaaclab_mimic[all]"