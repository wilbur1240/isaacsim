source /opt/ros/humble/setup.bash

# Check if we're already in the isaac environment
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "isaac" ]]; then
    source activate isaac
fi