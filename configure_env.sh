#!/bin/bash -e

export PATH=$HOME/.local/bin:$PATH

GIT_ROOT=$(git rev-parse --show-toplevel)
if [ $? -ne 0 ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Create symbolic links for all hooks
for hook in "${GIT_ROOT}/scripts/git-hooks/"*; do
    ln -sf "$hook" "${GIT_ROOT}/.git/hooks/"
done

# Make hooks executable
for hook in "${GIT_ROOT}/.git/hooks/"*; do
    chmod +x "$hook"
done

# Check for pip3
if [ ! "$(command -v pip3)" ]; then
    echo "Error: pip3 is not installed"
    exit 1
fi

# Check for uv
if [ ! "$(command -v uv)" ]; then
    echo "uv not found, installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh -q
fi

# Check if .pyenv directory already exists
if [ ! -d "${GIT_ROOT}/.pyenv" ]; then
    uv venv -p 3.12 "${GIT_ROOT}/.pyenv"
fi

source "${GIT_ROOT}/.pyenv/bin/activate"
uv pip install -e "${GIT_ROOT}[dev,openai,cohere,image]"

echo "Environment configured successfully!"
echo "To activate the environment, run: source ${GIT_ROOT}/.pyenv/bin/activate"