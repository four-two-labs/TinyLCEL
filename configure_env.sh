#!/bin/bash
set -uo pipefail

QUIET=false
PATH=$HOME/.local/bin:$PATH
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
source $PROJECT_ROOT/scripts/shared
[[ "$*" == *"--quiet"* ]] || [[ "$*" == *"-q"* ]] && QUIET=true

# Check for system dependencies
check_dependencies \
    '{"curl": {"apt": "curl", "brew": "curl"}}'

# Check for uv
if [ ! "$(command -v uv)" ]; then
    warning "UV not found, installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    info "uv installed successfully!"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "${PROJECT_ROOT}/.pyenv" ]; then
    uv venv -q -p 3.12 "${PROJECT_ROOT}/.pyenv"
fi

source "${PROJECT_ROOT}/.pyenv/bin/activate"
uv pip install -q -e "${PROJECT_ROOT}[dev,openai,cohere,image,pydantic]"
info "${GREEN}✅ Environment configured successfully!${NC}"
info

info "To activate the environment, run either of the following:"
info "  ${GREEN}source $0${NC}"
info "  ${GREEN}source ${PROJECT_ROOT}/.venv/bin/activate${NC}"
info