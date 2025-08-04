#!/bin/bash
# helper script for environment syncing

set -e  # exit on error

# colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

echo -e "${YELLOW}🐍 nimslo environment sync script${NC}"

# check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ conda not found. please install conda/miniconda first${NC}"
    exit 1
fi

# function to export environment
export_env() {
    echo -e "${YELLOW}📤 exporting current environment...${NC}"
    
    if conda env list | grep -q "nimslo_processing"; then
        conda activate nimslo_processing
        conda env export --no-builds > environment.yml
        echo -e "${GREEN}✅ environment exported to environment.yml${NC}"
        
        # show diff if git is available
        if command -v git &> /dev/null && git status &> /dev/null; then
            echo -e "${YELLOW}📊 environment changes:${NC}"
            git diff environment.yml || true
        fi
    else
        echo -e "${RED}❌ nimslo_processing environment not found${NC}"
        exit 1
    fi
}

# function to update environment
update_env() {
    echo -e "${YELLOW}📥 updating environment from file...${NC}"
    
    if [ ! -f "environment.yml" ]; then
        echo -e "${RED}❌ environment.yml not found${NC}"
        exit 1
    fi
    
    if conda env list | grep -q "nimslo_processing"; then
        echo -e "${YELLOW}🔄 updating existing environment...${NC}"
        conda env update -f environment.yml --prune
    else
        echo -e "${YELLOW}🆕 creating new environment...${NC}"
        conda env create -f environment.yml
        
        # install jupyter kernel
        conda activate nimslo_processing
        python -m ipykernel install --user --name nimslo_processing --display-name "nimslo processing"
    fi
    
    echo -e "${GREEN}✅ environment updated successfully${NC}"
}

# function to verify environment
verify_env() {
    echo -e "${YELLOW}🔍 verifying environment...${NC}"
    
    conda activate nimslo_processing
    
    echo "testing core packages..."
    python -c "
import sys
packages = ['numpy', 'cv2', 'matplotlib', 'PIL', 'sklearn', 'imageio']
failed = []

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        failed.append(pkg)

if failed:
    print(f'\\n❌ failed packages: {failed}')
    sys.exit(1)
else:
    print('\\n🎉 all packages loaded successfully!')
"
}

# main script logic
case "${1:-help}" in
    "export")
        export_env
        ;;
    "update")
        update_env
        ;;
    "verify")
        verify_env
        ;;
    "sync")
        export_env
        echo -e "${YELLOW}💾 remember to commit/push environment.yml to sync with remote${NC}"
        ;;
    "help"|*)
        echo "usage: $0 [command]"
        echo ""
        echo "commands:"
        echo "  export   - export current environment to environment.yml"
        echo "  update   - update environment from environment.yml"
        echo "  verify   - test that all packages load correctly"
        echo "  sync     - export and show git diff"
        echo "  help     - show this message"
        echo ""
        echo "examples:"
        echo "  $0 export    # after installing new packages"
        echo "  $0 update    # after pulling new environment.yml"
        echo "  $0 verify    # test environment health"
        ;;
esac