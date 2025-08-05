#!/bin/bash

# nimslo processor runner
# activates conda environment and runs the processor

set -e  # exit on any error

echo "🎬 nimslo processor launcher"
echo "================================"

# check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. please install anaconda/miniconda first."
    exit 1
fi

# check if environment exists
if ! conda env list | grep -q "nimslo_processing"; then
    echo "📦 creating nimslo_processing conda environment..."
    conda env create -f environment.yml
fi

# activate environment and run processor
echo "🚀 activating nimslo_processing environment..."
source /Users/ian/miniconda3/etc/profile.d/conda.sh
conda activate nimslo_processing

# check if processor script exists
if [ ! -f "nimslo_processor.py" ]; then
    echo "❌ nimslo_processor.py not found in current directory"
    exit 1
fi

echo "🎯 running nimslo processor..."
python nimslo_processor.py

echo "✅ done!" 