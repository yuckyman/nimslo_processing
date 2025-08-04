#!/bin/bash
# convenience script to activate nimslo environment

echo "🐍 activating nimslo processing environment..."
conda activate /opt/homebrew/Caskroom/miniconda/base/envs/nimslo_processing

echo "✅ environment activated!"
echo "💡 to start jupyter lab: jupyter lab"
echo "📝 to process nimslo images: open processor.ipynb"