#!/bin/bash
# ORACLE v4.0 Setup Script
# Adds new files to your local oracle-solubility repository

echo "🧬 Setting up ORACLE v4.0..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Please run this from the oracle-solubility root directory"
    exit 1
fi

# Create directories
mkdir -p scripts
mkdir -p oracle/structural
mkdir -p results/v4

echo "📁 Created directories"

# Note: You'll need to copy the files we created to these locations:
echo "
📋 Next steps:
1. Copy these files to your repository:
   - alphafold_features.py → oracle/structural/
   - oracle_v4.py → oracle/
   - enhance_data_pipeline.py → scripts/
   - train_oracle_v4_colab.py → scripts/

2. Update __init__.py files:
   - Add OracleV4 imports
   - Update version number to 4.0

3. Commit and push:
   git add .
   git commit -m 'Add ORACLE v4.0 with structural features'
   git push -u origin oracle-v4

4. Open Colab and run:
   !git clone https://github.com/61-Keys/oracle-solubility.git
   !git checkout oracle-v4
   !python scripts/enhance_data_pipeline.py
"

echo "✅ Setup script complete!"
