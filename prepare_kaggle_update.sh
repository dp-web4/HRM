#!/bin/bash
# Script to prepare Kaggle dataset update with debug files

echo "Preparing Kaggle dataset update..."

# Create a clean package directory
rm -rf kaggle_submission_package_v2
cp -r kaggle_submission_package kaggle_submission_package_v2

# Ensure debug files are included
echo "Files in package:"
ls -la kaggle_submission_package_v2/

# Create new zip file
zip -r arc_prize_2025_sage7m_debug.zip kaggle_submission_package_v2/

echo ""
echo "✓ Package created: arc_prize_2025_sage7m_debug.zip"
echo ""
echo "Next steps:"
echo "1. Go to https://www.kaggle.com/YOUR_USERNAME/datasets"
echo "2. Click on your sage-7m-2 dataset"  
echo "3. Click 'New Version' button"
echo "4. Upload arc_prize_2025_sage7m_debug.zip"
echo "5. Add version notes: 'Added debug scripts to troubleshoot output issues'"
echo "6. Click 'Create Version'"
