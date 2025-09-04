# How to Update a Kaggle Dataset

## Method 1: Through Kaggle Web UI

1. **Go to Your Dataset**
   - Navigate to https://www.kaggle.com/YOUR_USERNAME/datasets
   - Click on your "sage-7m" or "sage-7m-2" dataset

2. **Click "New Version"**
   - On your dataset page, click the "New Version" button (usually in the top right)
   - If you don't see it, click the three dots menu (...) and look for "New Version"

3. **Upload New Files**
   - Drag and drop the new files, or click to browse
   - You can add new files or replace existing ones
   - Files with same names will be replaced

4. **Add Version Notes**
   - Describe what changed (e.g., "Added debug scripts")
   - Click "Create" or "Save Version"

## Method 2: Using Kaggle API (Recommended)

### First Time Setup
```bash
# Install Kaggle API
pip install kaggle

# Set up credentials (get from Kaggle -> Account -> API -> Create New Token)
# This downloads kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Update Dataset
```bash
# Navigate to your submission package directory
cd /mnt/c/projects/ai-agents/HRM/kaggle_submission_package

# Create metadata file if it doesn't exist
kaggle datasets init -p .

# Edit dataset-metadata.json to have your dataset info
# Should look like:
# {
#   "id": "YOUR_USERNAME/sage-7m-2",
#   "licenses": [{"name": "MIT"}],
#   "title": "SAGE-7M ARC Prize 2025 Model v2"
# }

# Create new version with all files
kaggle datasets version -p . -m "Added debug scripts"
```

## Method 3: Quick File Update Script

Create this script to automate updates:

```bash
#!/bin/bash
# save as: update_kaggle_dataset.sh

# Package the files
cd /mnt/c/projects/ai-agents/HRM
cp kaggle_submission_package/kaggle_submission_debug.py kaggle_submission_package/
cp kaggle_submission_package/debug_notebook.ipynb kaggle_submission_package/

# Create new zip
zip -r arc_prize_2025_sage7m_v2.zip kaggle_submission_package/

echo "Now upload arc_prize_2025_sage7m_v2.zip as a new version on Kaggle"
```

## What Files to Include in Update

For the debug version, your dataset should contain:
```
sage-7m-2/
├── hrm_arc_best.pt              # Model checkpoint
├── kaggle_submission.py         # Original submission script  
├── kaggle_submission_debug.py   # NEW: Debug version
├── kaggle_requirements.txt      # Dependencies
├── debug_notebook.ipynb         # NEW: Debug notebook
└── submission_notebook.ipynb    # Original notebook
```

## Important Notes

1. **Version History**: Each update creates a new version, old versions are retained
2. **Competition Usage**: Update the notebook to use the latest version number
3. **File Paths**: After update, paths might change to include version number
   - Old: `/kaggle/input/sage-7m-2/file.py`
   - New: `/kaggle/input/sage-7m-2/file.py` (usually stays same)
4. **Processing Time**: New versions take a few minutes to process
5. **Size Limits**: 20GB per dataset (private), 10GB (public)

## Troubleshooting

If "New Version" button is missing:
- Check you're logged in as the dataset owner
- Try refreshing the page
- Use the API method instead
- Create a completely new dataset (sage-7m-3) if needed

## After Updating

In your Kaggle notebook:
1. Go to "Data" tab on the right
2. Remove old dataset version
3. Add new dataset version
4. Update paths if needed
5. Run the debug notebook