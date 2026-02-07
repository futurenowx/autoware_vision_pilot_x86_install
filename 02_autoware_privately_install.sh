#!/bin/bash

set -e  # stop on error

# Go to home directory
cd "$HOME"

# Check if folder exists
if [ -d "autoware.privately-owned-vehicles" ]; then
    mv autoware.privately-owned-vehicles autoware.privately-owned-vehicles_old
    echo "✅ Folder renamed to autoware.privately-owned-vehicles_old"
else
    echo "ℹ️ Folder autoware.privately-owned-vehicles not found, skipping rename"
fi

# Clone Autoware repository
git clone https://github.com/autowarefoundation/autoware.privately-owned-vehicles.git

# Rename Models folder
mv autoware.privately-owned-vehicles/Models/visualizations \
   autoware.privately-owned-vehicles/Models/visualizations_old
   
# Go to Downloads folder
cd "$HOME/Downloads" || { echo "❌ Cannot enter Downloads folder"; exit 1; }

# Clone the repo (rename old folder if exists)
if [ -d "autoware_privately" ]; then
    mv autoware_privately autoware_privately_old
    echo "Existing folder renamed to autoware_privately_old"
fi

git clone https://github.com/futurenowx/autoware_privately.git

# Enter the cloned repo
cd autoware_privately || { echo "❌ Cloned repo not found"; exit 1; }

# List files
echo "Files in the repo:"
ls -lh

# Ensure extraction folder exists
mkdir -p "$HOME/autoware_data2"

# Extract all ZIPs in the cloned repo into autoware_data2
for zipfile in *.zip; do
    echo "Extracting $zipfile ..."
    unzip -o "$zipfile" -d "$HOME/autoware_data2"
done

echo "✅ All ZIP files extracted into ~/autoware_data2"



# Create Project folder structure
cd "$HOME"

# Check if autoware_projects exists
if [ -d "autoware_projects" ]; then
    mv autoware_projects autoware_projects_old
    echo "Existing autoware_projects renamed to autoware_projects_old"
fi

# Create top-level folder structure
mkdir -p autoware_projects/{videos/{source,output},images/{source,output},weights,commands,camera_test}

echo "✅ New autoware_projects folder structure created"

# List of categories
categories=(EgoLanes Scene3D SceneSeg DomainSeg AutoSpeed AutoSteer)

# Create folders for videos
for cat in "${categories[@]}"; do
    mkdir -p "autoware_projects/videos/source/$cat"
    mkdir -p "autoware_projects/videos/output/$cat"
done

# Create folders for images
for cat in "${categories[@]}"; do
    mkdir -p "autoware_projects/images/source/$cat"
    mkdir -p "autoware_projects/images/output/$cat"
done

# Create folders for weights
for cat in "${categories[@]}"; do
    mkdir -p "autoware_projects/weights/$cat"
done

echo "✅ All category folders created successfully"



# Copy image files
cp "$HOME/autoware_data2/images"/highway_stuttgart.png autoware_projects/images/source/EgoLanes/ || true
cp "$HOME/autoware_data2/images"/highway_stuttgart.png autoware_projects/images/source/Scene3D/ || true
cp "$HOME/autoware_data2/images"/highway_stuttgart.png autoware_projects/images/source/SceneSeg/ || true
cp "$HOME/autoware_data2/images"/cones.jpg autoware_projects/images/source/DomainSeg/ || true
cp "$HOME/autoware_data2/images"/highway_stuttgart.png autoware_projects/images/source/AutoSpeed/ || true
cp "$HOME/autoware_data2/images"/highway_stuttgart.png autoware_projects/images/source/AutoSteer/ || true

# Copy video files
cp "$HOME/autoware_data2/videos"/highway_stuttgart.mp4 autoware_projects/videos/source/EgoLanes/ || true
cp "$HOME/autoware_data2/videos"/highway_stuttgart.mp4 autoware_projects/videos/source/Scene3D/ || true
cp "$HOME/autoware_data2/videos"/highway_stuttgart.mp4 autoware_projects/videos/source/SceneSeg/ || true
cp "$HOME/autoware_data2/videos"/cone_video.mp4 autoware_projects/videos/source/DomainSeg/ || true
cp "$HOME/autoware_data2/videos"/highway_stuttgart.mp4 autoware_projects/videos/source/AutoSpeed/ || true
cp "$HOME/autoware_data2/videos"/highway_stuttgart.mp4 autoware_projects/videos/source/AutoSteer/ || true

cp -r "$HOME/autoware_data2/visualizations" \
      "$HOME/autoware.privately-owned-vehicles/Models/" || true

# Copy others
cp "$HOME/autoware_data2/commands"/python_commands.py autoware_projects/commands/ || true
cp "$HOME/autoware_data2/commands"/help.txt autoware_projects/commands/ || true
cp "$HOME/autoware_data2/camera_test"/camera_test.py autoware_projects/camera_test/ || true


echo "✅ all files are copied successfully"

# Delete Downloads/autoware_privately if it exists
if [ -d "$HOME/Downloads/autoware_privately" ]; then
    rm -rf "$HOME/Downloads/autoware_privately"
    echo "✅ Deleted ~/Downloads/autoware_privately"
fi

echo "✅ Step1 install completed successfully, now download the weights and run autoware_weights_install.sh.
Please check the install.txt!"
