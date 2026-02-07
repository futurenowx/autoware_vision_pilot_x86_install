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
cd "$HOME" || { echo "❌ Cannot enter $Home folder"; exit 1; }


# Clone the repo (rename old folder if exists)
if [ -d "autoware_privately_temp" ]; then
    mv autoware_privately autoware_privately_old
    echo "Existing folder renamed to autoware_privately_temp_old"
fi


# Enter the cloned repo
cd autoware_privately_temp || { echo "❌ folder autoware_privately_temp do not exist"; exit 1; }

git clone https://github.com/futurenowx/autoware_privately.git


# List files
echo "Files in the repo:"
ls -lh


# Extract all ZIPs in the cloned repo into autoware_data2
for zipfile in *.zip; do
    echo "Extracting $zipfile ..."
    unzip -o "$zipfile" -d "$HOME/autoware_privately_temp"
done

echo "✅ All ZIP files extracted into $HOME/autoware.privately-owned-vehicles/Models"


# Extract x86_visualization.zip to $Home
for zipfile in x86_visualization.zip; do
    echo "Extracting $zipfile ..."
    unzip -o "$zipfile" -d "$HOME"
done

echo "✅ autoware_projects.zip file extracted into $Home"


# Extract autoware_projects.zip to $Home
for zipfile in autoware_projects.zip; do
    echo "Extracting $zipfile ..."
    unzip -o "$zipfile" -d "$HOME"
done

echo "✅ autoware_projects.zip file extracted into $Home"


# Delete Downloads/autoware_privately if it exists
if [ -d "$HOME/autoware_privately_temp" ]; then
    rm -rf "$HOME/autoware_privately_temp"
    echo "✅ Deleted ~/Downloads/autoware_privately_temp"
fi

echo "✅ Step1 install completed successfully, now download the weights and run 03_autoware_weights_install.sh.
Please check the install.txt!"
