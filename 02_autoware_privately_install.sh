#!/bin/bash
set -e  # stop on error

# Get the actual user's home directory (works for any user, even with sudo)
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(eval echo ~$SUDO_USER)
    ACTUAL_USER="$SUDO_USER"
else
    USER_HOME="$HOME"
    ACTUAL_USER="$USER"
fi

echo "Running installation for user: $ACTUAL_USER"
echo "Home directory: $USER_HOME"

# Go to user's home directory
cd "$USER_HOME"

# Check if folder exists
if [ -d "autoware.privately-owned-vehicles" ]; then
    mv autoware.privately-owned-vehicles autoware.privately-owned-vehicles_old
    echo "✅ Folder renamed to autoware.privately-owned-vehicles_old"
else
    echo "ℹ️ Folder autoware.privately-owned-vehicles not found, skipping rename"
fi

cd "$USER_HOME"

# Clone Autoware repository
git clone https://github.com/autowarefoundation/autoware.privately-owned-vehicles.git

# Rename Models folder
mv autoware.privately-owned-vehicles/Models/visualizations \
   autoware.privately-owned-vehicles/Models/visualizations_old

# Go to autoware_privately_x86_install folder
cd "$USER_HOME/autoware_privately_x86_install" || { echo "❌ Cannot enter autoware_privately_x86_install folder"; exit 1; }

# copy folder autoware_projetcs to user home
SOURCE="autoware_projetcs"
DEST="$USER_HOME"
if [ -d "$SOURCE" ]; then
    cp -r "$SOURCE" "$DEST"
    echo "✅ autoware_projetcs folder copied successfully"
else
    echo "❌ Error: autoware_projetcs folder does not exist"
    exit 1
fi

# copy folder visualization to autoware.privately-owned-vehicles/Models
SOURCE="visualization"
DEST="$USER_HOME/autoware.privately-owned-vehicles/Models"
if [ -d "$SOURCE" ]; then
    cp -r "$SOURCE" "$DEST"
    echo "✅ visualization folder copied successfully"
else
    echo "❌ Error: visualization folder does not exist"
    exit 1
fi

echo "✅ Step1 install completed successfully, now download the weights and run 03_autoware_weights_install.sh."
echo "Please check the install.txt!"
