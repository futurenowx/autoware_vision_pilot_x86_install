Base of this application is https://github.com/autowarefoundation/autoware.privately-owned-vehicles/
(Attention we rename the folder Models/visualization to Models/visualization_org and install ours)


Tested on Ubuntu 22.04, you can test on your own risk. If you have questions open an issue

how to install all

    01. Download the neccessary weights
    
    Open your Browsrer
    Download https://hidrive.ionos.com/lnk/ZCcx1DN7Y
    Copy the file to $HOME/Downloads
    
    02. Download Autoware_privately_vehicle and install updated files and autoware_projects
    
    mkdir $HOME/autoware_privately_temp
    cd $Home/autoware_privately_temp
    git clone https://github.com/futurenowx/

    sudo chmod +x 02_autoware_privately_install.sh
    sudo chmod +x 03_autoware_weights_install.sh
    
    run ./02_autoware_privately_install.sh
    run ./03_autoware_weights_install.sh

If all runned without error, go to autoware_projects/commands and test the commands
