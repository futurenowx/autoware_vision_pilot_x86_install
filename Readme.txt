ase of this application is https://github.com/autowarefoundation/autoware.privately-owned-vehicles/
(Attention we rename the folder Models/visualization to Models/visualization_org and install ours)

Tested on Ubuntu 22.04, on a X86 machine.
You can test on your own risk!! 
If you have questions open an issue


How to install all

Step 1. Download the neccessary weights

	Open your Browsrer
	Download https://hidrive.ionos.com/lnk/ZCcx1DN7Y
	Copy the file to $HOME/Downloads

Step 2. Download Autoware_privately_vehicle 

	cd $HOME
	git clone https://github.com/futurenowx/autoware_privately_x86_install.git

Step 3. Make install files executable

	cd $HOME/autoware_privately_x86_install
	sudo chmod +x 02_autoware_privately_install.sh
	sudo chmod +x 03_autoware_weights_install.sh

Step 4. install application

	cd $HOME/autoware_privately_x86_install

	pip install -r 01_requirements.txt   (Check the 01_how_to_install_requirements.txt first)
	sudo ./02_autoware_privately_install.sh
	sudo ./03_autoware_weights_install.sh

If all runned without error, go to autoware_projects/commands and test the commands
