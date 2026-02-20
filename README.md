General outline and flow for using Napari and the user interface (UI) developed by Anthony Moreno-Sanchez.

The current code and UI, has been setup to handle up to four channels, has both capabilities for handling TIFF files and LIF (Leica) files.
The purpose of this code is a simple UI to be able to load in imaging data for quick analysis. Simple functionality has been added to visualize channel data similar to Imaris/Fiji, but with a bit more control to the user. 

This current version of the pipeline includes slider controls for z stacks, opacity, brightness, contrast sliders for individual channels. 
The interface includes analysis pipelines for mask generation of individual channels, colocalcaization between channels , a cell counter, and intensity calculations from individual cells bodies.

The current working methodology is explained below:

#Installing git to pull repository
To install git for windows install using the following command
winget install --id Git.Git -e --source winget
Or
Install manually using git for windows
https://git-scm.com/download/win

To install git for mac
https://git-scm.com/install/mac
Or use the following commands
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git

### Setup of python ###

The current version of python that is used for this is 3.11.9.
It has only been tested with this version, and therefore has dependcy requirements using this version. 
Using Python 3.12 or older packages may cause compatibility errors.

Python can be downloaded from the following website
https://www.python.org/downloads/release/python-3119/


### Pulling the github repository ###
To pull the current version of the repository generate a folder in your directory
 - Navigate to that directory in a command prompt, or use a simple IDE like Visual Studio Code (https://code.visualstudio.com/download)
 - run the following command in the terminal
 git clone https://github.com/AusbornLab/Napari_UI.git
 - This generates a copy of the repository in your directory folder
 - Then change to the directory of the of the repository folder by using the following command 
 - cd Napari_UI, this pulls you to your new folder

########################

### Setting up virtual env. ###
Once python 3.11.9 is installed the following packages and versions must be install in a virtual env.
- open up a command prompt, or terminal via VS Code 
- navigate to the napari folder pulled from the github repository

#If you are using a windows machine use the following commands
- Generate virtual env, with the following: 
    py -3.11 -m venv napari_env


#If using Mac use the following commands:
- Generate virtual env, with the following: 
    python3.11 -m venv napari_env

########################

### Activation of the virtual environment ###
- In your napari_UI folder a new folder named napari_env should be there now.

#If you are using a windows machine use the following commands in the command terminal
- if using a cmd line .\napari_env\Scripts\activate.bat
- if using a powershell .\napari_env\Scripts\Activate.ps1

#If you are using a Mac, use the following commands
- use the following in a terminal (bash/zsh):
    source napari_env/bin/activate

This activates your napari_env, where the installation of packages will be done

#If you wish to deactivate the envirnment
 - In the command line use the command deactivate

########################

### Installation of packages ###

- use the following command to install the packages to your virtual env from the requirements.txt file 
#The following command works with both mac and windows machines
- pip install -r requirements.txt (for windows)
- pip install -r requirements_mac.txt (for mac systems)

########################

### Running of napari UI ###
- In the now activated napari_env you can run the Napari user interface for analysis.

#If you are using a windows machine then use the following command to start the user interface
- python Napari_UI.py

#If you are using a mac machine then use the following command to star the user interface
- python3.11 Napari_UI.py

