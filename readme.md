## TrafficSignClassification (Björn Bulkens, Miriam Lorenz, Daniel Fauland)
### Requirements
- Necessary packages can be found in 'requirements.txt'
- If you do not have a NVIDIA GPU install TF2 via this command: 'pip install tensorflow'
- If you have a NVIDIA GPU make sure to install the following:
    - Latest NVIDIA GeForce drivers
    - [NVIDIA CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
    - NVIDIA cuDNN v7.6.5 for CUDA 10.1: Available [here](https://developer.nvidia.com/rdp/cudnn-download)
    - Note: You have to create a free NVIDIA developer account in order to download the software mentioned above
- Tested with Python 3.6.4 / Windows 10 version 1909 / RTX 2080 Ti 

### Get the training data
- To reduce the size of the GitHub repo the training data has to be downloaded separately
- Download the data from [Google Drive](https://drive.google.com/open?id=1B8ZRGXy273lBVUbva_xclOld-7sKPt3q)
- Extract the folder 'trainingData' to the main project directory



### Train the model yourself
- **Note**: Make sure you have at least 16GB of Ram installed or the program will crash during training!
- If you want to train the model yourself you should run the 'createAugmentedImages' file in the 'run' folder first 
(This will take several minutes)
- Then you can run the 'trainModel' file in the same directory (This process can take very long without a GPU!)
- After the training is complete you can see the accuracy and validation accuracy in a graph as well as the optimal 
amount of epochs for the validation accuracy


### Predict the validation data
- To predict the validation data you have to run the file 'predictValidationData' in the run folder
- This will show the a series of pictures with the prediction as title and the actual label as caption
- You can change the number of images with the variable 'num'


### Predict real world data
- To predict real data you have to run the file 'predictRealData'
- Any picture that is inside the folder 'realData' will be predicted
- You can add more pictures to the folder without having to change anything in the code
- All 'png' or 'jpeg' files are accepted no matter the size or resolution


### Useful git commands
    - 'git add -A'  # adds all files directories and subdirectories to the queue (dependent on your current directory)
    - 'git commit -am "Mesaage"'  # commits changes to the local repo
    - 'git push'  # Pushes all changes to the online repo
    - 'git pull' or 'git pull <link>'  # pulls newest version from github (Necessary before push command if changes werde made to the github repo)
    - 'git branch'  # Shows all available branches
    - 'git checkout <branch-name>'  # Switches to different branch
    
        
        
