# Problem Set 0 Environment verification

This Problem Set is to setup the working environment for all other problem sets.

1. Start by installing Conda for your operating system following the 
instructions [here](https://conda.io/docs/user-guide/install/index.html).
 

2. Now install the environment described in requirements.yaml using 
the following command:
**NOTE: ALTHOUGH THE CONDA ENVIRONMENT HAS BEEN TESTED FOR CROSS-PLATFORM 
COMPATIBILITY WITH THE AUTOGRADER ENVIRONMENT, THE CONDA ENVIRONMENT IS NOT
 THE EXACT ENVIRONMENT YOUR CODE IS RUN IN BY THE AUTOGRADER. YOU ARE RESPONSIBLE 
 TO ENSURE YOUR CODE WORKS ON THE AUTOGRADER SYSTEM–IT IS NOT ENOUGH THAT IT 
 WORKS ON YOUR SYSTEM IN THE CONDA ENVIRONMENT.**
```bash
conda env create -f requirements.yaml
```
Windows users may have to look at the 'Import Fails on Windows' question in the 
[pypi FAQ](https://pypi.org/project/opencv-contrib-python/4.0.0.21/).
3. To activate the environment run the command:
```bash
conda activate CS6476
```

4. Once inside the environment run the following command:
```bash
python ps0.py
```
You should see the image of a turtle. Press enter and all the tests should 
pass without errors.
