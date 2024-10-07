# FAIML_templates

Exercise code templates for students in the course Foundations of Artificial Intelligence

## How to setup a virtual Python environment with the required packages (for running the notebooks locally)

### FAQs

-   **What is a Python virtual environment?**
    -   A Python virtual environment creates a local sandbox of your Python version such that you don't need to install globally packages used here. This is done to ensure that the environment matches the enviroment in which the notebooks have been created/tested.
-   **Do I need to do this or can I use my global Python?**
    -   No this is not required to run but you should try to install the required packages. This might cause version issues if you've installed packages before or have conflicting package versions.
-   **Can I use cloud computing services such as Colab to run the task notebooks?**
    -   Yes but you have to ensure that the needed packages are either installed or can be installed to the service you intend to use.

### REQUIREMENTS

-   Have a Python installed (recommended python 3.9 or newer)
-   Have pip installed

### STEPS

1. Create a folder (or clone this repository) for the FAIML tasks (can have subfolders)
2. Download and copy the requirements.txt file from this repository to the folder (not needed if you've cloned the repository as a whole)
3. Navigate to the folder in your terminal
4. Create the python virtual environment with the command `python -m venv .$NAME OF YOUR ENVIRONMENT$` (you can change the NAME OF THE ENVIRONMENT to whatever you like for example venv or Faiml_venv)
5. Activate the virtual environment with the command `.\$NAME OF THE ENVIRONMENT$\Scripts\activate` on Windows commandline, `source .$NAME OF THE ENVIRONMENT$/bin/activate` on Linux or Mac
6. (Optional) Run pip -V, it should print out the path of the virtual environment if you've activated it
7. First install these packages manually with commands
    - `pip install cython`
    - `pip install vose --no-build-isolation`
8. Then install the rest of required packages with the command `pip install -r requirements.txt` (this can take a few minutes)
9. After installation, you can deactivate the environment in the command shell or leave it running if you wish. To deactivate just enter the command `deactivate`.

**Usage**

After the setup, the virtual environment should pop up as a choosable kernel when a Jupyter notebook is open. Choose it and run the codes normally. The environment is only locally available in the folder in which you've created it thus it is best to create folders for your exercise/homework tasks in the same folder with the virtual environment.
