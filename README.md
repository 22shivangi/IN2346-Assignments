# Introduction to Deep Learning
# Technical University Munich - SS 2018

1. Python Setup
2. PyTorch Installation
3. Exercise Download
4. Dataset Download
5. Exercise Submission
6. Google Cloud
7. Acknowledgments


## 1. Python Setup

Prerequisites:
- Unix system (Linux or MacOS)
- Python version 3
- Terminal (e.g. iTerm2 for MacOS)
- Integrated development environment (IDE) (e.g. PyCharm or Sublime Text)

For the following description, we assume that you are using Linux or MacOS and that you are familiar with working from a terminal. The exercises are implemented in Python 3.

If you are using Windows, the procedure might slightly vary and you will have to Google for the details. A fellow student of yours compiled this (https://gitlab.lrz.de/yuesong.shen/DL4CV-win) very detailed Windows tutorial for a previous course. Please keep in mind, that we will not offer any kind of support for its content.

To avoid issues with different versions of Python and Python packages we recommend to always set up a project specific virtual environment. The most common tools for a clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*.

In this README we provide you with a short tutorial on how to use and setup a *virtuelenv* environment. To this end, install or upgrade *virtualenv*. There are several ways depending on your OS. At the end of the day, we want

`which virtualenv`

to point to the installed location.

On Ubuntu, you can use:

`apt-get install python-virtualenv`

Also, installing with pip should work (the *virtualenv* executable should be added to your search path automatically):

`pip3 install virtualenv`

Once *virtualenv* is successfully installed, go to the root directory of the i2dl repository (where this README.md is located) and execute:

`virtualenv -p python3 --no-site-packages .venv`

Basically, this installs a sandboxed Python in the directory `.venv`. The
additional argument ensures that sandboxed packages are used even if they had
already been installed globally before.

Whenever you want to use this *virtualenv* in a shell you have to first
activate it by calling:

`source .venv/bin/activate`

To test whether your *virtualenv* activation has worked, call:

`which python`

This should now point to `.venv/bin/python`.

From now on we assume that that you have activated your virtual environment.

Installing required packages:
We have made it easy for you to get started, just call from the i2dl root directory:

`pip3 install -r requirements.txt`

The exercises are guided via Jupyter Notebooks (files ending with `*.ipynb`). In order to open a notebook dedicate a separate shell to run a Jupyter Notebook server in the i2dl root directory by executing:

`jupyter notebook`

A browser window which depicts the file structure of directory should open (we tested this with Chrome). From here you can select an exercise directory and one of its exercise notebooks!


## 2. PyTorch installation

In exercise 3 we will introduce the *PyTorch* deep learning framework which provides a research oriented interface with a dynamic computation graph and many predefined, learning-specific helper functions.

Unfortunately, the installation depends on the individual system configuration (OS, Python version and CUDA version) and therefore is not possible with the usual `requirements.txt` file.

Follow the *Get Started* section on the official PyTorch [website](http://pytorch.org/) to choose and install your version.


## 3. Exercise Download

Our exercise is structured with git submodules. At each time we start with a new exercise you have to populate the respective exercise directory. __Access to the corresponding repositories will be granted once the new exercise starts.__
You obtain the exercises by first updating the i2dl root repository:

`git fetch origin`

`git reset --hard origin/master`

`git pull origin master`

and then pulling the respective exercise submodule:

`git submodule update --init -- exercise_{0, 1, 2, 3, 4}`


## 4. Dataset Download

To download the datasets required for an exercise, execute the respective download script located in the exercise directory:

`./get_datasets.sh`

You will need ~400MB of disk space.


## 5. Exercise Submission

After completing an exercise you will be submitting trained models to be
automatically evaluated on a test set on our server. To this end, login or register for an account at:

https://dvl.in.tum.de/teaching/submission/

Note that only students, who have registered for this class in TUM Online can
register for an account. This account provides you with temporary credentials to login onto the machines at our chair.

After you have worked through an exercise, your saved models will be in the corresponding `models` subfolder of this exercise. In order to submit the models you execute our submit script:

`./submit_exercise.sh X s9999`

where `X={0,1,2,3,4}` for the respective exercise and `s9999` has to be substituted by your username in our system.

This script uses *rsync* to transfer your code and the models onto our test server and into your user's home directory `~/submit/EX{0, 1, 2, 3, 4}`. Make sure *rsync* is installed on your local machine and don't change the filenames of your models!

Once the models are uploaded to `~/submit/EX{0, 1, 2, 3, 4}`, you can login to the above website, where they can be selected for submission. Note that you have to explicitly submit the files through our web interface, just uploading them to the respective directory is not enough.

You will receive an email notification with the results upon completion of the
evaluation. To make it more fun, you will be able to see a leader board of everyone's (anonymous) scores on the login part of the submission website.

Note that you can re-evaluate your models until the deadline of the current exercise. Whereas the email contains the result of the current evaluation, the entry in the leader board always represents the best score for the respective exercise.


## 6. Google Cloud

Starting from the third exercise, we will use PyTorch which supports GPU computations. This means, that students with a GPU will be able to run a more in-depth hyperparameter search. However, our exercise goals can be reached quite easily with CPU only, so there is no disadvantage for students that want to get the bonus and don't have access to a GPU.
In a previous class we used [google cloud](https://cloud.google.com/) to offer access to students as it offers a free 300$ trial. If you are interested in working on a remote machine, please check out the "google_cloud.pdf" for a short tutorial. Since mis-usage of this tutorial can lead to costs on your side, we don't require it and don't take any responsibility for potential problems on your end. You do this at your own risk.


## 6. Acknowledgments

We want to thank the **Stanford Vision Lab** and **PyTorch** for allowing us to build these exercises on material they had previously developed.
