.. _`installation`:

Installation
============

.. note::
    Ptychography 4.0 is currently not working with Python 3.9
    because support in LiberTEM is still pending:
    https://github.com/LiberTEM/LiberTEM/issues/914. Python 3.6, Python
    3.7 and Python 3.8 are supported.

The short version
-----------------

.. code-block:: shell

    $ virtualenv -p python3 ~/ptychography-venv/
    $ source ~/ptychography-venv/bin/activate
    (ptychography) $ python -m pip install ptychography

    # optional for GPU support
    (ptychography) $ python -m pip install cupy

For details, please read on!


Linux and Mac OS X
------------------

Creating an isolated Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To provide an isolated environment for ptychography and its dependencies, you can
use virtualenvs or conda environments.

Using virtualenv
################

You can use `virtualenv <https://virtualenv.pypa.io/>`_ or `venv
<https://docs.python.org/3/tutorial/venv.html>`_ if you have a system-wide
Python 3.6 or 3.7 installation. For Mac OS X, using conda is recommended.

To create a new virtualenv for ptychography, you can use the following command:

.. code-block:: shell

    $ virtualenv -p python3.7 ~/ptychography-venv/

Replace :code:`~/ptychography-venv/` with any path where you would like to create
the venv. You can then activate the virtualenv with

.. code-block:: shell
    
    $ source ~/ptychography-venv/bin/activate

Afterwards, your shell prompt should be prefixed with :code:`(ptychography)` to
indicate that the environment is active:

.. code-block:: shell

    (ptychography) $ 

For more information about virtualenv, for example if you are using a shell
without `source`, please `refer to the virtualenv documentation
<https://virtualenv.pypa.io/en/latest/user_guide.html>`_. If you are often
working with virtualenvs, using a convenience wrapper like `virtualenvwrapper
<https://virtualenvwrapper.readthedocs.io/en/latest/>`_ is recommended.

Using conda
###########

If you are already using conda, or if you don't have a system-wide Python 3.6, 3.7 or
3.8 installation, you can create a conda environment for ptychography.

This section assumes that you have `installed conda
<https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_
and that your installation is working.

You can create a new conda environment to install ptychography with the following
command:

.. code-block:: shell

    $ conda create -n ptychography python=3.7

To install or later run ptychography, activate the environment with the following
command:

.. code-block:: shell  

    $ conda activate ptychography

Afterwards, your shell prompt should be prefixed with :code:`(ptychography)` to
indicate that the environment is active:

.. code-block:: shell

    (ptychography) $ 

Now the environment is ready to install ptychography.
    
For more information about conda, see their `documentation about creating and
managing environments
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

.. Installing from PyPi
.. ~~~~~~~~~~~~~~~~~~~~

.. To install the latest release version, you can use pip. Activate the Python
   environment (conda or virtualenv) and install using:

.. .. code-block:: shell

..    (ptychography) $ python -m pip install ptychography

.. This should install ptychography and its dependencies in the environment. Please
   continue by reading about the :ref:`algorithms`.

.. _`installing from a git clone`:

Installing from a git clone
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to follow the latest development or contribute to ptychography, you
should install ptychography from a git clone:

.. code-block:: shell

    $ git clone https://github.com/Ptychography-4-0/ptychography

Activate the Python environment (conda or virtualenv) and change to the newly
created directory with the clone of the ptychography repository. Now you can start
the ptychography installation. Please note the dot at the end, which indicates the
current directory!

.. code-block:: shell
    
    (ptychography) $ python -m pip install -e .

This should download the dependencies and install ptychography in the environment.
Please continue by reading about the :ref:`algorithms`.

Updating
~~~~~~~~

If you have installed from a git clone, you can easily update it to the current
status. Open a command line in the base directory of the ptychography clone and
update the source code with this command:

.. code-block:: shell

    $ git pull
    
The installation with ``pip install -e`` has installed ptychography in `"editable"
mode <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_.
That means the changes pulled from git are active immediately. Only if the
requirements for installed third-party packages have changed, you can re-run
``pip install -e .`` in order to install any missing packages.

CuPy
----

GPU support is based on `CuPy <https://cupy.dev/>`_.

.. code-block:: shell

    (libertem) $ python -m pip install cupy

Windows
-------

The recommended method to install ptychography on Windows is based on `Miniconda 64
bit with Python version 3.6 or 3.7 <https://docs.conda.io/en/latest/miniconda.html>`_.
This installs a Python distribution.

For `installing from a git clone`_ you require a suitable git client, for
example `GitHub Desktop <https://desktop.github.com/>`_, `TortoiseGit
<https://tortoisegit.org/>`_, or `git for windows
<https://gitforwindows.org/>`_. Clone the repository
https://github.com/Ptychography-4-0/ptychography in a folder of your choice.

From here on the installation and running of ptychography on Windows with the
Anaconda Prompt is very similar to `Using conda`_ on Linux or Mac OS X.

Differences:

* The command to activate a conda environment on Windows is

.. code-block:: shell

    > conda activate ptychography
    
* You might have to install pip into your local ptychography conda environment to
  make sure that ``pip install`` installs packages into your local environment and
  not into the global Anaconda base environment. This helps to avoid permission
  issues and interference between environments.

.. code-block:: shell

    (ptychography) > conda install pip

Jupyter
-------

To use the Python API from within a Jupyter notebook, you can install Jupyter
into your ptychography virtual environment.

.. code-block:: shell

    (ptychography) $ python -m pip install jupyter

You can then run a local notebook from within the ptychography environment, which
should open a browser window with Jupyter that uses your ptychography environment.

.. code-block:: shell

    (ptychography) $ jupyter notebook

JupyterHub
----------

If you'd like to use the Python API from a ptychography virtual environment on a
system that manages logins with JupyterHub, you can easily `install a custom
kernel definition
<https://ipython.readthedocs.io/en/stable/install/kernel_install.html>`_ for
your ptychography environment.

First, you can launch a terminal on JupyterHub from the "New" drop-down menu in
the file browser. Alternatively you can execute shell commands by prefixing them
with "!" in a Python notebook.

In the terminal you can create and activate virtual environments and perform the
ptychography installation as described above. Within the activated ptychography
environment you additionally install ipykernel:

.. code-block:: shell

    (ptychography) $ python -m pip install ipykernel

Now you can create a custom ipython kernel definition for your environment:

.. code-block:: shell

    (ptychography) $ python -m ipykernel install --user --name ptychography --display-name "Python (ptychography)"

After reloading the file browser window, a new Notebook option "Python
(ptychography)" should be available in the "New" drop-down menu. You can test it by
creating a new notebook and running

.. code-block:: python

    In [1]: import ptychography

Troubleshooting
---------------

If you are having trouble with the installation, please let us know by `filing
an issue <https://github.com/Ptychography-4-0/ptychography/issues>`_.
