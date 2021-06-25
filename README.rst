|github|_

.. |github| image:: https://img.shields.io/badge/GitHub-GPL--3.0-informational
.. _github: https://github.com/Ptychography-4-0/ptychography/

.. note::

    The Ptychography 4.0 repository and documentation are currently under construction.

This repository collects implementations for ptychography that result from the work of the
`Ptychography 4.0 project <https://ptychography.helmholtz-muenchen.de/>`_.

Installation
------------

The short version:

.. To be updated after first release to install from PyPi

.. code-block:: shell

    $ virtualenv -p python3.8 ~/ptychography-venv/
    $ source ~/ptychography-venv/bin/activate
    (ptychography-venv) $ git clone https://github.com/Ptychography-4-0/ptychography
    (ptychography-venv) $ cd ptychography
    (ptychography-venv) $ python -m pip install -e .

Please see `our documentation <https://ptychography-4-0.github.io/ptychography/>`_ for details!

Applications
------------

- Scalable, parallel implementation of the Single Side Band method that is suitable for live data processing.

Please see `the algorithms section
<https://ptychography-4-0.github.io/ptychography/algorithms.html>`_ of our documentation
for details!

Ptychography 4.0 is evolving rapidly and prioritizes features following user
demand and contributions. In the future we'd like to implement live acquisition,
and more analysis methods for all applications of pixelated STEM and other
large-scale detector data. If you like to influence the direction this project
is taking, or if you'd like to `contribute
<https://ptychography-4-0.github.io/ptychography/contributing.html>`_, please
contact us in the `GitHub Issue tracker <https://github.com/Ptychography-4-0/ptychography/issues>`_.

License
-------

Ptychography 4.0 is licensed under GPLv3.
