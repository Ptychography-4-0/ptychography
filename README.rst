.. commented out with ".. "
.. |travis|_ |appveyor|_ |zenodo|_ |github|_

.. .. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1477847.svg
.. .. _zenodo: https://doi.org/10.5281/zenodo.1477847

.. .. |github| image:: https://img.shields.io/badge/GitHub-GPL--3.0-informational
.. .. _github: https://github.com/LiberTEM/LiberTEM/

Here goes the general description

Installation
------------

The short version:

.. code-block:: shell

    $ # TODO update me
    $ virtualenv -p python3.6 ~/libertem-venv/
    $ source ~/libertem-venv/bin/activate
    (libertem) $ pip install libertem[torch]

Please see `our documentation TODO <https://libertem.github.io/LiberTEM/install.html>`_ for details!

Deployment as a single-node system for a local user is thoroughly tested and can be considered stable. Deployment on a cluster is 
experimental and still requires some additional work, see `Issue #105 <https://github.com/LiberTEM/LiberTEM/issues/105>`_.

Applications
------------

- Scalable, parallel implementation of the Single Side Band method that is suitable for live data processing

Please see `the applications section TODO update
<https://libertem.github.io/LiberTEM/applications.html>`_ of our documentation
for details!


Ptychography 4.0 is evolving rapidly and prioritizes features following user
demand and contributions. In the future we'd like to implement live acquisition,
and more analysis methods for all applications of pixelated STEM and other
large-scale detector data. If you like to influence the direction this project
is taking, or if you'd like to `contribute TODO update
<https://libertem.github.io/LiberTEM/contributing.html>`_, please contact us via TODO. 

License
-------

Ptychography 4.0 is licensed under TODO discuss GPLv3.
