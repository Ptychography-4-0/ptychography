.. _`algorithms`:

Algorithms
==========

Single side band
----------------

The algorithm after :cite:`Pennycook2015` is reformulated to allow incremental
processing using a `LiberTEM user-defined function
<https://libertem.github.io/LiberTEM/udf.html>`_ :cite:`Strauch2021b`. The
notebooks can be downloaded at
https://github.com/Ptychography-4-0/ptychography/tree/master/examples.

.. figure:: ./images/live-ptychography.gif
   :alt: Animation of progressive SSB processing

   Incremental ptychography animation :cite:`Strauch2021b`.


.. toctree::
   :maxdepth: 2

   algorithms/ssb

Live processing with Quantum Detectors Merlin camera using `LiberTEM-live <https://libertem.github.io/LiberTEM-live/>`_:

.. toctree::
   :maxdepth: 2

   algorithms/live-ssb

Simplified implementation as a reference:

.. toctree::
   :maxdepth: 2

   algorithms/simple-ssb
