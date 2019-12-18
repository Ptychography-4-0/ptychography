Debugging
=========

.. testsetup:: *

    import numpy as np
    from libertem import api
    from libertem.executor.inline import InlineJobExecutor

    ctx = api.Context(executor=InlineJobExecutor())
    data = np.random.random((16, 16, 32, 32)).astype(np.float32)
    dataset = ctx.load("memory", data=data, sig_dims=2)
    roi = np.random.choice([True, False], dataset.shape.nav)

There are different parts of LiberTEM which can be debugged with different tools and methods.

.. _`debugging udfs`:

Debugging UDFs or other Python code
-----------------------------------

If you are trying to write a UDF, or debug other Python parts of LiberTEM, you can
instruct LiberTEM to use simple single-threaded execution using the
:class:`~libertem.executor.inline.InlineJobExecutor`.

.. testsetup::

    from libertem.udf.logsum import LogsumUDF

    udf = LogsumUDF()

.. testcode::

   from libertem.executor.inline import InlineJobExecutor
   from libertem import api as lt

   ctx = lt.Context(executor=InlineJobExecutor())

   ctx.run_udf(dataset=dataset, udf=udf)


You can then use all usual debugging facilities, including
`pdb <https://docs.python.org/3.7/library/pdb.html>`_ and
`the %pdb magic of ipython/Jupyter <https://ipython.org/ipython-doc/3/interactive/magics.html#magic-pdb>`_.

If the problem is only reproducible using the default executor, you will have to follow the
`debugging instructions of dask-distributed <https://docs.dask.org/en/latest/debugging.html>`_.
As the API server can't use the synchronous :class:`~libertem.executor.inline.InlineJobExecutor`,
this is also the case when debugging problems that only occur in context of the API server.
