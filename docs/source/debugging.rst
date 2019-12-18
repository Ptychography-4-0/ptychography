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

There are different parts of Ptychography 4.0 which can be debugged with different tools and methods.

.. _`debugging udfs`:

Debugging LiberTEM UDFs or other LiberTEM-related Python code
-------------------------------------------------------------

See `LiberTEM debugging
<https://libertem.github.io/LiberTEM/debugging.html#debugging-udfs-or-other-python-code>`_!
