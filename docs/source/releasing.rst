Releasing
=========

This document describes release procedures and infrastructure that is relevant
for advanced contributors. See :ref:`contributing` for information on regular
contributions.

Release checklist
-----------------

Not all aspects of ptychography are covered with automated unit tests. For that
reason we should perform some manual tests before and after a release.

Tagging a version
~~~~~~~~~~~~~~~~~

Install :code:`pygithub`, which is used by :code:`scripts/release`. Then call the script with
the :code:`bump` command, with the new version as parameter:

.. code-block:: shell

    $ ./scripts/release bump v0.3.0.rc0 --tag

If you are bumping to a .dev0 suffix, omit :code:`--tag` and only pass :code:`--commit`:

.. code-block:: shell

    $ ./scripts/release bump v0.4.0.dev0 --commit

Before (using a release candidate package)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Review open issues and pull requests
* Full documentation review and update, including link check using
  ``sphinx-build -b linkcheck "docs/source" "docs/build/html"``
* Update the expected version in notes on changes, i.e. from :code:`0.3.0.dev0`
  to :code:`0.3` when releasing version 0.3.
* Update and review change log in :code:`docs/source/changelog.rst`, merging
  snippets in :code:`docs/source/changelog/*/` as appropriate.
* Update the JSON files in the ``packaging/`` folder with author and project information
* Create a release candidate using :code:`scripts/release`. See :code:`scripts/release --help` for details.
* `Confirm that wheel and tar.gz are built for the release candidate on
  GitHub <https://github.com/Ptychography-4-0/ptychography/releases>`_
* Confirm that a new version with the most recent release candidate is created in the
  `Zenodo.org sandbox <https://example.com/fixme>`_ that is ready for submission.
* Install release candidate packages in a clean environment
  (for example: 
  :code:`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'ptychography==0.2.0rc11'`)
* Copy test files of all supported types to a fresh location or purge the parameter cache
    * Include floats, ints, big endian, little endian, complex raw data
* Open each test file
    * Are parameters recognized correctly, as far as implemented?
    * Any bad default values?
    * Does the file open correctly?
* Perform all analyses on each test file.
    * Does the result change when the input parameters are changed?
    * Reasonable performance?
* Try opening all file types with wrong parameters
    * Proper understandable error messages?
* Run all examples
* Confirm that pull requests and issues are handled as intended, i.e. milestoned and merged
  in appropriate branch.

After releasing on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~

* Confirm that all release packages are built and release notes are up-to-date
* Install release package
* Confirm correct version info
* confirm package upload to PyPi
* Publish new version on zenodo.org
* Update documentation with new links, if necessary
    * Add zenodo badge for the new release to Changelog page
* Send announcement message on mailing list
