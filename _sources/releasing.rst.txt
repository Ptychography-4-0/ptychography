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

Install dependencies from :code:`scripts/requirements.txt`,
which are used by :code:`scripts/release`. Then call the script with
the :code:`bump` command, with the new version as parameter:

.. code-block:: shell

    $ ./scripts/release bump v0.3.0.rc0 --tag

If you are bumping to a .dev0 suffix, omit :code:`--tag` and only pass :code:`--commit`:

.. code-block:: shell

    $ ./scripts/release bump v0.4.0.dev0 --commit

.. note::
   In normal development, the version in the master branch will be x.y.z.dev0,
   if the next expected version is x.y.z. When starting the release process, it
   will be bumped up to x.y.zrc0 (note: no dot before rc!) and possibly
   additional release candidates afterwards (rc1, ..., rcN). These release candidates
   are done mostly to assure our release scripts work as expected and for doing
   additional QA. See below for our QA process.

Before (using a release candidate package)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Review open issues and pull requests
* Handle deprecation, search the code base for :code:`DeprecationWarning`
  that are supposed to be removed in that release.
* Full documentation review and update, including link check using
  ``sphinx-build -b linkcheck "docs/source" "docs/build/html"``
* Run complete test suite, including slow tests that are deactivated by default
  and tests that require sample files. Also run tests that require a GPU.
* Update the expected version in notes on changes, i.e. from :code:`0.3.0.dev0`
  to :code:`0.3.0` when releasing version 0.3.0.
* Update and review change log in :code:`docs/source/changelog.rst`, merging
  snippets in :code:`docs/source/changelog/*/` as appropriate.
* Update the JSON files in the ``packaging/`` folder with author and project information
* Create a release candidate using :code:`scripts/release`. See :code:`scripts/release --help` for details.
* `Confirm that wheel and tar.gz are built for the release candidate on
  GitHub <https://github.com/Ptychography-4-0/ptychography/releases>`_
* Confirm that a new version with the most recent release candidate is created in the
  `Zenodo.org sandbox <https://sandbox.zenodo.org/record/743636>`_ that is ready for submission.
* Install release candidate packages in a clean environment
  (for example:
  :code:`python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'ptychography40==0.1.0rc0'`)
* Make sure you have test files for all supported algorithms available
* Run all examples and applications using the test files
    * Are parameters recognized correctly, as far as implemented?
    * Any bad default values?
    * Does the file open correctly?
    * Does the result change when the input parameters are changed?
    * All display channels present and looking reasonable?
    * Reasonable performance?
* Run all examples and applications on Windows
* Confirm that pull requests and issues are handled as intended, i.e. milestoned and merged
  in appropriate branch.

After releasing on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~

* Confirm that all release packages are built and release notes are up-to-date
* Install release package
* Confirm correct version info
* confirm package upload to PyPI
* Publish new version on zenodo.org
* Update documentation with new links, if necessary
    * Add zenodo badge for the new release to Changelog page
* Send announcement message on mailing list
* Bump version in master branch to next .dev0
