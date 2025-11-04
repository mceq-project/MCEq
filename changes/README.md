# How to use `towncrier`

A tutorial can be found [here](https://towncrier.readthedocs.io/en/stable/tutorial.html).

1. Create a new file for your changes `<PULL REQUEST ID>.<TYPE>.md` in the corresponding folder. The following `TYPEs` are available:
    - feat: `New Features`
    - bugfix: `Bug Fixes`
    - api: `API Changes`
    - chore: `Maintenance`
    - docs: `Documentation Updates`

2. Write a suitable message for the change:
    ```
    Fixed ``crazy_function`` to be consistent with ``not_so_crazy_function``
    ```

3. (For maintainers) How to generate a change log:
    - Execute the following command in the base directory of the project
    ```
    towncrier build
    ```
    - Use the `--draft` flag to check the output before
    - You might need to provide a `--version=...`. By default the versions of the currently installed `MCEq` is used. You can also updated via bumping the version number (which should be done in any case) and running `pip install -e.` again.
