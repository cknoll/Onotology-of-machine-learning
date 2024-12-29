# Explanations

This file contains explanations for changes to the repository which might-not be self-explaining.

The idea is that for every git-commit which contains such a change, there is some change to this file as well. That change should contain the explanation.

---

## Reasons

- the tests should be executable both by
    - `pytest`
        - I prefer and recommend this for manual testing (due to nicer output)
    - `python -m unittest`
        - this is used for the CI job (to avoid an unnecessary dependency and it seems to be faster)
        - Note: the `unittest` package is part of Python standard-library while `pytest` is not.
