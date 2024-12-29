# Explanations

This file contains explanations for changes to the repository which might-not be self-explaining.

The idea is that for every git-commit which contains such a change, there is some change to this file as well. That change should contain the explanation.

---

## Reasons

IRK-packages are not intended to be installed like ordinary python packages. Thus I implemented an own loading mechanism. This commit switches from ordinary python import to this loading mechanism. This also makes the `__init__.py` file obsolete.
