[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![ci](https://github.com/cknoll/Onotology-of-machine-learning/actions/workflows/python-app.yml/badge.svg)

# Ontology of Machine Learning

This repo contains formally represented knowledge of the domain of *Machine Learning*. Is is still in **early stage of development** and should be considered as experimental.


## Background

This ontology is specified using the framework [pyirk](https://github.com/ackrep-org/pyerk-core) (not OWL). It is developed in parallel to pyirk.

## Tips:

- Use `pytest` (executed in the root directory of this repo) to run the OCSE unittests.
- Use `pyirk -ac` to generate `.ac_candidates.txt` file used for [autocompletion](https://github.com/ackrep-org/irk-fzf) in *code* editor.
- `pyirk -i -l oml.py ml`
    - load pyirk in interactive mode (useful for debugging and exploring)



# Coding style

- We use `black -l 110 ./` to ensure coding style consistency.
- For commit messages we try to follow the [conventional commits specification](https://www.conventionalcommits.org/en/).
