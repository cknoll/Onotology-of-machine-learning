import pytest
import pyirk as p
from oml import I1001, I1002

def test_subclass_relationship():
    """
    Checks that I1002-Supervised Learning is a subclass of I1001-Machine Learning
    """

    relations_dict = I1002.get_relations()
    subclass_statements = relations_dict.get("irk:/builtins#R3", [])

    found_relationship = False
    for stmt in subclass_statements:
        if stmt.object == I1001["Machine Learning"]:
            found_relationship = True
            break

    assert found_relationship, (
        "I1002 should be a subclass of I1001, but no matching Statement was found."
    )
