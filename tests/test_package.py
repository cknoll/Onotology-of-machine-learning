import pyirk as p
import unittest
from pathlib import Path
from os.path import join as pjoin


PACKAGE_ROOT_PATH = Path(__file__).parent.parent.absolute().as_posix()
ml = p.irkloader.load_mod_from_path(pjoin(PACKAGE_ROOT_PATH, "oml.py"), prefix="ml")


class TestPackage01(unittest.TestCase):

    def test_subclass_relationship(self):
        """
        Checks that I1002-Supervised Learning is a subclass of I1001-Machine Learning
        """

        relations_dict = ml.I1002.get_relations()
        subclass_statements = relations_dict.get("irk:/builtins#R3", [])

        found_relationship = False
        for stmt in subclass_statements:
            if stmt.object == ml.I1001["Machine Learning"]:
                found_relationship = True
                break

        assert found_relationship, (
            "I1002 should be a subclass of I1001, but no matching Statement was found."
        )
