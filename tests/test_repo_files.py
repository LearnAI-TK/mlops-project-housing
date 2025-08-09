import os


def test_has_readme():
    assert os.path.exists("README.md"), "Please keep a README.md in the repo root"
