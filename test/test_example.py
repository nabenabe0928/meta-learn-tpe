import unittest

from repo_name.example import hello


def test_hello():
    assert hello() == "hello"


if __name__ == "__main__":
    unittest.main()
