import oz_viewer


def test_imports_with_version():
    assert isinstance(oz_viewer.__version__, str)
