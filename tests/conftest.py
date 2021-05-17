import os
import re
import shutil
import pytest


ROOT = os.path.dirname(os.path.realpath(__file__))


def pytest_addoption(parser):
    parser.addoption("--out_folder", action="store", default=None)


def get_test_path(path, request):
    """ Constructs path for the test execution from the test file's name, which it gets from
        pytest.FixtureRequest (https://docs.pytest.org/en/latest/reference.html#request).
    """
    test_name = re.findall("test_(.*).py", request.fspath.basename)[0]

    return os.path.join(ROOT, 'data', test_name, path)


@pytest.fixture(scope='module')
def input_folder(request):
    """ Creates the input folder path `dione-sr/tests/data/test_name/input`.
    """
    return get_test_path('input', request)


@pytest.fixture(scope='module')
def compare_folder(request):
    """ Creates the compare folder path `dione-sr/tests/data/test_name/compare`.
    """
    return get_test_path('compare', request)


@pytest.fixture(scope='module')
def output_folder(request):
    """ Creates the output folder path `dione-sr/tests/data/test_name/output`.

        It also cleans the output folder before the test runs.
    """

    out_path = request.config.getoption("out_folder")

    if out_path is None:
        out_path = get_test_path('output', request)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        os.makedirs(out_path)

    yield out_path

    # shutil.rmtree(OUTPUT_FOLDER)

