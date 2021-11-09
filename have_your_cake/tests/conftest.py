import pytest
from have_your_cake.mrc import read_mrc

MRC_FILE = '../../emd_10160.map'


@pytest.fixture
def mrc_file():
    return MRC_FILE


@pytest.fixture
def volume():
    return read_mrc(MRC_FILE)