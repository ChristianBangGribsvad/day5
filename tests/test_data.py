from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
import torch
import os.path
import pytest

# Consider loading data by (might be smarter when project is presented with new data)
#train_set = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")
#train_set = CorruptMnist(train=False, in_folder="data/raw", out_folder="data/processed")

train_set = torch.load(_PATH_DATA+'/processed/train_processed.pt')
N_train = 40000
shape = [1,28,28]
test_set = torch.load(_PATH_DATA+'/processed/test_processed.pt')
N_test = 5000

# Test length of dataset
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_len_train_set():
    assert len(train_set[1]) == N_train, "Training set did not have the correct number of samples"
def test_len_test_set():
    assert len(test_set[1]) == N_test, "Test set did not have the correct number of samples"

# Test shape of dataset
def test_train_shape():
    assert list(train_set[0].shape[1:]) == shape
def test_test_shape():
    assert list(test_set[0].shape[1:]) == shape

# Test all labels are represented
def test_train_labels():
    assert len(train_set[1].unique()) == 10
def test_test_labels():
    assert len(test_set[1].unique()) == 10

# Try parameterization
#@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
#def test_eval(test_input, expected):
#    assert eval(test_input) == expected
