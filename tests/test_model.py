from tests import _PATH_DATA

from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel
import pdb
import torch
import pytest

# Load data
train_set = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")

# Define dataloader
dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyAwesomeModel()
model = model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Raise error work best by omitting pytest.raises(ValueError, match='')
# By best I mean a wrong input will actually raise an error 
def test_error_on_wrong_shape():
   with pytest.raises(ValueError, match ='Expected each sample to have shape 1, 28, 28' ):
      #model(torch.randn(1,2,3,1))
      model(torch.randn([1,2,28,28]))

# Test the shape of output from model
def test_output_shape():
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        preds = model(x.to(device))
        assert len(x) == len(preds)

