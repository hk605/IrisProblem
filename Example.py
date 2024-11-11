from BuildDataset import load_data
from NeuralNetwork import NeuralNetwork

# Load data and initialize model
train_loader, test_loader = load_data()
model = NeuralNetwork()