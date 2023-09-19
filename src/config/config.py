# manifest file path that indexes training data
train_manifest_path="output/train_mnist.csv"
# manifest file path that indexes test data
test_manifest_path="output/test_mnist.csv"
# Batch size for training (default: 64)
batch_size=64
# Batch size for testing (default: 1000)
test_batch_size=1000
# Number of epochs to train (default: 14)
epochs=4
# Learning rate (default: 1.0)
lr=1
# Learning rate step gamma (default: 0.7)
gamma=0.7
# Enables CUDA training
use_cuda=True
# Enables macOS GPU training
use_mps=True
# Quickly check a single pass
dry_run=False
# Random seed (default: 1)
seed=1
# How many  batches to wait before logging training status
log_interval=10
# For saving the current Model
save_model=True
# Number of classes used for classification
num_classes=10
