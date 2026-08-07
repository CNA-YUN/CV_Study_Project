from torchvision import datasets
from _init_ import BASE_ROOT,DATA_DIR
full_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR, split="trainval", target_types="segmentation", download=True
)
test_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR, split="test", target_types="segmentation", download=True)
