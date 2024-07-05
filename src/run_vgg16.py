from dotenv import load_dotenv

load_dotenv()
from utils import start_training
from nn2d import VGG16

if __name__ == "__main__":
    model_parameters = {
        "input_size": 9,
        "num_classes": 4,
    }
    parameters = {
        "epochs": 100,
        "patience_limit": 8,
        "batch_size": 8,
        "learning_rate": 0.1,
        "l2_decay": 0.0,
        "optimizer": "adam",
        "device": "cuda",
    }
    start_training(VGG16, 2, "vgg16_224", "9x224x224", parameters, model_parameters)
