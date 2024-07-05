from dotenv import load_dotenv

load_dotenv()
from utils import start_training
from nn2d import AlexNet

if __name__ == "__main__":
    model_parameters = {
        "input_size": 9,
        "num_classes": 4,
        "dropout": 0.5,
    }
    parameters = {
        "epochs": 100,
        "patience_limit": 8,
        "batch_size": 4,
        "learning_rate": 0.01,
        "l2_decay": 0.0,
        "optimizer": "sgd",
        "device": "cuda",
    }
    start_training(AlexNet, 2, "alexnet224", "9x224x224", parameters, model_parameters)
