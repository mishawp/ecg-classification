from dotenv import load_dotenv

load_dotenv()
from utils import start_training
from nn1d import BiGRU

if __name__ == "__main__":
    model_parameters = {
        "input_size": 3,
        "hidden_size": 256,
        "num_layers": 1,
        "dropout": 0.0,
        "device": "cuda",
    }
    parameters = {
        "epochs": 100,
        "patience_limit": 15,
        "batch_size": 256,
        "learning_rate": 0.01,
        "l2_decay": 0.0,
        "optimizer": "adam",
        "device": "cuda",
    }
    start_training(BiGRU, 1, "bigru", "1d-data", parameters, model_parameters)
