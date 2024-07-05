from dotenv import load_dotenv

load_dotenv()
from data import make_dataset


def f(func):
    def wraper(a, b):
        func(f"{(a / b * 100):.2f}%", end="\r")

    return wraper


new_print = f(print)

make_dataset("ptbxl", "9x224x224", 2, 100, ["I", "II", "V2"], new_print)
