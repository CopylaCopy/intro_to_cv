import numpy as np

def predict(img: np.ndarray, model_path: str) -> np.ndarray:
    model = np.load(model_path)   
    return (img - model[0])/model[1]

def train(img: np.ndarray, save_model_path: str) -> None:
    arr = np.array((np.mean(img), np.std(img)))
    np.save(save_model_path, arr)