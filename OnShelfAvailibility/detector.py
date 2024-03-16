from dataclasses import dataclass
from collections import Counter
import pathlib
from typing import Union, Tuple
import pandas as pd
import torch
import numpy as np
import cv2
import pickle


"""
To avoid 'cannot instantiate 'PosixPath' on your system. Cache may be out of date, try `force_reload=True`' error
for some reason i get this error on this model, it didn't happenon models I've trained in the past/
"""
pathlib.PosixPath = pathlib.WindowsPath


@dataclass
class Detector:
    model_path: str
    conf_threshold: float = .3
    ultralitycs_path: str = "ultralytics/yolov5"
    model_type: str = "custom"
    force_reload: bool = True

    def __post_init__(self) -> None:
        self.model = torch.hub.load(self.ultralitycs_path, self.model_type, self.model_path, self.force_reload)
        self.model.conf = self.conf_threshold

    def detect(self, img: Union[str, np.array]) -> Tuple[np.array, pd.DataFrame]:
        results = self.model([img])

        return np.squeeze(results.render()), results.pandas().xyxy[0]


if __name__ == '__main__':
    detector = Detector(model_path=fr"model\best.pt")

    image = cv2.imread("images2/dff.png")

    converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_draw, res = detector.detect(img=converted)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow('MainWindow', image_draw)
    cv2.waitKey(0)
