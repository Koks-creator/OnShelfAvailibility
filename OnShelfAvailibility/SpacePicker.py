from dataclasses import dataclass, field
import os
from typing import Tuple, Union
import cv2
import pickle


from OnShelfAvailibility.config import Config


@dataclass
class Picker:
    max_corners: int = 2
    output_file_path: Union[str, bool] = False
    __temp_points: list = field(default_factory=lambda: [])
    __regions: list = field(default_factory=lambda: [])
    regions_path: str = ""

    def __post_init__(self):
        if self.regions_path:
            with open(self.regions_path, "rb") as f:
                self.__regions = pickle.load(f)

    @staticmethod
    def check_inside2p(point: Tuple[int, int], top_left_p: Tuple[int, int], bot_right_p: Tuple[int, int]) -> bool:
        if top_left_p[0] < point[0] < bot_right_p[0] and top_left_p[1] < point[1] < bot_right_p[1]:
            return True
        return False

    def mouse_click(self, event, x, y, flags, params) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__temp_points.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            for index, region in enumerate(self.__regions):
                if self.check_inside2p(point=(x, y), top_left_p=region[0], bot_right_p=region[1]):
                    self.__regions.pop(index)

    @staticmethod
    def nothing(x) -> None:
        pass

    def run(self, image_path: str) -> None:
        print(self.__temp_points)
        if not self.output_file_path:
            self.output_file_path = f"{Config.REGIONS_FOLDER}/{os.path.split(image_path)[-1].split('.')[0]}_regions.pkl"
        while True:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (1280, 720))

            cv2.putText(img, f"Press 's' to save", (15, 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 225), 2)

            for region in self.__regions:
                cv2.rectangle(img, (region[0]), (region[-1]), (255, 0, 255), 2)

            if len(self.__temp_points) == self.max_corners:
                self.__regions.append(self.__temp_points)
                self.__temp_points = []
            for point in self.__temp_points:
                cv2.circle(img, point, 8, (255, 0, 200), -1)

            key = cv2.waitKey(1)

            if key == 27:
                break

            if key == ord("s"):
                with open(self.output_file_path, "wb") as f:
                    pickle.dump(self.__regions, f)
                print(f"saved to {self.output_file_path}")

            cv2.imshow("res", img)
            cv2.setMouseCallback("res", self.mouse_click)


if __name__ == '__main__':
    picker = Picker(regions_path="regions/dff_regions.pkl")
    # picker = Picker()
    picker.__temp_points = '1'
    picker.run(image_path=r"images/dff.png")
