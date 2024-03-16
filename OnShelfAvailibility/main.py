from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Union, Tuple, List
import cv2
import pickle
import requests
import json
import numpy as np

from OnShelfAvailibility.detector import Detector
from OnShelfAvailibility.config import Config


@dataclass
class StoreShelfSystem:
    model_path: str
    conf_threshold: float = .3
    ultralitycs_path: str = "ultralytics/yolov5"
    model_type: str = "custom"
    force_reload: bool = True
    telegram_message: bool = True
    telegram_tokens_file: str = "telegram_tokens.json"
    show_detections: bool = False

    def __post_init__(self) -> None:
        self.detector = Detector(
            model_path=self.model_path,
            conf_threshold=self.conf_threshold,
            ultralitycs_path=self.ultralitycs_path,
            model_type=self.model_type,
            force_reload=self.force_reload
        )

        if self.telegram_message:
            with open(self.telegram_tokens_file) as f:
                data = json.load(f)
                self.__bot_token, self.__chat_id = data["BotToken"], data["ChatId"]

    @staticmethod
    def get_center(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
        cx = p1[0] + (abs(p1[0] - p2[0]) // 2)
        cy = p1[1] + (abs(p1[1] - p2[1]) // 2)

        return cx, cy

    @staticmethod
    def check_inside2p(point: Tuple[int, int], top_left_p: Tuple[int, int], bot_right_p: Tuple[int, int]) -> bool:
        if top_left_p[0] < point[0] < bot_right_p[0] and top_left_p[1] < point[1] < bot_right_p[1]:
            return True
        return False

    def send_tg_message(self, msg: str) -> None:
        url = f"https://api.telegram.org/bot{self.__bot_token}/sendMessage?" \
              f"chat_id={self.__chat_id}&parse_mode=Markdown&text={msg}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error when making sending tg message: {response.status_code}, {response.content}")

    @staticmethod
    def draw_region_info(img: np.array, region: list[Tuple[int, int]], region_label: str, region_id: int,
                  region_color: Tuple[int, int, int]) -> None:
        cv2.rectangle(img, region[0], region[1], region_color, 2)
        # cv2.putText(img, region_label, (region[0][0], region[0][1] + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, region_color, 2)
        cv2.putText(img, str(region_id), (region[0][0], region[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, region_color, 2)

    @staticmethod
    def draw_summary(img: np.array, states_perc: dict, start_cord: List[int], xstep: int = 25, ystep: int = 0):
        """
        :param img:
        :param states_perc: dictionary of percentage per status - product: 80% and etc
        :param start_cord:
        :param xstep: spacing between text for x axis
        :param ystep: spacing between text for y axis
        :return:
        """
        for state, perc in states_perc.items():
            cv2.putText(img, f"{state}: {perc}%", start_cord, cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
            start_cord[1] += xstep
            start_cord[0] += ystep

    @staticmethod
    def create_msg(statuses_indexes: Union[defaultdict, dict], base: str = "Shelf status summary:\n") -> str:
        for status, status_list in statuses_indexes.items():
            index_str = ", ".join(status_list)
            base += f"{status} positions ({len(status_list)}): {index_str}\n"

        return base

    def run_on_image(self, image_path: str, regions_pkl_file: str, resize: Union[tuple, bool] = False,
                     summary_rect_size: Tuple[float, float] = (.2, .13),
                     summary_rect_start_point: Tuple[int, int] = (20, 10), alpha: float = .3,
                     summary_start_point: Tuple[int, int] = (25, 35)) -> None:
        """
        :param image_path:
        :param regions_pkl_file:
        :param resize:
        :param summary_rect_size: value 0 < v < 1.0, first value is percentage took from width, 2nd from height (x * w, y * h)
        :param summary_rect_start_point:
        :param alpha: 0.0-1.0, the bigger value the more transparent summary rectangle
        :param summary_start_point:
        :return: None
        """

        with open(regions_pkl_file, "rb") as f:
            regions = pickle.load(f)

        states_perc = {"product": 0, "no product": 0, "unknown": 0}
        regions_dict = {region_id: {"Region": region, "Status": None} for region_id, region in enumerate(regions)}

        image = cv2.imread(image_path)
        if resize:
            image = cv2.resize(image, resize)

        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
        image_draw, res = self.detector.detect(img=converted)
        image_draw = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)

        for region_id, value in regions_dict.items():
            region = value["Region"]

            region_color = (0, 0, 0)
            label = "unknown"

            for row in res.iterrows():
                det = row[1]

                det_p1 = (int(det.xmin), int(det.ymin))
                det_p2 = (int(det.xmax), int(det.ymax))
                det_center = self.get_center(p1=det_p1, p2=det_p2)

                class_name = det.to_list()[-1]
                if self.check_inside2p(point=det_center, top_left_p=region[0], bot_right_p=region[1]):
                    if det["class"] == 1:
                        region_color = (0, 0, 200)
                        label = class_name
                    else:
                        region_color = (0, 200, 0)
                        label = class_name
                    break

            regions_dict[region_id]["Status"] = label
            self.draw_region_info(img=image, region=region,
                                  region_id=region_id, region_color=region_color,
                                  region_label=label)

        states = [value["Status"] for value in regions_dict.values()]
        states_len = len(states)
        counter = Counter(states)

        for s, v in counter.items():
            states_perc[s] = round((v * 100) / states_len, 1)

        overlay = image.copy()
        # Draw summary
        h, w, _ = image.shape
        summary_rect_w = int(w * summary_rect_size[0])
        summary_rect_h = int(h * summary_rect_size[1])
        cv2.rectangle(image, summary_rect_start_point, (summary_rect_w, summary_rect_h), (0, 0, 0), -1)
        final_img = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        self.draw_summary(img=final_img, states_perc=states_perc, start_cord=list(summary_start_point))
        cv2.rectangle(final_img, (summary_rect_start_point[0]-2, summary_rect_start_point[1]-2),
                      (summary_rect_w+2, summary_rect_h+2), (255, 255, 255), 2)
        #
        statuses_indexes = defaultdict(list)
        for index, details in regions_dict.items():
            status = details["Status"]
            statuses_indexes[status].append(str(index))

        if self.telegram_message:
            msg = self.create_msg(statuses_indexes=statuses_indexes)
            self.send_tg_message(msg=msg)

        cv2.imshow("MainWindow", final_img)
        if self.show_detections:
            cv2.imshow("Detections", image_draw)
        cv2.waitKey(0)


if __name__ == '__main__':
    store_shelf_system = StoreShelfSystem(model_path=Config.MODEL_FILE,
                                          conf_threshold=Config.CONF_THRESHOLD,
                                          telegram_tokens_file=Config.TELEGRAM_TOKENS_FILE,
                                          show_detections=Config.SHOW_DETECTION,
                                          telegram_message=Config.TELEGRAM_MESSAGE)
    store_shelf_system.run_on_image(image_path=Config.IMAGE_FILE, resize=Config.RESIZE,
                                    regions_pkl_file=Config.REGION_FILE)