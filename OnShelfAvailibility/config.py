class Config:
    IMAGES_FOLDER: str = "images"
    IMAGE_FILE: str = fr"{IMAGES_FOLDER}/cc.png"
    REGIONS_FOLDER: str = "regions"
    REGION_FILE: str = fr"{REGIONS_FOLDER}/cc_regions.pkl"
    MODEL_FOLDER: str = "model"
    MODEL_FILE: str = rf"{MODEL_FOLDER}/best.pt"
    CONF_THRESHOLD: float = .3
    TELEGRAM_MESSAGE: bool = True
    TELEGRAM_TOKENS_FILE: str = "telegram_tokens.json"
    SHOW_DETECTION: bool = True
    RESIZE: tuple = (1280, 720)
