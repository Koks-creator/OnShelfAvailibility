import os


ROOT_FOLDER = "."
MAIN_FOLDER = "train_data"
SUB_FOLDERS = {
    "images": ["train", "val"],
    "labels": ["train", "val"],
}

main_folder_path = f"{ROOT_FOLDER}/{MAIN_FOLDER}"
if not os.path.exists(main_folder_path):
    os.mkdir(main_folder_path)

for folder, sub_folders in SUB_FOLDERS.items():
    path_to_create = None
    for sub_folder in sub_folders:
        try:
            path_to_create = f"{ROOT_FOLDER}/{MAIN_FOLDER}/{folder}/{sub_folder}"
            os.makedirs(path_to_create)
        except FileExistsError:
            print(f"Path: {path_to_create} already exists")


#labelimg C:\Users\table\PycharmProjects\test2\OnShelfAvailibility\train_data\images\train C:\Users\table\PycharmProjects\test2\OnShelfAvailibility\train_data\labels\train\classes.txt C:\Users\table\PycharmProjects\test2\OnShelfAvailibility\train_data\labels\train
