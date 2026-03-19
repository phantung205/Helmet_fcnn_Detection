import os.path

from src import config
import random
import shutil


# create folder
def create_folders():
    for split in config.splits:
        path_images = os.path.join(config.data_processed_dir,split,"images")
        path_anno = os.path.join(config.data_processed_dir,split,"labels")
        if not os.path.isdir(path_images):
            os.makedirs(path_images)
        if not os.path.isdir(path_anno):
            os.makedirs(path_anno)

# split data
def split_data():
    images = [f for f in os.listdir(config.image_raw_dir) if f.endswith(".png")]
    random.shuffle(images)

    total = len(images)
    train_end = int(total*config.train_ratio)
    val_end = train_end + int(total * config.val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]

    return train_files,val_files

#copy file
def copy_files(file_list,split):
    for file in file_list:
        # copy label to processor
        label_name = file.replace(".png", ".xml")
        label_src = os.path.join(config.anno_raw_dir, label_name)
        label_dst = os.path.join(config.data_processed_dir, split, "labels", label_name)
        #Check if there is a label
        if not os.path.exists(label_src):
            continue
        shutil.copy(label_src, label_dst)

        # copy image to processor
        img_src = os.path.join(config.image_raw_dir,file)
        img_dst = os.path.join(config.data_processed_dir,split,"images",file)
        shutil.copy(img_src, img_dst)



if __name__ == '__main__':
    create_folders()
    train_files, val_files = split_data()
    copy_files(train_files, "train")
    copy_files(val_files, "val")