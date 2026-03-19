import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#-----------------------
# path data
#-----------------------
data_dir = os.path.join(base_dir,"data")
# path data raw
image_raw_dir = os.path.join(data_dir,"raw","images")
anno_raw_dir = os.path.join(data_dir,"raw","annotations")
# path data processed
data_processed_dir = os.path.join(data_dir,"processed")
processed_train = os.path.join(data_processed_dir,"train")
processed_val = os.path.join(data_processed_dir,"val")
#split data
splits = ["train","val"]
train_ratio = 0.8
val_ratio = 0.2

categories = ["Without Helmet","With Helmet"]

batch_size = 4
num_worker=2




