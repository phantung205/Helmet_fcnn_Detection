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
#split data
splits = ["train","val"]
train_ratio = 0.8
val_ratio = 0.2

categories = ["Without Helmet","With Helmet"]

batch_size = 4
num_worker=2
learning_rate = 1e-3
momentum = 0.9
num_epochs = 100

report_dir = os.path.join(base_dir,"reports")
path_tensorboard = os.path.join(report_dir,"tensorboard")


model_dir = os.path.join(base_dir,"trained_models")




