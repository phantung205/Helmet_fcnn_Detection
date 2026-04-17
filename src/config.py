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
#path image test
image_test = os.path.join(data_dir,"test","t.jpg")
video_test = os.path.join(data_dir,"test","p3.mp4")

#----------------------
#save result image and video
#----------------------
dir_results = os.path.join(base_dir,"results")

#--------------------
#parameter
#--------------------
#split data
splits = ["train","val"]
train_ratio = 0.9
val_ratio = 0.1
categories = ["Without Helmet","With Helmet"]
batch_size = 6
num_worker=2
learning_rate = 1e-4
momentum = 0.9
num_epochs = 100
train_backbone = 5

#-----------------------
# directory report
#-----------------------
report_dir = os.path.join(base_dir,"reports")
path_tensorboard = os.path.join(report_dir,"tensorboard")

#-------------------------
# directory model
#------------------------
model_dir = os.path.join(base_dir,"trained_models")
checkpoint_best = os.path.join(model_dir,"best.pt")
checkpoint_last = os.path.join(model_dir,"last.pt")





