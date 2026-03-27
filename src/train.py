import os.path
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src import config,dataloader
import argparse
import shutil
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def get_args():
    parser = argparse.ArgumentParser(description="train faster cnn model")
    parser.add_argument("--data_path","-d",type=str,default=config.data_processed_dir,help="root data")
    parser.add_argument("--num_epochs","-n",type=int,default=config.num_epochs,help="Number of epochs")
    parser.add_argument("--batch_size","-b",type=int,default=config.batch_size,help="batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=config.learning_rate, help=" learning rate for optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=config.momentum, help="momentum optimizer")
    parser.add_argument("--log_folder", "-p", type=str, default=config.path_tensorboard, help="path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default=config.model_dir, help="path to save checkpoint")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="continue from this checkpoint")
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader = dataloader.helmet_dataloader(root=args.data_path,batch_size=args.batch_size)

    # use model faster cnn
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
                                                  trainable_backbone_layers=config.train_backbone)
    # replace num class for faster cnn train to num class for data helmet
    in_channels =  model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels,
                                                      num_classes=len(config.categories)+1)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1
    )

    # load model
    if args.saved_checkpoint:
        checkpoint = torch.load(args.saved_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["map"]
        print(start_epoch)
    else:
        start_epoch = 0
        best_map = -1


    # remove tensorboard old
    if os.path.isdir(args.log_folder):
        shutil.rmtree(args.log_folder)

    #create fodel save checkpoint modl
    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    writer = SummaryWriter(args.log_folder)

    num_iters_per_epoch = len(train_dataloader)
    for epoch in range(start_epoch,args.num_epochs):
        #training
        model.train()
        progress_bar = tqdm(train_dataloader,colour="cyan")
        train_loss = []
        for iter ,(images,labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            labels =  [{"boxes":target["boxes"].to(device),"labels":target["labels"].to(device)} for target in labels]

            # forwark
            losses= model(images,labels)
            final_losses = sum([loss for loss in losses.values()])

            #backward
            optimizer.zero_grad()
            final_losses.backward()

            # 🔥 chống nổ gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            train_loss.append(final_losses.item())
            mean_loss = np.mean(train_loss)

            progress_bar.set_description("epoch {}/{}. loss {:0.4f}".format(epoch+1,args.num_epochs,mean_loss))

            writer.add_scalar("train/loss",mean_loss,epoch*num_iters_per_epoch + iter)
        # 🔥 update learning rate
        scheduler.step()

        model.eval()
        progress_bar = tqdm(val_dataloader,colour="yellow")
        metric = MeanAveragePrecision(iou_type="bbox")
        for iter,(images,labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)

            preds = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu")
                })
            targets = []
            for label in labels:
                targets.append({
                    "boxes": label["boxes"],
                    "labels": label["labels"]
                })
            metric.update(preds, targets)
        result = metric.compute()
        # pprint(result)
        writer.add_scalar("val/mAP", result["map"], epoch)
        writer.add_scalar("val/mAP_50", result["map_50"], epoch)
        writer.add_scalar("val/mAP_75", result["map_75"], epoch)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "map": result["map"],
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_folder, "last.pt"))
        if result["map"] > best_map:
            best_map = result["map"]
            torch.save(checkpoint, os.path.join(args.checkpoint_folder, "best.pt"))

if __name__ == '__main__':
    args = get_args()
    train(args)