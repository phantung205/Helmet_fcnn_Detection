from torch.utils.data import DataLoader
from src.dataset import HelmetDataset
from torchvision.transforms import ToTensor,Compose,Normalize,RandomAffine,ColorJitter
from src import config

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def helmet_dataset():
    transform_train = Compose([
        # RandomAffine(
        #     degrees=(-5, 5),
        #     translate=(0.15, 0.15),
        #     scale=(0.85, 1.15),
        #     shear=10
        # ),
        ColorJitter(
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
        ToTensor(),
    ])
    transform_val = Compose([
        ToTensor()
    ])
    # data train
    train_dataset = HelmetDataset(root=config.processed_train,transform=transform_train)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_worker,
        collate_fn=collate_fn
    )

    #data val
    val_dataset = HelmetDataset(root=config.processed_val, transform=transform_val)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_worker,
        collate_fn=collate_fn
    )

    return train_dataloader,val_dataloader