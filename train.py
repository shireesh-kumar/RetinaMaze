import torch
import config
import utils
from datapreprocessing import RVDataModule
import pytorch_lightning as pl
from model import build_unet
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


print(f"Available GPUs: {torch.cuda.device_count()}")
device = torch.device('cuda')
utils.seeding(42)

wandb_logger = WandbLogger(project='retinal_vessel_segmentation')
# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = config.BATCH_SIZE

checkpoint_callback = ModelCheckpoint(
    monitor="Val_Epoch_Loss",  
    mode="min",          
    save_top_k=1,        
    dirpath="checkpoints/", 
    filename="best-checkpoint",
    verbose=True,
)

unet_model = build_unet()
unet_model = unet_model.to(device)
datamanager = RVDataModule(data_dir="dataset/", batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu',
    devices=1,
    min_epochs=1,
    max_epochs=config.NUM_EPOCHS,
    precision=32,
    enable_progress_bar=False,
    callbacks=[checkpoint_callback]
)

trainer.fit(model=unet_model,datamodule=datamanager)










