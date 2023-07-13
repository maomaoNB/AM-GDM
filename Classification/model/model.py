from pickletools import optimize
import pytorch_lightning as pl
from module.vgg16 import VGG16
from torch import nn, optim
from main import instantiate_from_config


class Classification(pl.LightningModule):
    def __init__(self, model, data_key="data",
                 pred_key="result", **kwargs):
        super().__init__()
        # self.model = VGG16(**c_config)
        self.model = instantiate_from_config(model)
        self.loss = nn.BCELoss()
        self.pred_key = pred_key
        self.data_key = data_key
        self.accuracy = pl.metrics.Accuracy()
        
    def get_input(self, batch):
        data = batch[self.data_key]
        pred = batch[self.pred_key]  
        if len(data.shape) == 2:
            data = data.unsqueeze(1)
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(1)
        return pred, data
    
    def forward(self, data):
        pred_hat = self.model(data)
        return pred_hat
        
    def training_step(self, batch, batch_idx):
        pred, data = self.get_input(batch)
        pred_hat = self(data)
        loss = self.loss(pred_hat, pred)
        acc = self.accuracy(pred_hat, pred)
        self.log("train/pred_loss", loss, prog_bar=True, 
                 logger=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, on_epoch=True, on_step=False)
        return loss
        
    def validation_step(self, batch, batch_idx):
        pred, data = self.get_input(batch)
        pred_hat = self(data)
        loss = self.loss(pred_hat, pred)
        acc = self.accuracy(pred_hat, pred)
        self.log("val/pred_loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True, on_step=False)
        return loss
        
    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-5)
        return optimizer
