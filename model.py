#dataloader
"""
    This code implements a training and validation pipeline for an image classification model using the PyTorch and PyTorch-Ignite libraries.
    It includes the following key components:
    1.Dataset handling and data loading using DataLoader.
    2.Model definition with a modified ResNet50 architecture.
    3.Training and validation setup using PyTorch-Ignite.
    4.Logging and checkpointing with Tensorboard and Ignite handlers.
"""
from datasets import ImageDataset
from transforms import  get_img_transform

import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision.models import resnet18
import ignite
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlershandlers import TensorboardLogger, global_step_from_engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #ensures the code runs on GPU if available, falling back to CPU otherwise.
# BATCH_SIZE=16

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.model = resnet50(num_classes=8)

        self.model.conv1 = self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        return self.model(x)


model = Net().to(device)


train_loader = DataLoader(
    ImageDataset(csv_path=r'C:\Users\Dell\Desktop\grandchallenge\data\processed_train_.csv' ,transforms= get_img_transform(img_size=(224, 224))),batch_size=5,shuffle=True)


val_loader = DataLoader(
    ImageDataset(csv_path=r'C:\Users\Dell\Desktop\grandchallenge\data\processed_val-5K.csv',transforms=get_img_transform(img_size=(224, 224))),batch_size=5,shuffle=False
)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)




log_interval = 100

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


def score_function(engine):
    return engine.state.metrics["accuracy"]




model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
    require_empty= False
)

val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

trainer.run(train_loader, max_epochs=5)

tb_logger.close()


