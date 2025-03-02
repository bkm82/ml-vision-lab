# Set up logging
import logging
import math
import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import load, save
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_n_params(module):
    return sum([math.prod(list(p.shape)) for p in module.parameters()])


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_channels = 1  # 1 for black/white, 3 for color
        n_pixels = 28  # 28 + 2 padding (either end) for minst
        convolution_channel_list = [input_channels, 32, 64, 64]
        # Should a maxpool layer happen after the conv list.
        maxpool_list = [False, False, False]  # size (conv_channel_list -1)
        # TODO each maxpool takes the size / 2, i.e 32 -> 16 -> 8 ...
        # There should be  a check to verify the size doesnt get below
        # a certain point
        kernel = (3, 3)
        stride = 1
        padding = 0
        linear_hidden = 5
        output_classes = 10

        convolution_layers = []

        # Create a conv2d and ReeLU layers based on the channel list
        for i in range(len(convolution_channel_list) - 1):
            in_channels = convolution_channel_list[i]
            out_channels = convolution_channel_list[i + 1]
            convolution_layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            logger.debug(f"i:{i} in:{in_channels} out:{out_channels}")

            convolution_layers.append(torch.nn.ReLU())
            if maxpool_list[i]:
                convolution_layers.append(torch.nn.MaxPool2d(kernel_size=2))

        convolution_layers.append(torch.nn.Flatten())

        self.convolution_sequential = torch.nn.Sequential(*convolution_layers).to(device)

        # Get the shape of the convolution output automatically
        # By creating a  tensor of the correct image and creating
        # the convolutional_seq
        shape_numpy = np.zeros([1, input_channels, n_pixels, n_pixels])
        shape_tensor = torch.from_numpy(shape_numpy).float().to(device)
        conv_seq_out = self.convolution_sequential(shape_tensor)
        n_data, conv_hidden = conv_seq_out.shape
        logger.debug(f"conv_hidden: {conv_hidden}")

        self.linear_sequential = torch.nn.Sequential(
            torch.nn.Linear(conv_hidden, linear_hidden), torch.nn.ReLU(), torch.nn.Linear(linear_hidden, output_classes)
        )

        self.sequential = torch.nn.Sequential(self.convolution_sequential, self.linear_sequential).to(device)

        conv_params = get_n_params(self.convolution_sequential)
        linear_params = get_n_params(self.linear_sequential)
        total_params = get_n_params(self.sequential)
        logger.debug(f"Convoluted = {conv_params}, Linear = {linear_params} Total= {total_params}")

    def forward(self, x):
        return self.sequential(x).to(device)


class ConvNet_Learner:
    def __init__(self):
        self.learning_rate = 1e-3
        self.classifier = ConvNet().to(device)
        self.optimizer = Adam(self.classifier.parameters(), self.learning_rate)
        self.loss_function = torch.nn.CrossEntropyLoss()
        # Paramaters

    def take_step(self, inputs, labels):
        inputs, labels = inputs.to(device), labels.to(device)
        predicitons = self.decision_function(inputs)
        loss = self.loss_function(predicitons, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def decision_function(self, inputs):
        """Take inputs and return a prediction

        TODO: Add typing for input shape and types
        """

        return self.classifier(inputs)

    def validation_loss(self, validation_dataset):
        running_val_loss = 0
        for batch in validation_dataset:
            X, y = batch
            X, y = X.to(device), y.to(device)
            predictions = self.decision_function(X)
            running_val_loss += self.loss_function(predictions, y)
        val_loss = running_val_loss / len(validation_dataset)
        return val_loss

    def fit(self, training_dataset, validation_dataset=None):
        max_epocs = 20
        for epoch in range(max_epocs):
            logger.info(f"Starting Epoch {epoch}")

            running_loss = 0.0

            for batch in training_dataset:
                X, y = batch

                X, y = X.to(device), y.to(device)
                running_loss += self.take_step(inputs=X, labels=y)

            epoch_loss = running_loss / len(training_dataset)
            logger.debug(f"Epoch {epoch} loss is {epoch_loss}")
            if validation_dataset:
                validataion_loss = self.validation_loss(validation_dataset=validation_dataset)
                logger.debug(f"validataion_loss: {validataion_loss}")

    def save_model(self):
        with open("model_state.pt", "wb") as f:
            save(self.classifier.state_dict(), f)


def load_dataset():
    batch_size = 100
    subset = 0.50
    validation_split = 0.20
    random_seed = 4

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    full_train = datasets.MNIST(root="data", download=True, train=True, transform=transform)

    # Down sample the original dataset for faster training
    sample_size = int(len(full_train) * subset)
    _, sampled_indices = train_test_split(
        range(len(full_train)), test_size=sample_size, stratify=full_train.targets, random_state=random_seed
    )

    sampled_dataset = Subset(full_train, sampled_indices)

    # Split the data into train and validation sets
    (
        train_indices,
        val_indices,
        _,
        _,
    ) = train_test_split(
        range(len(sampled_dataset)),
        sampled_dataset.dataset.targets[sampled_indices],
        stratify=sampled_dataset.dataset.targets[sampled_indices],
        test_size=validation_split,
        random_state=random_seed,
    )

    train_split = Subset(sampled_dataset, train_indices)
    validation_split = Subset(sampled_dataset, val_indices)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)

    validation_loader = DataLoader(validation_split, batch_size=batch_size)
    logger.debug(f"train shape: {len(train_loader.dataset)} val shape:{len(validation_loader.dataset)}")
    return (train_loader, validation_loader)


def train():
    train_dataset, validation_dataset = load_dataset()
    learner1 = ConvNet_Learner()
    learners = [learner1]
    for learner in learners:
        learner.fit(training_dataset=train_dataset, validation_dataset=validation_dataset)
        learner.save_model()


# example api
# def main():
#  learner1 = KNN_Learner()
#  learner2 = ConvNet_Learner()
#  learners = [learner1, learner2]
#  train_data = train_dataset
#  for learner in learners:
#     learner.train(data = train_data, k_fold)


def main():
    logger.info(f"This is running on the {device}")
    # Instance of neural network, loss, optimizer
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    clf = ConvNet().to(device)
    # opt = Adam(clf.parameters(), lr=1e-3)
    # loss_fn = nn.CrossEntropyLoss()
    # train()

    current_file_path = os.path.dirname(__file__)
    classifier_path = os.path.join(current_file_path, "model_state.pt")
    with open(classifier_path, "rb") as f:
        clf.load_state_dict(load(f, map_location=device))
    current_file_path = os.path.dirname(__file__)

    clf.eval()
    project_root = os.path.dirname(os.path.dirname(current_file_path))

    img_path = os.path.join(project_root, "examples/test/img_3.jpg")

    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # make a prediction
        outputs = clf(img_tensor)
        #
        probability = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probability.topk(1, dim=1)
        # _, predicted = torch.max(outputs, 1)
        logger.info(f"I am {top_p.item():.2f}% sure that is a {top_class.item()}")
        # print(f"The predicted class is: {predicted.item()}")


if __name__ == "__main__":
    main()
