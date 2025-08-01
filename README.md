# Image Recognition Model: Hotdog or Not Hotdog?

## Overview
This project implements an image recognition model using Python and PyTorch to classify images as either 'hotdog' or 'not hotdog'. Leveraging transfer learning with a pre-trained ResNet18 model, the project demonstrates a practical application of deep learning for image classification.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Tools and Libraries](#tools-and-libraries)
- [Model Definition](#model-definition)
- [Training Process](#training-process)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Have you ever wanted to create an image detector that would tell you whether or not a picture is of a certain object? This project addresses a classic image classification problem: distinguishing between images of hotdogs and non-hotdogs. It explores the power of deep learning, specifically convolutional neural networks (CNNs) and transfer learning, to achieve this task. Unlike traditional programming approaches that rely on explicit rules, this model learns to recognize patterns directly from a large dataset of real-world images.

## Project Structure
The project is primarily contained within a single Jupyter Notebook, `Train-an-Image-Recognition-Model-with-Python(1).ipynb`, which guides the user through the entire process from environment setup and data preparation to model training and evaluation. 

## Environment Setup
To run this project, you will need a Python environment with PyTorch installed. The notebook begins by ensuring PyTorch is up-to-date. While a GPU is beneficial for faster training, the project can also run on a CPU. The notebook includes a check for GPU availability.

## Dataset
The model is trained on the 


Hotdog Not Hotdog Dataset, which is automatically downloaded by the provided `HotDogDataset` class. This dataset contains images labeled as either "hotdog" or "not_hotdog". The dataset metadata, including labels and file names, is stored in a CSV file. The `HotDogDataset` class handles the loading, transformation, and matching of images to their respective labels.

Key columns in the dataset CSV:
| Column    | Values                   | Description                                                                                                                       |
| --------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Label     | "hotdog" or "not_hotdog" | Describes whether or not an image is of a hotdog.                                                                                 |
| y         | "0" or "1"               | Numerical representation of the label, i.e. "0": denotes that an image is of a hotdog, "1": denotes the image is not of a hotdog. |
| file_name | Various Unique Values    | Denotes the path to the images, e.g. `hotdognothotdogfull/hotdog_1.jpg`.                                                          |

The dataset is split into training, validation, and testing sets to ensure robust model evaluation. The default split is 70% for training, 15% for validation, and 15% for testing. Image preprocessing, including resizing, random horizontal flips, random rotations, conversion to tensor, and normalization, is applied to augment the dataset and improve model generalization.

## Tools and Libraries
This project utilizes several key Python libraries for deep learning, image processing, and data visualization:

- **PyTorch**: The primary deep learning framework used for building, training, and evaluating the neural network model.
- **PIL (Pillow)**: Used for manipulating image file types.
- **Matplotlib**: Employed for plotting and data visualization, particularly for displaying images and training statistics.
- **Numpy**: Utilized for numerical operations, especially with arrays and matrices.
- **tqdm**: Provides progress bars to visualize the training process.
- **torchvision**: Used for image transformations and accessing pre-trained models.

Helper functions and classes are defined within the notebook to streamline the process:
- `imshow(inp: Tensor)`: A utility function to display images from tensors.
- `HotDogDataset(Dataset)`: A custom PyTorch `Dataset` class for loading and preprocessing the hotdog/not hotdog images.
- `ClassificationModelTrainer`: A comprehensive class designed to train, validate, and test the classification model, providing methods for tracking loss and accuracy.

## Model Definition
The model employs transfer learning based on the **ResNet18** architecture, pre-trained on the ImageNet dataset. This approach leverages the knowledge gained from a large, diverse dataset, significantly reducing the training time and data requirements for our specific task.

Since the original ResNet18 model is designed for 1,000 classes, the final layer (`model.fc`) is replaced with a new `nn.Linear` layer configured for 2 classes (hotdog and not hotdog). The parameters of the pre-trained layers are frozen to prevent their weights from being updated during initial training, focusing the learning on the newly added classification layer.

## Training Process
The training process involves:
1.  **Loss Function**: `CrossEntropyLoss` is used as the minimizing criterion, a common choice for multi-class classification problems.
2.  **Optimizer**: The **Adam optimizer** is selected to adjust model parameters and minimize the loss function.
3.  **Training Framework**: The `ClassificationModelTrainer` class orchestrates the training, handling data loading, forward and backward passes, and parameter updates.

The model is trained over multiple epochs. Due to computational constraints, a pre-trained model (25 epochs) is loaded, and then additional epochs are run to fine-tune the model. Training and validation metrics (loss and accuracy) are tracked and plotted to monitor the model's performance over time.

## Results
The model achieved an accuracy rate of **86.09%** on images it had never seen before (the test set). This indicates a strong ability to generalize and correctly classify hotdogs and non-hotdogs.

## Usage
To use this project:
1.  Clone the repository.
2.  Ensure you have Python and Jupyter installed.
3.  Install the required libraries: `pip install torch torchvision matplotlib numpy pandas requests tqdm pillow`
4.  Open the `Train-an-Image-Recognition-Model-with-Python(1).ipynb` notebook in Jupyter and run the cells sequentially.

## Contributing
Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.



