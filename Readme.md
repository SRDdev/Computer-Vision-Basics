# Computer Vision Basics

This project provides a basic introduction to computer vision concepts and techniques through a series of assignments.

## Project Structure

*   `Assignment_1.ipynb`: CNN Visualization.
*   `Assignment_2.ipynb`: Object Detection using CNN.
*   `Assignment_3.ipynb`: Text Classification using RNN.
*   `Assignment_4.ipynb`: Autoencoder for MNIST dataset.
*   `data/`: (Optional) Contains any data files needed for the project.
*   `requirements.txt`: Lists the Python packages required to run the assignments.
*   `Readme.md`: This file, providing an overview of the project.

## Assignments

### Assignment 1: CNN Visualization

*   **Objective**: Visualize feature maps of convolutional layers in a CNN.
*   **Description**: Loads a pre-trained CNN model (or defines a simple CNN), passes an image through the model, and visualizes feature maps of convolutional layers.

### Assignment 2: Object Detection using CNN

*   **Objective**: Implement CNN-based architecture for object detection.
*   **Description**: Uses TensorFlow and Keras to load a pre-trained model (e.g., SSD MobileNet), load an image, perform object detection, and draw bounding boxes on the image.

### Assignment 3: Text Classification using RNN

*   **Objective**: Implement a Recurrent Neural Network (RNN) for text classification.
*   **Description**: Uses PyTorch and Hugging Face datasets to load and preprocess text data, define an RNN model, train the model, and evaluate its performance.

### Assignment 4: Autoencoder for MNIST Dataset

*   **Objective**: Build an Autoencoder for the MNIST dataset.
*   **Description**: Uses PyTorch to load and preprocess the MNIST dataset, define an Autoencoder architecture, train the Autoencoder, and visualize the results.

## How to Run

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd Computer-Vision-Basics
    ```
2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Jupyter Notebooks:

    *   Open the desired assignment (`Assignment_1.ipynb`, `Assignment_2.ipynb`, `Assignment_3.ipynb`, `Assignment_4.ipynb`) in Jupyter Notebook or JupyterLab.
    *   Execute the cells in the notebook sequentially.

## Dependencies

The project uses the following Python libraries:

*   tensorflow
*   tensorflow-hub
*   opencv-python
*   numpy
*   matplotlib
*   torch
*   torchvision
*   transformers
*   datasets
*   tqdm
*   Pillow

These dependencies are listed in the `requirements.txt` file.

## License

[Specify the license, e.g., MIT]
