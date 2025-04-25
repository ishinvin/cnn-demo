# AlexNet CNN

## 1. What is AlexNet?

AlexNet is a Convolutional Neural Network (CNN) architecture designed by **Alex Krizhevsky** in 2012. It won the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**, significantly outperforming the competition and popularizing deep learning techniques in computer vision.

AlexNet was designed to classify images into predefined categories and consists of 8 layers (5 convolutional layers and 3 fully connected layers). It was trained on over a million images for image classification tasks.

---

## 2. Key Components of AlexNet

### Convolutional Layers

- Convolutional layers apply filters (kernels) to detect features like edges, textures, and objects in an image.
- AlexNet has **5 convolutional layers**, each detecting more complex features as you go deeper.

### Activation Function (ReLU)

- **ReLU (Rectified Linear Unit)** is an activation function that introduces non-linearity to the network. It speeds up training and helps the model learn complex patterns.
- Formula:  
  `f(x) = max(0, x)`

### Pooling Layers (Max Pooling)

- Pooling layers reduce the spatial dimensions (width and height) of the feature map, helping reduce computation and prevent overfitting.
- AlexNet uses **Max Pooling**, selecting the maximum value from each region in the feature map.

### Fully Connected Layers

- After convolution and pooling, fully connected layers are used to make final predictions based on the features detected.
- AlexNet has **3 fully connected layers** at the end of the network.

### Softmax Layer

- The final layer is a **Softmax** layer that converts the raw output into probabilities for each class, allowing the network to predict the class with the highest probability.

---

## 3. AlexNet Architecture

### Layer Breakdown:

1. **Input Layer**:

   - Input image size: **224x224x3** (224x224 RGB image).

2. **Convolutional Layer 1**:

   - Filter size: **11x11**, **stride 4**.
   - Output: **55x55x96**.

3. **ReLU Activation Layer**:

   - Applies ReLU activation to the output of Convolutional Layer 1.

4. **Max Pooling Layer 1**:

   - Pool size: **3x3**, **stride 2**.
   - Output: **27x27x96**.

5. **Convolutional Layer 2**:

   - Filter size: **5x5**, **stride 1**.
   - Output: **27x27x256**.

6. **ReLU Activation Layer**.

7. **Max Pooling Layer 2**:

   - Pool size: **3x3**, **stride 2**.
   - Output: **13x13x256**.

8. **Convolutional Layer 3**:

   - Filter size: **3x3**, **stride 1**.
   - Output: **13x13x384**.

9. **ReLU Activation Layer**.

10. **Convolutional Layer 4**:

    - Filter size: **3x3**, **stride 1**.
    - Output: **13x13x384**.

11. **ReLU Activation Layer**.

12. **Convolutional Layer 5**:

    - Filter size: **3x3**, **stride 1**.
    - Output: **13x13x256**.

13. **Max Pooling Layer 3**:

    - Pool size: **3x3**, **stride 2**.
    - Output: **6x6x256**.

14. **Fully Connected Layer 1**:

    - Input: **6x6x256 = 9216**.
    - Output: **4096** neurons.

15. **Fully Connected Layer 2**:

    - Output: **4096** neurons.

16. **Fully Connected Layer 3**:

    - Output: **1000** (number of ImageNet classes).

17. **Softmax Layer**:
    - Converts the output into class probabilities.

---

## 4. How AlexNet Works: A Step-by-Step Process

1. **Input Image**: An image of size 224x224x3 is fed into the model.
2. **Convolution**: The model applies convolutional filters to detect basic features.
3. **Activation (ReLU)**: Introduces non-linearity, enabling the model to learn complex patterns.
4. **Pooling**: Reduces the size of the feature maps while retaining important information.
5. **Repeat**: The process repeats through multiple convolutional and pooling layers.
6. **Fully Connected Layers**: The model uses these layers to make the final class prediction.
7. **Output**: The final layer outputs probabilities for each class, and the class with the highest probability is the prediction.

---

## 5. Key Concepts to Understand

### Overfitting

- AlexNet uses **Dropout** to avoid overfitting. Dropout randomly disables some neurons during training, helping the network generalize better.

- Dropout is a regularization technique used in deep learning models, particularly Convolutional Neural Networks (CNNs), to prevent overfitting.

- Overfitting occurs when a model performs well on the training data but fails to generalize to new, unseen data.

- Dropout addresses this issue by randomly “dropping out” (setting to zero) a fraction of neurons during training, forcing the network to learn more robust and generalizable features.

### Data Augmentation

- To prevent overfitting due to limited data, AlexNet uses **image augmentation** techniques like **rotation**, **flipping**, and **color jittering** to artificially increase the size of the training dataset.

### GPU Acceleration

- AlexNet was trained using **Graphics Processing Units (GPUs)**, which significantly sped up the training process compared to CPUs.

---

## Conclusion

AlexNet made a significant impact in the field of deep learning, proving that deep neural networks could outperform traditional models in computer vision tasks. By understanding AlexNet, you're gaining a foundational understanding of how CNNs work and how they can be used for tasks like image classification.
