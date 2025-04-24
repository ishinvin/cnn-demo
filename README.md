# CNN Demo

A simple Convolutional Neural Network implementation for a sharing session.<br>
This is an AlexNet implementation of glasses/no-glasses classification.

## Setup Project

### Create a virtual environment (optional)

```shell
python3 -m venv venv
```

### Activate the virtual environment

- On macOS/Linux:

```shell
source venv/bin/activate
```

- On Windows:

```shell
venv\Scripts\activate
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Train Model

```shell
python train.py
```

### View TensorBoard

```shell
tensorboard --logdir=runs
```

- Once TensorBoard starts, open http://localhost:6006

### Test Model

```shell
python eval.py imgs/glasses.jpg
```

### Test Webcam

```shell
python camera.py
```

### Deactivate the virtual environment

```shell
deactivate
```
