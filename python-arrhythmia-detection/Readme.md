
About
====
This repository by Jordan Bennett concerns arrhythmia detection using machine learning, based on [damien's repository](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.).

The cool thing about the [default application](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.), is that it converts numerical ECG data ([found in examples.csv](https://github.com/JordanMicahBennett/Brower-Ai-Arrythmia-Detection/blob/master/python-arrhythmia-detection/example.csv)) pertaining to a person, to an image, which is then used to train the neural network, [based on this scientific paper](https://arxiv.org/pdf/1804.06812.pdf).

Running the defaut code generates some images from the numerical ECG data ([found in examples.csv](https://github.com/JordanMicahBennett/Brower-Ai-Arrythmia-Detection/blob/master/python-arrhythmia-detection/example.csv)) which can be used as a dataset on the same code, since the default code accepts both numerical CSV data, and images.

I directed the code to generate the aforesaid images in [outputs/](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/tree/master/python-arrhythmia-detection/outputs).

Furthermore, these images (each pertaining to a person's ECG) are used as inputs in my [Brower-Ai-Arrythmia-Detection](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/) application. 
 
Purpose
====
The purpose of this exercise was mainly to use damien's saved neural network model, by converting it to web-friendly format, for usage in my [Brower-Ai-Arrythmia-Detection](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/) application. 
 
Notably, [there was a saved neural network model](https://drive.google.com/open?id=1WaenBnWYyhiumkvfaqEcDzti4S9aEuhS) in the [default code](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.) but it is not the actual neural network from said deault code, instead it represents a configuration for the neural network. (For a visualization of the difference between the **configuration** and the **actual neural network**, try to picture **a model aka the neural network**, versus the **model's setup or features like number of neurons etc, aka the configuration**).

The saved model captured the state of the Python neural network from the [default code](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.), after being trained for 60 hours on an Nvidia Tesla GPU, on arrhythmia data according to the author of the default code.

This is good, since this means I don't have to spend 60 hours to train the model to learn how to detect arrhythmia, since I can used the saved neural network configuration. Another big plus is that I don't need to spend [thousands of US dollars to buy the expensive GPU used to train the default neural network](https://www.amazon.com/Nvidia-Tesla-GDDR5-Cores-Graphic/dp/B00Q7O7PQA).

The only slight-drawback is that [I still had to write a tensorflow js neural network](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/), that utilizes [the saved python neural network model](https://drive.google.com/open?id=1WaenBnWYyhiumkvfaqEcDzti4S9aEuhS) from the default code.

The test of damien's default neural network worked properly, which motivated the usage of his saved configuration in my [Brower-Ai-Arrythmia-Detection](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/) application.


Modifications
====
This repository represents a modified version of [damien's repository](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.).
The code was modified to simply store the predictions after the model was ran.

[My version of main.py](https://github.com/JordanMicahBennett/Brower-Ai-Arrythmia-Detection/blob/master/python-arrhythmia-detection/main.py) includes code to simply store the predictions [which I stored in results/predictions.csv](https://github.com/JordanMicahBennett/Brower-Ai-Arrythmia-Detection/tree/master/python-arrhythmia-detection/results), so I could roughly evaluate whether damien's model was working.

Here is the code I wrote below:

```python
"""
The block below written by Jordan to save predictions.

The resulting predictions.csv file has two columns:
A.] 'Person ECG Data': This column is the ECG image of each person's ECG data. Each entry corresponds to a set of values in example.csv, where there are sets of ECG numbers per person.
B.] 'Prediction': This column is the prediction of each person's ECG data. Each prediction is a class, from the ones specified on the original github repository.

The possible classes are: 
        0. Atrial premature contraction beat (APC)
        1. Normal, healthy
        2. Left bundle branch block beat (LBB)
        3. Paced beat (PAB)
        4. Premature ventricular contraction beat (PVC)
        5. Right bundle branch block beat (RBB)
        6. Ventricular escape beat (VEB)

Notice that results/predictions.csv file has mostly 1's in the prediction column.
The 1 there signifies that the person has been detected to be normal.
The 4 there signifies that the person probably has a problem.
"""
import os
import pandas as pd

# Note: The 'ecgDataOutputDirectory' must match the one supplied above, after samples.csv is provided to the neural network in the prompt above.
ecgDataOutputDirectory = 'outputs/';

predictionCount = 0

predictionResults = []

for file in os.listdir(ecgDataOutputDirectory):
    predictionResults.append((file,pred_li[predictionCount]))
    predictionCount += 1

df = pd.DataFrame(predictionResults,columns=['Person ECG Data', 'Prediction'])

os.mkdir('results/')

df.to_csv('results/predictions.csv')
```

Requirements
====

1. Python 3.6

    i. Numpy
 
    ii. Biosppy
 
    iii. Keras
 
    iv. Opencv_python
 
    v. Cv2
 
    vi. Pandas

You may need to uninstall protobuf 3.7, if you had already installed protobuf 3.7 +. Instead, install protobuf 3.6.0 or 3.5.2.


