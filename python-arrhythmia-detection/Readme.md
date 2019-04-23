
About
====
This repository concerns arrhythmia detection using machine learning, based on [damien's repository](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.).

The cool thing about the default application, is that it converts numerical ECG data pertaining to a person, to an image, which is then used to train the neural network.

Running the defaut code generates some images from the ECG data, which can be used as a dataset on the same code, since the default code accepts both numerical CSV data, and images.

I directed the code to generate the aforesaid images in [outputs/](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/tree/master/python-arrhythmia-detection/outputs).

Furthermore, these images (each pertaining to a person's ECG) are used as inputs in my [Brower-Ai-Arrythmia-Detection](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/) application. 
 
Purpose
====
The purpose of this exercise was mainly to use damien's saved neural network model, by converting it to web-friendly format, for usage in my [Brower-Ai-Arrythmia-Detection](https://github.com/JordanMicahBennett/Browser-Ai-Arrythmia-Detection/) application. 
 
test if damien's neural network worked properly.

Modifications
====
This repository represents a modified version of [damien's repository](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.).
The code was modified to simply store the predictions after the model was ran.

[My version of main.py](https://github.com/JordanMicahBennett/Brower-Ai-Arrythmia-Detection/blob/master/python-arrhythmia-detection/main.py) includes code to.

Here is the code I wrote below:

```python
"""
The block below written by Jordan to save predictions.

The resulting predictions.csv file has two columns:
A.] 'Person ECG Data': This column is the ECG image of each person's ECG data. Each entry corresponds to each value in example.csv, which contains an ECG nunber per person.
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

# Note: The 'ecgDataOutputDirectory' must match the one supplied above, after samples.csv is provided to the neural network in the prompt above.
ecgDataOutputDirectory = 'output/';

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

You may need to uninstall protobuf 3.7, if you had already installed protobuf 3.7 +. Instead, install protobuf 3.6.0 or 3.5.2.


