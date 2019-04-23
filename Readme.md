![Alt text](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection/blob/master/browser-arrhythmia-detection/data/banner/banner.png "default page")

Description
===
A scratch written, strictly browser Convolutional artificial neural network application for arrhythmia detection by Jordan Bennett, using tensorflow js, with reference to [the python equivalent of this machine learning model](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection/tree/master/python-arrhythmia-detection).

Although [the python-arryhthmia-detection version](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection/tree/master/python-arrhythmia-detection) is included [in this repository](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection), it is separate from the [browser arrythmia-detection version](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection/tree/master/browser-arrhythmia-detection), which does not require any python to work, as specified in [the technology stack section below](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection#technology-stack).

Read about some struggles below, to discover the benefits of using this repository
===
1. There are several python neural network codes that can diagnose arrhythmia. I explored [this one which uses tensorflow](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.).

    For testing purposes, I setup the code, and modified it to store the predictions it produced in a folder. 
    
    My modified python code: [Browser-Ai-Arrhythmia-Detection/python-arryhthmia-detection](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection/tree/master/python-arrhythmia-detection).
    
    From item 2 below and onwards, I explain the steps I took to construct the browser equivalent: [Browser-Ai-Arrythmia-Detection](https://github.com/JordanMicahBennett/Browser-Ai-Arrhythmia-Detection/tree/master/browser-arrhythmia-detection).

    The neural network above first converts sample person ECG numerical data to images, which are then used to train the neural network.

2. Fortunately, there is something called [tensorflow js](https://github.com/tensorflow/tfjs), since I seek to establish a neural network for arrhythmia diagnosis strictly on browser. (Browser version of python tensorflow)

3. Unfortunately, although tensorflow js can run saved tensorflow neural network models, it has to run those that have been converted from python to web friendly format.

   Notably, [there was a saved neural network model](https://drive.google.com/open?id=1WaenBnWYyhiumkvfaqEcDzti4S9aEuhS) from 1, but it is not the actual neural network from 1, instead it represents a configuration for the neural network. (For a visualization of the difference between the **configuration** and the **actual neural network**, try to picture **a model aka the neural network**, versus the **model's setup or features like number of neurons etc, aka the configuration**).

   The saved model captured the state of the Python neural network, after being trained for 60 hours on an Nvidia Tesla GPU, on arrhythmia data according to [the author of the original code](https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network.).

4. After a little bit of hacking, I managed to get [tensorflow js converter](https://github.com/tensorflow/tfjs-converter) working. (The      regular setup instructions were not sufficient to get the converter working)

    i. Based on the error reported while trying to invoke the converter, I figured out that the problem was that a file called  **_convert_to_constants.py_** was missing from my python tensorflow framework directory.

   ii. This problem was solved, by copying the [convert_to_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/convert_to_constants.py) found on tensorflow's official [web-page framework directory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/convert_to_constants.py), into my **Python36\Lib\site-packages\tensorflow\python\framework** folder.

   Using the command below, as instructred on the [tensorflow js converter page](https://github.com/tensorflow/tfjs-converter), I converted the saved python neural network from (1) to web friendly format.
```python
tensorflowjs_converter --input_format=keras C:/Users/Fleet2/Desktop/arrythmia-ml/damien/saved_keras_model/ecg_model_own.hdf5 C:/Users/Fleet2/Desktop/arrythmia-ml/damien/keras_model/
```
   
5. Since the saved neural network model is not the actual neural network, and it actually represents a configuration for a neural network, I still had to write a neural network in [tensorflow js](https://github.com/tensorflow/tfjs), for browser.

   After that I loaded the converted saved model from (4).

   At this point, in the browser app I wrote, I can take as input ECG data image pertaining to a person, and then determine whether          arrhythmia is present using the new tensorflow neural network written, while using the saved python tensorflow model from (4).


Technology stack
====
1. Tensorflow js

2. Jquery js

Note: This is strictly browser based, and doesn't require any special setup, apart from running on an http server, like xammp. I used xampp in particular.
