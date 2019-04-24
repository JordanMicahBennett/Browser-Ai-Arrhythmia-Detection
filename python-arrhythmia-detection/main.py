Img_or_csv = input('Do you have a csv file or do you have Images. Press I for Image and C for csv: ')
if Img_or_csv == 'C':
	import csv_to_image
	from csv_to_image import directory
elif Img_or_csv == 'I':
	directory = input('Enter the directory where Images are present: ')
from glob import glob
from keras.models import load_model
import numpy as np
from collections import Counter

model = load_model('C:/Users/Fleet2/Desktop/arrythmia-ml/damien/ecg_model_own.hdf5')

images = glob(directory + '/*.png')

pred_li = []
import cv2
for i in images:
    image = cv2.imread(i)
    pred = model.predict(image.reshape((1, 128, 128, 3)))
    
    y_classes = pred.argmax(axis=-1)
    pred_li.append(y_classes[0])    

most_common,num_most_common = Counter(pred_li).most_common(1)[0]

print("MOST COMMON")
if(most_common == 0):
	print('The patient may have atrial premature contraction beat (APC).')
	print('Number of beats of APC tpye are ' + str(num_most_common) + ' out of ' + str(len(images)))

elif(most_common == 1):
	print('The patient may be healthy.')
	print('Number of beats of healthy tpye are ' + str(num_most_common) + ' out of ' + str(len(images)))

elif(most_common == 2):
	print('The patient may have left bundle branch block beat (LBB).')
	print('Number of beats of LBB tpye are ' + str(num_most_common) + ' out of ' + str(len(images)))

elif(most_common == 3):
	print('The patient may have paced beat (PAB).')
	print('Number of beats of PAB tpye are ' + str(num_most_common) + ' out of ' + str(len(images)))

elif(most_common == 4):
	print('The patient may have premature ventricular contraction beat (PVC).')
	print('Number of beats of PVC tpye are ' + str(num_most_common) + ' out of ' + str(len(images)))

elif(most_common == 5):
	print('The patient may have right bundle branch block beat (RBB).')
	print('Number of beats of RBB tpye are ' + str(num_most_common) + ' out of ' + str(len(images)))

elif(most_common == 6):
	print('The patient may have ventricular escape beat (VEB).')
	print('Number of beats of VEB tpye are ' + str(num_most_common) + ' out of ' + str(len(images)))

try:
	second_most_common,num_second_most_common = Counter(pred_li).most_common(2)[1]
	print("\n\nSECOND MOST COMMON")
	if(second_most_common == 0):
		print('The patient may have atrial premature contraction beat (APC).')
		print('Number of beats of APC tpye are ' + str(num_second_most_common) + ' out of ' + str(len(images)))

	elif(second_most_common == 1):
		print('The patient may be healthy.')
		print('Number of beats of healthy tpye are ' + str(num_second_most_common) + ' out of ' + str(len(images)))

	elif(second_most_common == 2):
		print('The patient may have left bundle branch block beat (LBB).')
		print('Number of beats of LBB tpye are ' + str(num_second_most_common) + ' out of ' + str(len(images)))

	elif(second_most_common == 3):
		print('The patient may have paced beat (PAB).')
		print('Number of beats of PAB tpye are ' + str(num_second_most_common) + ' out of ' + str(len(images)))

	elif(second_most_common == 4):
		print('The patient may have premature ventricular contraction beat (PVC).')
		print('Number of beats of PVC tpye are ' + str(num_second_most_common) + ' out of ' + str(len(images)))

	elif(second_most_common == 5):
		print('The patient may have right bundle branch block beat (RBB).')
		print('Number of beats of RBB tpye are ' + str(num_second_most_common) + ' out of ' + str(len(images)))

	elif(second_most_common == 6):
		print('The patient may have ventricular escape beat (VEB).')
		print('Number of beats of VEB tpye are ' + str(num_second_most_common) + ' out of ' + str(len(images)))
except IndexError:
	pass

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
