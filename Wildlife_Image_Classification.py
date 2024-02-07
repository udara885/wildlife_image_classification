#import libraries

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

#load pre trained VGG16 model

model = VGG16(weights='imagenet')

#load an image for classification

image_path = '' #set path of the image
img = image.load_img(image_path,target_size=(224,224))

#preprocess the image

x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

#make predictions on the image

predictions = model.predict(x)

#decode and display

decoded_predictions = decode_predictions(predictions,top=5)[0]

#print the result

for _, label, score in decoded_predictions:
	print(f'{label}:{score:.2f}')