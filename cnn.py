"""
Created on Fri oct 18:45:05 2020

@author: suhas
"""



#  Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True
                                  )

training_set = train_datagen.flow_from_directory(r'D:\Projects\Projects\Image classification\PetImages',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'D:\Projects\Projects\Image classification\PetImages',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 600,
                         epochs = 100,
                         validation_data = test_set,    
                         validation_steps = 200)

#classifier.save("model.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C://users/Downloads/cat.11.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)