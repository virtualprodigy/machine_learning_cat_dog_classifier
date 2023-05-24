# README

## Welcome
Welcome to this humble little project. This is a Machine Learning Project. It is currently aimed at categorizing photos of Cats and Dogs using TensorFlow Metal, Keras, Numpy. This project has been created with the intent to be run locally on your Apple Silicon Mac using the GPU. 

## Disclaimer 
1. Don't expect to see this project to get a ton of updates. 
2. Images for training & testing were provided by https://www.kaggle.com/ 
3. If you use this code or fork this code you must provide attribution to this original project and to VirtualProdigy.
4. This project is for fun, so if you aren't having fun....You must start having fun now

## Setup
1. Setup TensorFlow -> https://www.tensorflow.org/install/pip
	2. Mac Users follow these instructions -> https://developer.apple.com/metal/tensorflow-plugin/
2. At this point you should have setup you miniconda envirnoment. In your envirnoment run this pip command
	3. `python -m pip install matplotlib pillow opencv-python numpy keras scipy coremltools`
3. Now you need to get your photos from somewhere, kaggle.com is pretty nice, check them out
4. run your code
	5. resize photos so you save on memory during training. Use my script for it 
		6. `python resize_images.py`
	6. Train your model
		7. `python train_cat_dog_ml.py`
	7. Test your model 
		8. `python test_cat_dog_ml.py`
	8. Activate Celebrate.exe 
		9. You've done your first bit of ML. Whether you ran into bugs or it worked perfectly, you took a major step just now. Congratulations! 

# Created with Love and for Fun from VirtualProdigy 