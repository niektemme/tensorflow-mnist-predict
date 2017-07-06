# TensorFlow<sup>TM</sup> MNIST predict (recognise handwriting)

This repository accompanies the blog post [Using TensorFlow<sup>TM</sup> to create your own handwriting recognition engine](http://niektemme.com/2016/02/21/tensorflow-handwriting/). 

## Installation & Setup

### Overview
This project uses the MNIST tutorials from the TensorFlow website. The two tutorials, the beginner tutorial and the expert tutorial, use different deep learning models. The python scripts ending with _1 use the model from the beginner tutorial. The scripts ending with _2 use the model from the advanced tutorial. As expected scripts using the model from the expert tutorial give better results.

This projects consists of four scripts: 

1. _create_model_1.py_ – creates a model model.ckpt file based on the beginners tutorial.
2. *create_model_2.py* – creates a model model2.ckpt file based on the expert tutorial.
3. *predict_1.py* – uses the model.ckpt (beginners tutorial) file to predict the correct integer form a handwritten number in a .png file.
4. *predict_2.py* – uses the model2.ckpt (expert tutorial) file to predict the correct integer form a handwritten number in a .png file.

### Dependencies
The following Python libraries are required.

- sys - should be installed by default
- tensorflow - [TensorFlow](https://www.tensorflow.org/)
- PIL – [Pillow](http://pillow.readthedocs.org)

### Installing TensorFlow
Of course TensorFlow needs to be installed. The [TensorFlow website](https://www.tensorflow.org/versions/master/get_started/index.html) has a good manual .

### Installing Python Image Library (PIL)
The Python Image Library (PIL) is no longer available. Luckily there is a good fork called Pillow. Installing is as easy as:

```sudo pip install Pillow```

Or look at the [Pillow documentation ](http://pillow.readthedocs.org) for other installation options.

### The python scripts
The easiest way the use the scripts is to put all four scripts in the same folder. If TensorFlow is installed correctly the images to train the model are downloaded automatically. 

## Running
Running is based on the steps:

1. create the model file
2. create an image file containing a handwritten number
3. predict the integer 

### 1. create the model file
The easiest way is to cd to the directory where the python files are located. Then run:

```python create_model_1.py```

or

```python create_model_2.py```

to create the model based on the MNIST beginners tutorial (model_1) or the model based on the expert tutorial (model_2).

### 2. create an image file
You have to create a PNG file that contains a handwritten number. The background has to be white and the number has to be black. Any paint program should be able to do this. Also the image has to be auto cropped so that there is no border around the number.

### 3. predict the integer
The easiest way again is to put the image file from the previous step (step 2) in the same directory as the python scripts and cd to the directory where the python files are located. 

The predict scripts require one argument: the file location of the image file containing the handwritten number. For example when the image file is ‘number1.png’ and is in the same location as the script, run:

```python predict_1.py ‘number1.png’```

or

```python predict_2.py ‘number1.png’```

The first script, predict_1.py, uses the model.ckpt file created by the create_model_1.py script. The second script, predict_2.py, uses the model2.ckpt file created by the create_model_2.py script. 


