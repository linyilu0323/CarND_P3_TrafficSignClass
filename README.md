
[//]: # "Image References"

[image1]: ./examples/raw_img.png "Raw image is in RGB"
[image2]: ./examples/precond_img.png "Preprocessed image in grayscale"
[image3]: ./examples/custom_test.png "Custom German traffic sign images"
[image4]: ./examples/custom_test_topK.png "Top K for custom test images"
[image5]: ./examples/distribution.png "Distribution of data sets"
[image6]: ./examples/accuracy_evolve.png "Training and Validation Accuracy vs Epochs"
# **Traffic Sign Recognition** 

## Objective:

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I get the size of training, validation, and test data set by looking at their length: `len(X_???)` , this tells me the number of images in each data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.

Then, I use the shape attribute of any single image, more specifically, I use the shape of first image: `X_train[0].shape` 

* The shape of a traffic sign image is (32, 32, 3)

Then, I look at the unique labels in `y_train` array and then look at the length: `len(np.unique(y_train))` 

* The number of unique classes/labels in the data set is 43 - this aligns with the number of entries provided in  `signnames.csv`.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the normalized frequency of each traffice sign in the datasets with red representing training set, green representing validation set, and blue representing test set. It can be seen that the distribution is not quite even, with more examples for outputs 1~13 and 38. However, the distribution between 3 sets of data seemed to match closely with each other.

![alt text][image5]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. 

I pre-processed the image data by two steps: first step is to convert to gray scale, second is to normalize it.

```python
X_train_norm = np.sum(X_train_norm/3, axis=3, keepdims=True)
X_train_norm = (X_train_norm - 128)/128
```

The difference between the original data set and the augmented data set is the following:

![alt text][image1]

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|-----------------------------------------------|
| Input         		| 32x32x1 grayscale image |
| Convolution 5x5   | filter depth 6, stride 1x1, valid padding, output 28x28x6 |
| RELU					| activation |
| Convolution 19x19	| filter depth 16, stride 1x1, valid padding, output 10x10x16 |
| RELU	| activation  |
| Max Pooling	| kernal 2x2, stride 2x2, valid padding, output 5x5x16 |
| Drop-out	| "keep probability" set to 0.5 during training |
| Flatten | output 400 array |
| Fully Connected | input 400, output 120 |
| RELU | activation |
| Drop-out | "keep probability" set to 0.5 during training |
| Fully Connected | input 120, output 84 |
| RELU | activation |
| Drop-out | "keep probability" set to 0.5 during training |
| Fully Connected | input 84, output 43 |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 30 epochs, with each batch size of 512 (since the workspace is capable, I make the batch size larger to reduce training time), and the learning rate of 0.001.

The optimizer used for training is `AdamOptimizer`.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.6%
* validation set accuracy of 95.0%
* test set accuracy of 94.1%

I used an iterative approach in defining the model architecture and training the model:
* In the very begining, the default LeNet architecture without any drop-out was used, with a learning rate of 0.001 and 10 epochs. When I started to train the model, I first found the accuracy improvement was not as fast as I expected, it barely reached 90% accuracy at the end of 10 epochs, and I did not want to increase learning rate because of potential risk of missing the optimum. 
* So the first thing I did was to increase the epochs to 50, and added a chart plotting training set and validation set accuracy, so I can visualize the changing accuracy vs number of epochs. I can shorten the epochs if I find any tendency of over-fitting. With this, I found even after 50 epochs, the training dataset accuracy was OK but the validation set accuracy was <80%, this indicates an **over-fitting model**. 
* With this observation, I started to add new layers into the model to avoid over-fitting. I ended up with a few additional drop-outs, and also I removed one of the max-pooling layer. I then repeated the training and it showed good accuracy. By observing the trend of accuracy vs. number of epochs, I decided to shorten it to 30 epochs to further avoid over-fitting and reduce training time.
* Below chart shows the final model performance over epochs:

![alt text][image6]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. I first imported them and do the same pre-processing (grayscale, normalization), the images before and after pre-processing looked like below:

![alt text][image3]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Label |                       Image                        |                     Prediction                     |
|:---------------------:|:---------------------------------------------:|:---------------------:|
|  26   |                  Traffic Signals                   |                  Traffic Signals                   |
|   5   |                Speed Limit (80km/h)                |                Speed Limit (80km/h)                |
|  27   |                    Pedestrains                     |                    Pedestrains                     |
|  42   | End of no passing by vehicles over 3.5 metric tons | End of no passing by vehicles over 3.5 metric tons |
|  20   |            Dangerous curve to the right            | Dangerous curve to the right	|
|  40   |              **Roundabout mandatory**              | **End of speed limit (80km/h)**	|
|  33   |                  Turn right ahead                  | Turn right ahead	|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Below is the top 5 softmax probabilities for each custom test image, I was a bit surprised the Image 40 (roundabout) is not recognized correctly since the image appeared to be pretty clear to me, the correct prediction showed up to be second probable on the top-5 softmax probabilities. 

![alt text][image4]