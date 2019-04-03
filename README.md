# ML_PROJECTS
MACHINE LEARNING

#####################################################################################
                         OBJECT_RECOGNITION
#####################################################################################

The Object Recognition model is implemented using ALL-CNN using Keras and Tensorflow. 

The summary of the architecture is shown in the table below:

The All-CNN Model-C

* Input 32 × 32 RGB image
* 3 × 3 conv. 96 ReLU
* 3 × 3 conv. 96 ReLU
* 3 x 3 max-pooling stride 2
* 3 × 3 conv. 192 ReLU
* 3 × 3 conv. 192 ReLU
* 3 × 3 max-pooling stride 2
* 3 × 3 conv. 192 ReLU
* 1 × 1 conv. 192 ReLU
* 1 × 1 conv. 10 ReLU
* global averaging over 6 × 6 spatial dimensions
* 10 or 100-way softmax


Layer (type)   ,              Output Shape    ,          Param    
=================================================================
conv2d_41 (Conv2D)     ,      (None, 32, 32, 96)   ,     2688      
_________________________________________________________________
activation_32 (Activation) ,  (None, 32, 32, 96)   ,     0         
_________________________________________________________________
conv2d_42 (Conv2D)   ,        (None, 32, 32, 96)   ,     83040     
_________________________________________________________________
activation_33 (Activation)  , (None, 32, 32, 96)   ,     0         
_________________________________________________________________
conv2d_43 (Conv2D)        ,   (None, 16, 16, 96)   ,    83040     
_________________________________________________________________
dropout_10 (Dropout)     ,    (None, 16, 16, 96)   ,     0          
_________________________________________________________________
conv2d_44 (Conv2D)        ,   (None, 16, 16, 192)   ,    166080    
_________________________________________________________________
activation_34 (Activation) ,  (None, 16, 16, 192)   ,    0         
_________________________________________________________________
conv2d_45 (Conv2D)         ,  (None, 16, 16, 192)   ,    331968    
_________________________________________________________________
activation_35 (Activation) ,  (None, 16, 16, 192)   ,    0         
_________________________________________________________________
conv2d_46 (Conv2D)         ,  (None, 8, 8, 192)     ,    331968    
_________________________________________________________________
dropout_11 (Dropout)       ,  (None, 8, 8, 192)     ,    0         
_________________________________________________________________
conv2d_47 (Conv2D)        ,   (None, 8, 8, 192)     ,    331968    
_________________________________________________________________
activation_36 (Activation) ,  (None, 8, 8, 192)     ,    0         
_________________________________________________________________
conv2d_48 (Conv2D)        ,   (None, 8, 8, 192)     ,    37056     
_________________________________________________________________
activation_37 (Activation) ,  (None, 8, 8, 192)     ,    0         
_________________________________________________________________
conv2d_49 (Conv2D)        ,   (None, 8, 8, 10)      ,    1930      
_________________________________________________________________
global_average_pooling2d_5 ,  ( (None, 10)            ,    0 
_________________________________________________________________
activation_38 (Activation) ,   (None, 10)            ,    0

=================================================================
Total params: 1,369,738
Trainable params: 1,369,738
Non-trainable params: 0
_________________________________________________________________
None
10000/10000 [==============================] - 45s 5ms/step
Accuracy:0.9088


NATURAL LANGUAGE PROCESSING

#####################################################################################
                         TEXT_CLASSIFICATION
#####################################################################################

The text classification is done using 
ensemble method - Voting classifier...i.e.

E.g., if the prediction for a given sample is

classifier 2 -> class 1

classifier 3 -> class 2

the VotingClassifier (with voting="hard")

would classify the sample as--> “class 1”

based on the majority class label. In the cases of a tie, the VotingClassifier will select the class based on the ascending sort order. E.g., in the following scenario

classifier 1 -> class 2

classifier 2 -> class 1

-->the class label 1 will be assigned to the sample.

-->Here class 1,2 are spam,ham...

In contrast to majority voting (hard voting), -->soft voting" returns the class label as argmax of the sum of predicted probabilities.

The weighted average probabilities for a sample would then be calculated as follows:

classifier ------class 1-------class 2---------class 3

classifier 1 --- w1 * 0.2------w1 * 0.5 -------w1 * 0.3

classifier 2 --- w2 * 0.6 -----w2 * 0.3 -------w2 * 0.1

classifier 3 ----w3 * 0.3 -----w3 * 0.4 -------w3 * 0.3

weighted avg ---- 0.37 -------0.4 ------------0.23

-->Here, the predicted class label is 2, since it has the highest average probability.

If your algorithms are optimized then go for soft

The results are as follows:
K Nearest Neighbors :Accuracy:94.34386216798278
Decision Tree :Accuracy:97.27709978463747
Random Forest :Accuracy:97.95979899497488
Logistic Regression :Accuracy:93.91816223977028
SGDClassifier :Accuracy:96.06173725771716
Naive-Bayes :Accuracy:97.27709978463747
SVM Linear :Accuracy:95.91816223977028




