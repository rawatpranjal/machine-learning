"""
In this exercise we replicate the Adaboost algorithm as it was originally published in the paper "A decision theoretic
generalization of online learning and an application to boosting" by Freund & Schapire in 1996. 

The exercise uses Decision Trees as "Weak" learners that can be "boosted" to become "strong" learning w.r.t. evaluation metrics
like Accuracy, AUC, so on. 

Boosting centers around the 'sample weight' given to training examples. Unweighted trees will treat each training example as
equal before its eyes as it performs recursive splits. Weighted trees will however, treat examples with a higher weights as
more "important" and of "higher priority" than examples with lower weight. Weighted trees will try, through altered metrics 
to perform the split, to classify a greater proportion of the more "important" examples than unweighted trees. <How exactly, 
is something I'm yet to figure out>. 

Boosting changes the sample weights in a way that the newer Decision Trees are forced to classify the examples that the
older trees could not. Finally, the 'votes' of all trees - new and old - are combined in a particular fashion that gives 
us predictions that not only give good accuracy on examples that were easy, but also on some examples that were hard. 
Boosting, thus, helps Trees focus on classifying examples that are 'hard' to classify. 

The 'Weighted Misclassification error" i.e the sum of sample weights of misclassified examples,
is what decides how much 'say' a tree will have. The first tree has a large say because it will obtain 
the lowest misclassification error -- on all samples. And for the first tree the 'Weighted' Misclassification error
is equal to the 'Unweighted Misclassification error' because all sample weights are initally equal. But for later trees, 
as they are forced to focus on harder examples, the 'Unweighted Misclassification error' is much higher. However, their 
'Weighted' Misclassification error will be a bit lesser. Thus it will continue to have a 'say' despite a poorer accuracy. 

If trees are given a 'say' based on "Misclassification error" instead of "Weighted Misclassification error", 
then later trees have very very little or no say. The boosting magic lies in the ability of the ensemble to give later trees
non-zero say, by virtue of their ability to classify a few hard examples. 

Boosting follows the following steps: 
1. initialize sample weights to 1/M, where M is the no of examples. 
2. begin loop: 
  1. Get predictions on training data using features + sample weights. 
  2. Calculate 'weighted misclassification error' i.e the sum of sample weights of incorrectly classified examples
  3. Calculate beta normalizes the "weighted misclassification error" on the 0 to 1 scale. 
  4. Update sample weights. For those examples that were incorrectly classified; leave their weights unchanged. 
      For those examples correctly classified, reduce the weight by a factor of 'beta'. 
       
      This ensure that a model with low "weighted misclassification error" will find that those examples that it rightly classified
      had their weights decreased by a very large margin. Model with high "weighted misclassification error", will not have much influence
      on the weight updates. 
  5. end loop if sufficient number of iterations (T) have passed.
3. We now have T betas and T set of predictions. We combine these to get a final prediction through a weighted vote. 
   Trees with low "weighted misclassification error" are given more say in the final outcome. 
   Log(1/beta[t]) is taken to be the amount of 'say' that a tree "t" can have. # note the monotonically decreasing transformation
   If y_hat[t, i] is the prediction of t tree on ith example; 
   OUTCOME[i] is defined to be ----> SUM over t: y_hat[t, i] * Log(1/beta[t]) 
   OUTCOME[i] is the sum of all predictions made by all trees, adjusting for their say, for ith example. 
   
   if OUTCOME[i] is high, most trees and more important trees probably indicate that yhat[i] should be 1
   vice versa. To decide the threshold value, we take the maximum value and minimum value of OUTCOME[i] 
   and divide it by two. The maximum value is "SUM over t: Log(1/beta[t])" and minimum value is 0. 
   Thus the threshold is 0.5 * SUM over t: Log(1/beta[t]).
   
   Thus Adaboost y_hat[i] = 1 if SUM over t trees: y_hat[t, i] * Log(1/beta[t]) is larger than "SUM over t trees: Log(1/beta[t])"
                          = 0, otherwise. 

Read more here: http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf
You can find the pseudo code on page 126. 
"""
                          
import pandas as pd, numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Synthetic Dataset 
x, y = make_classification(n_samples = 10000, n_features=20, n_informative=5, n_clusters_per_class=2, random_state = 42)
x = np.where(x > np.mean(x, axis = 0), 1, 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Y & S 1996 - Original paper
from sklearn.tree import DecisionTreeClassifier
depth = 3
m = x_train.shape[0]
w = np.ones(m)/m
n_trees = 100

betas = np.empty(n_trees)
yhats = np.empty((m, n_trees))
for i in range(n_trees): 
    p = w/np.sum(w)
    model = DecisionTreeClassifier(max_depth = depth)
    model.fit(x_train, y_train, sample_weight = w.tolist())
    ε = np.sum(p * np.abs(y_train - model.predict(x_train)))
    β = ε/(1-ε)
    w = w * np.power(β, 1 - np.abs(y_train - model.predict(x_train)))
    
    # Save Model, Beta & Model Predictions
    betas[i] = β
    yhats[:, i] = model.predict(x_train)
    
adaboost_yhat = np.where(np.dot(yhats, np.log(1/betas)) > 0.5 * np.sum(np.log(1/betas)), 1, 0)
print(np.mean(np.where(adaboost_yhat==y_train, 1, 0)))

# Compare to Sklearn
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = depth))
model.fit(x_train, y_train)
print(accuracy_score(y_train, model.predict(x_train)))
