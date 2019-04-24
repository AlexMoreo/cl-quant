# Cross-Lingual Sentiment Quantification

This repository contains the code to replicate the experiments of the article entitled ["Cross-Lingual Sentiment Quantification"](https://arxiv.org/abs/1904.07965).

This code is mainly built upon *Scikit-Learn*, *NumPy*, *SciPy*, and *Pandas*, that you might install before proceeding.
It also utilizes most of the functionalities of [PyDCI](https://github.com/AlexMoreo/pydci) (implementing *Distributional Correspondence Indexing*), 
[nut](https://github.com/pprett/nut) (implementing *Cross-Lingual Structural Correspondence  Learning*), and [QuaNet](https://github.com/HLT-ISTI/QuaNet).
Both *PyDCI* and *QuaNet* were implemented by us and included here for the sake of ease (some modifications are required though to use the Prettenhofer's *nut* package, that I can distribute upon request).

The following scripts can be used to replicate all experiments involving DCI:
* **generate_dci_vectors.py** produces, for each task, the (numpy) cross-lingual vectors for the training and test documents (*train.vec.npy* and *test.vec.npy*). 
It also generates the vectors of sentiment predictions (*train.y_pred.npy* and *test.y_pred.npy*, i.e., the classification of the training documents according to a 10-fold cross validation
and the classification of the test documents using all training documents --  the classifier is a LinearSVC), and a copy of the
true labels (*train.y_pred.npy* and *test.y_pred.npy*).
* **generate_probabilities_logreg.py** produces, for each task, the vector of (calibrated) posterior probabilities 
(*train.y_prob.npy* and *test.y_prob.npy*) by training and using a logistic regressor using the cross-lingual vectors
with their true labels.
* **quantification_evaluation.py** computes, using the vectors of predictions previously generated, the Classify and Count (CC), 
Adjusted CC (ACC), Probabilistic CC (PCC), and Probabilistic Adjusted CC (PACC) quantification results.
* **generate_quanet_results.py** computes the quantification for QuaNet using all vectors generated previously.


 
