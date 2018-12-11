# Malware Detection with Naive Bayes Classifier

**Note:** This README provides a simple overview of the homework. The report is available [here](Homework1/report).

### Overview
Learn a traget function f: APK -> {good, malware} in order to identify malicious Android applications, starting from the feature vector. 

### Dataset
The dataset used to train the model is the DREBIN dataset. Its authors used the well-known tools of the static analysis to extract features from the APK samples, both from the Android Manifest.xml file and from the disassembled Java code.
The dataset contains 

* 123453 harmless samples
* 5,560 malware.

### How to run
For now, the code is provided as a Jupyter Notebook.

### Bernoulli
![Bernoulli confusion matrix](Homework1/report/cnf_m_bernoulli.png)

### Support Vector Machine
![SVM confusion matrix](Homework1/report/cnf_m_svm.png)
