The file plotter.py uses Matplotlib and is built to generate eleven graphs in order to visualise the data from attribute 3 to attribute 13 against age and sex.

Inside the src/algorithms directory, there are implementations of three algorithms, including kNN, kMeans and SVM. Each of them is compared with each other based on their accuracy of prediction.

Important factors are found using the three algorithms. The important factors in kMeans and kNN are found by removing a different attribute from prediction each time and listing the attributes with largest accuracy decrements. The important factors in SVM is found by listing out the attributes with the largest weights.

Prediction accuracy of each algorithm is evaluated and kNN is found to be best. So it’s used to predict heart disease in part three.

The bonus part is the processing of input attributes, including normalising continuous data and splitting discrete data.