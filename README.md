# waste-sorting-monitering

The Waste-Sorting Monitoring and Evaluation System primarily utilizes IoT, Machine Learning, Deep Learning (MLDL), and Computer Vision. 
The IoT system uploads data from a Wi-Fi camera to a cloud platform. Image-based MLDL models are then applied to the collected footage 
to detect movement, identify waste (object of interest), and classify the waste. 
Once the waste is detected, a verification algorithm runs to ensure the correctness of the classification. 
The results are stored on the cloud platform and displayed on a dashboard.

app.py contains the code to read recorded videos on that current day stored on GCP bucket, perform all the detections, tracking and algorithm on the videos and then upload the result to GCP bucket.
