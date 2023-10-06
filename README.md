## Predictive Maintenance using Deep Learning Models
# Introduction
Predictive maintenance uses machine learning models to determine the condition of industrial equipment inorder to preemptively trigger a maintenance schedule and thereby avoid any adverse machine performance. This project leverages the Deep Learning capabilities to build Machine Learning models aimed at following tasks.
- Predicting Remaining Useful Life(RUL) of a machine based on historical performance and individial parameter readings at regular cycles
- Predicting the probability of a machine failure in 'n' future cycles

# Data Preparation
Datasets under Predictive Maintenance/sensordata consists of multiple multivariate timeseries attribute values divided into training and test dataset. Data corresponds to a fleet of engines of same type at regular cycles. There are three operational settings along with multiple sensor data for each engine for 'n' cycles. At the start of each timeseries each engine is operating normally and gradually lands in a failure state after 'n' cycles. Training dataset is provided till the engine failure i.e, Remaining Useful Life(RUL) of 0 cycles for each engine. Test dataset is provided till 'n' cycles and the truth dataset consists of Run to Failure (RTF) of each engine. The columns in each dataset correspond to:
- Engine id
- Cycle number
- Operational setting 1
- Operational setting 2
- Operational setting 3
- sensor readings from 1 to 21

# Model Building and Evaluation
Based on the truth values for each engine, RUL is calculated in training dataset and fed during model building phase. Machine Learning model for predicting Remaining Useful Life(RUL) of a machine along with Deep Learning model for predicting the probability of machine failure in 10, 20 and 30 cycles is built and serialized for saving the same under models directory. Depending on the RUL and failure probability, each prediction is categorized into one of the three(red, yellow, green) severity levels.

Severity Categories based on failure probability

- Red : Failure probability > 50%
- Yellow: 20% < Failure probability < 50%
- Green: Failure Probability < 20%

Categories based on RUL

- Red : RUL < 10 cycles
- Yellow : 10 <= RUL <= 30
- Green : RUL > 30 cycles


# Operationalization
Flask application is built to calcaulte RUL and failure prediction of a given engine id. A grid for displaying the stats of RUL and failure probabilities of all the engines is also built and displayed with different maintenance schedule categories. UI samples for the same can be found under templates directory.
