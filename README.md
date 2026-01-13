# Student Performance Risk Predictor




## About the Project

Student Performance Risk Predictor is an end-to-end Machine Learning web application that predicts a student’s academic performance category based on assessment scores.

The goal of this project is to identify students who may be at academic risk at an early stage, so that timely interventions can be planned. The project focuses not only on model building, but also on deploying a working ML system that can be used by real users.

This project was built and deployed as part of my hands-on learning in Machine Learning and Data Science.



## Problem Statement

In academic environments, it is often difficult to identify struggling students before it is too late. Traditional evaluation methods are usually reactive rather than proactive.

This project aims to:
- Predict student performance risk using machine learning
- Classify students into meaningful performance categories
- Provide a simple web interface for real-time predictions
- Demonstrate a production-ready ML deployment



## Machine Learning Solution

- Type: Multi-class classification
- Model: XGBoost Classifier
- Target Output:
  - At Risk
  - Medium Risk
  - High Performer

### Input Features
- Quiz Average (%)
- Assignment Average (%)
- Midterm Exam Score (%)
- Final Exam Score (%)




## Tech Stack

- Python
- XGBoost
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Streamlit Community Cloud



## Application Features

- Interactive sliders for input scores
- Real-time prediction using a trained ML model
- Clear performance category output
- Deployed and accessible via a public URL
- Cloud-compatible model loading


## Live Demo

The application is deployed on Streamlit Cloud and can be accessed here:

https://student-performance-risk-predictor.streamlit.app



## Author

Devendra Kumar  
B.Tech Student | Data Science & Machine Learning Enthusiast  
GitHub: https://github.com/DevendraKumar577  
Live App: https://student-performance-risk-predictor.streamlit.app





