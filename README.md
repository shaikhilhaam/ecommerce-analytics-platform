# E-commerce Customer Segmentation and LTV Prediction Platform

## Project Goal

This project aims to build a comprehensive, end-to-end data platform for analyzing customer behavior from the Olist e-commerce dataset. The primary goals are to perform customer segmentation to identify key customer groups and to predict Customer Lifetime Value (LTV). The entire pipeline, from data ingestion to model deployment, follows MLOps best practices.

---

## Tech Stack Overview

* **Data Storage:** PostgreSQL
* **Data Analysis & Transformation:** Python (Pandas, NumPy)
* **Experiment Tracking:** MLflow
* **Model Development:** Scikit-learn, PyTorch
* **API Development:** FastAPI
* **Containerization:** Docker
* **Automation (CI/CD):** GitHub Actions
* **Visualization:** Tableau Public

---

## Project Phases

1.  **Data Engineering:** Ingest raw CSV data into a structured PostgreSQL database. Establish schema and table relationships.
2.  **Exploratory Data Analysis (EDA) & Segmentation:** Analyze data using SQL and Python. Perform customer segmentation using RFM analysis and clustering (K-Means). Visualize segments in Tableau.
3.  **NLP for Review Analysis:** Apply sentiment analysis to customer reviews to create features for product quality and satisfaction metrics.
4.  **Predictive Modeling (LTV):** Train and track machine learning models using MLflow to predict future customer value.
5.  **Deployment:** Expose the model through a REST API using FastAPI and containerize the application with Docker for production readiness.
