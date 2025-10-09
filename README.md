# End-to-End MLOps: Predicting Customer Satisfaction

This project demonstrates a complete, end-to-end machine learning pipeline for the Olist e-commerce dataset. The primary goal is to predict customer satisfaction by classifying their potential review score (1-5 stars) based on initial order data. The entire lifecycle, from raw data ingestion to a containerized API, is managed using professional MLOps practices.

[![CI/CD Pipeline](https://github.com/shaikhilhaam/ecommerce-analytics-platform/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shaikhilhaam/ecommerce-analytics-platform/actions)

**Docker Hub Image:** [`shaikhilhaam/olist-review-api:latest`](https://hub.docker.com/r/shaikhilhaam/olist-review-api)

---

## Tech Stack

| Category                | Technologies                                                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| **Data Storage & Querying** | PostgreSQL, SQL                                                                                          |
| **Data Science & EDA** | Python, Pandas, Jupyter Notebook, Matplotlib, Seaborn, Tableau                                           |
| **Feature Engineering** | Scikit-learn, Sentence-Transformers (for NLP), Haversine                                                 |
| **Modeling & MLOps** | XGBoost, MLflow (Tracking & Model Registry), Optuna, SHAP                                                |
| **API & Deployment** | FastAPI, Uvicorn                                                                                          |
| **Containerization** | Docker                                                                                                   |
| **Automation & CI/CD** | Git, GitHub Actions                                                                                      |

---

## Project Pipeline Overview

This project follows a structured, multi-stage pipeline:

1.  **Data Engineering:** Raw CSV files are ingested and structured into a **PostgreSQL** database. A comprehensive master table is generated using a single SQL query that joins 9 different tables, performing the heavy data lifting within the database.

2.  **Exploratory Data Analysis (EDA):** An extensive analysis revealed key business insights, most notably that **delivery performance (speed and accuracy) is the single biggest driver of customer satisfaction**, and that the platform has an extremely low customer retention rate (~3%). This insight guided the pivot from an LTV model to a more impactful review score prediction model.

3.  **Feature Engineering:** A sophisticated feature set was engineered, including:
    * **Logistical Features:** `delivery_time_vs_estimated`, `customer_seller_distance`.
    * **Temporal Features:** Cyclical encoding of month and day of the week.
    * **NLP Features:** State-of-the-art text embeddings from review comments, compressed with PCA.
    * **Seller Features:** Seller's average review score and total order count.

4.  **Modeling & Experiment Tracking:**
    * An **XGBoost** multiclass classifier was trained to predict the 1-5 star review score.
    * **MLflow** was used to manage the entire modeling lifecycle. Experiments were tracked, and the final, best-performing model was versioned and stored in the **MLflow Model Registry**.
    * Model performance was deeply analyzed using a confusion matrix, classification reports, and **SHAP** for expert-level explainability.

5.  **Containerization & Deployment:**
    * The registered model is served via a **FastAPI** application, which exposes a `/predict` endpoint.
    * The entire application is containerized using **Docker**, creating a lightweight, portable, and production-ready image.

6.  **Automation (CI/CD):** A **GitHub Actions** workflow automates the entire process. On every push to `main`, the workflow automatically builds the Docker image and pushes the `:latest` tag to Docker Hub, ensuring the application is always up-to-date.

---

## How to Run This Project

### 1. Setup

* Clone the repository.
* Create a Python virtual environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
* Set up a PostgreSQL database and populate it using the `ingest_data.py` script.
* Create a `.env` file with your database credentials.

### 2. Run the Data & Training Pipeline

* Generate the final modeling dataset:
    ```bash
    python main.py
    ```
* Train the model and register it with MLflow:
    ```bash
    python src/train.py
    ```

### 3. Run the API Locally

* Start the FastAPI server:
    ```bash
    uvicorn api:app --reload
    ```
* Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 4. Build and Run with Docker

* Build the Docker image:
    ```bash
    docker build -t olist-review-api .
    ```
* Run the container:
    ```bash
    docker run -p 8000:8000 olist-review-api
    ```