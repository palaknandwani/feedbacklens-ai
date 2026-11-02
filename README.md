# FeedbackLens AI 🔍

### AI-Powered Customer Feedback Intelligence Platform

FeedbackLens AI is an interactive customer feedback analytics application built using Python, Streamlit, Pandas, Plotly and Scikit-learn.

It helps analyse customer reviews, classify sentiment, identify recurring product issues and generate actionable product improvement recommendations.

## Features

- **Sentiment Analysis:** Classifies customer feedback as positive, negative or neutral using machine learning.
- **Executive Dashboard:** Displays review KPIs and sentiment distribution.
- **Customer Feedback Explorer:** Search, filter and download customer reviews.
- **Product Insights:** Identify categories receiving negative feedback.
- **AI Recommendations:** Generate rule-based product improvement suggestions.
- **CSV Upload:** Analyse your own feedback dataset.

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- TF-IDF
- Logistic Regression

## Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/feedbacklens-ai.git
cd feedbacklens-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Dataset

The project includes sample customer feedback for demonstration. Users can also upload their own CSV file containing a `review` column.

## Machine Learning

The sentiment classifier uses TF-IDF text features and Logistic Regression. It is trained on a small demonstration dataset and is intended for portfolio demonstration, not production decisions.

## Project Purpose

This project demonstrates practical skills in machine learning, customer analytics, product thinking and interactive dashboard development.

**Created by Palak Nandwani**