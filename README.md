# 📝 Article Summarization Using with NN and ML

# Article Summary

## Overview
Article Summary is a web application that summarizes long articles into concise summaries using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) and traditional machine learning models such as Logistic Regression, Linear Regression, Decision Tree Classifier, and Decision Tree Regressor.

## Purpose
The project aims to provide users with quick and accurate summaries of lengthy articles, leveraging both deep learning and traditional machine learning techniques to cater to different summarization needs.

## Key Features
- **Summary with LSTM**: Generates summaries using a deep learning model based on Recurrent Neural Networks with LSTM.
- **Summary with Logistic Regression**: Uses logistic regression for text summarization.
- **Summary with Linear Regression**: Applies linear regression for summarization tasks.
- **Summary with Decision Tree Classifier**: Employs a decision tree classifier for summary generation.
- **Summary with Decision Tree Regressor**: Utilizes a decision tree regressor for summarization.

## Dataset
The project uses the [CNN/DailyMail dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail) from Kaggle for training and testing the models. It has id, article, highlight columns.

## Tech Stack / Libraries Used
- **Python**: Core programming language.
- **Streamlit**: For building the web-based user interface.
- **TensorFlow**: For implementing the LSTM-based deep learning model.
- **Scikit-learn**: For traditional machine learning models (Logistic Regression, Linear Regression, Decision Tree Classifier, Decision Tree Regressor).
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib**: For data visualization.
- **Pickle**: For saving and loading trained models and vectorizers.

## Installation / Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd article-summary

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Download the pre-trained model files and vectorizers from the repo.
   
4. Download the summarization_combined_results.json file and place it in the project directory.

5. Run the application:
   ```bash
   bashstreamlit run main.py

## How to Use
1. Run the main.py file on your local machine with
   ```bash
   streamlit run main.py
2. Paste the article text into text box.
3. Select the desired model (LSTM, Logistic Regression, Linear Regression, Decision Tree Classifier, or Decision Tree Regressor).
4. Choose the preferred summary length.
5. Click the "Generate Summary" button to view the summarized output.

## Model Overview & Performance

| Model                      | Overview                                                                 | Performance |
|----------------------------|---------------------------------------------------------------------------|-------------|
| **LSTM (RNN)**             | Deep learning model that captures sequential dependencies in text data. | High        |
| **Logistic Regression**    | Simple, interpretable classifier used for text classification tasks.    | Medium      |
| **Linear Regression**      | Predicts summary score/length based on extracted features.               | Low-Medium  |
| **Decision Tree Classifier** | Rule-based model splitting features to classify sentences.              | Medium      |
| **Decision Tree Regressor**  | Regression-based tree predicting importance scores for sentences.       | Medium      |


## Team

| Name                     | Duties                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| **Arman Verma**          | Deep learning model that captures sequential dependencies in text data. |
| **Hargun Singh Lamba**   | Simple, interpretable classifier used for text classification tasks.    |
| **Pooja Shrestha**       | Predicts summary score/length based on extracted features.               |
| **Sudip Shrestha**       | Rule-based model splitting features to classify sentences.              |



