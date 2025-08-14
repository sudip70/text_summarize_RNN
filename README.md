Article Summary
Overview
Article Summary is a web application that summarizes long articles into concise summaries using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) and traditional machine learning models such as Logistic Regression, Linear Regression, Decision Tree Classifier, and Decision Tree Regressor.
Purpose
The project aims to provide users with quick and accurate summaries of lengthy articles, leveraging both deep learning and traditional machine learning techniques to cater to different summarization needs.
Key Features

Summary with LSTM: Generates summaries using a deep learning model based on Recurrent Neural Networks with LSTM.
Summary with Logistic Regression: Uses logistic regression for text summarization.
Summary with Linear Regression: Applies linear regression for summarization tasks.
Summary with Decision Tree Classifier: Employs a decision tree classifier for summary generation.
Summary with Decision Tree Regressor: Utilizes a decision tree regressor for summarization.

Dataset
The project uses the CNN/DailyMail dataset from Kaggle for training and testing the models.
Tech Stack / Libraries Used

Python: Core programming language.
Streamlit: For building the web-based user interface.
TensorFlow: For implementing the LSTM-based deep learning model.
Scikit-learn: For traditional machine learning models (Logistic Regression, Linear Regression, Decision Tree Classifier, Decision Tree Regressor).
Pandas: For data manipulation and preprocessing.
Matplotlib: For data visualization.
Pickle: For saving and loading trained models and vectorizers.

Installation / Setup Instructions

Clone the repository:git clone <repository-url>
cd article-summary


Install the required dependencies:pip install -r requirements.txt


Download the pre-trained model files and vectorizers from the provided source (e.g., a cloud storage link or repository).
Download the summarization_combined_results.json file and place it in the project directory.
Run the application:streamlit run main.py



How to Use

Open the application in your browser (Streamlit will provide the local URL, typically http://localhost:8501).
Paste the article text into the provided text box.
Select the desired model (LSTM, Logistic Regression, Linear Regression, Decision Tree Classifier, or Decision Tree Regressor).
Choose the preferred summary length.
Click the "Generate Summary" button to view the summarized output.

Future Improvements

Bigger Training Data: Incorporate larger and more diverse datasets to improve model performance.
Better Models: Experiment with advanced architectures like Transformers or fine-tune pre-trained language models for enhanced summarization quality.

License
This project is licensed under the MIT License. See the LICENSE file for details.
