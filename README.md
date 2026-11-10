ReviewIQ AI – Product Review Sentiment Analysis

An interactive machine learning web application that analyzes product reviews and classifies customer opinions into Positive, Negative, and Neutral sentiments.

ReviewIQ AI helps understand customer feedback through sentiment prediction, visual dashboards, aspect-wise analysis, batch processing, and review history.

🚀 Project Overview

Product reviews play an important role in customer purchasing decisions. Reading hundreds of reviews manually can be time-consuming.

ReviewIQ AI provides an easy way to analyze customer reviews and understand overall product sentiment using Machine Learning.

✨ Features

- Sentiment Prediction: Classifies reviews as Positive, Negative, or Neutral.
- Confidence Score: Displays the model's prediction confidence.
- Interactive Dashboard: Shows review statistics and sentiment charts.
- Aspect-wise Sentiment Analysis: Analyzes Battery, Price, Quality, Performance, Camera, and Display using keyword-based rules.
- Batch Analysis: Upload a CSV file to analyze multiple reviews.
- Review History: Stores analyzed reviews using SQLite.
- Model Performance: Displays evaluation metrics and confusion matrix.
- Product Comparison: Helps compare product sentiment.

🛠️ Technologies Used

Technology| Purpose
Python| Programming language
Streamlit| Web application
Pandas| Data handling
Scikit-learn| Machine Learning
Plotly| Interactive visualizations
SQLite| Review history database
Pickle| Saving trained model files

🧠 Machine Learning Approach

The project uses a trained sentiment classification model with a text vectorizer.

The model predicts one of three sentiment classes:

- Positive
- Negative
- Neutral

The project includes a small sample dataset containing 36 reviews, with 12 examples in each sentiment class.

Note: Aspect-wise sentiment analysis currently uses keyword-based rules rather than a separately trained machine learning model.

📂 Project Structure

Product_Review_Sentiment_Analysis/
│
├── app.py
├── train_model.py
├── reviews.csv
├── reviews_advanced.csv
├── sentiment_model.pkl
├── vectorizer.pkl
├── evaluation_metrics.json
├── confusion_matrix.csv
├── reviews.db
└── README.md

⚙️ Installation and Setup

1. Clone the repository

git clone  https://github.com/VishalMaurya68/Review-IQ-AI-Product-Review-Sentiment-Analysis.git

2. Open the project folder

cd Product_Review_Sentiment_Analysis

3. Install required libraries

python -m pip install pandas scikit-learn streamlit plotly

4. Run the application

python -m streamlit run app.py

5. Open in your browser

http://localhost:8501

📊 Application Modules

1. Dashboard
2. Analyze Review
3. Aspect-wise Sentiment Analysis
4. Batch Analysis
5. Review History
6. Model Performance
7. Product Comparison

🔮 Future Scope

- Increase the size and diversity of the review dataset.
- Improve model accuracy using larger datasets.
- Add multilingual sentiment analysis.
- Use advanced NLP and deep learning techniques.
- Add live product review collection through supported APIs.

⚠️ Limitations

- The current training dataset contains only 36 sample reviews.
- Model predictions may not always be correct.
- Aspect-wise sentiment analysis is keyword-based.
- Results should be treated as a basic academic project demonstration.

👨‍💻 Developed By

Vishal Maurya
Akshat Diwakar
- Course: B.Tech CSE (Data Science & Artificial Intelligence)
- Roll Number: 202410101150067,202410101150080
- Group: 53
- University: Shri Ramswaroop Memorial University (SRMU)
- Faculty Guide: Mr. Abhishek

📌 Project Purpose

Developed as a college academic project to demonstrate Machine Learning, Natural Language Processing concepts, data visualization, and interactive web application development.

---

⭐ If you find this project useful, consider giving the repository a star!
