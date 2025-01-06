# 💸 BitAlyze - Bitcoin Sentiment Analysis Project

## 🔍 Overview
This project analyzes the relationship between Bitcoin's historical price trends and sentiment expressed on social media, specifically Twitter/X. The primary goal is to investigate whether sentiment from tweets mentioning Bitcoin correlates with Bitcoin's price fluctuations, volatility, and trading volume.

Through data collection, preprocessing, and statistical analysis, the project ultimately finds that there is **no significant relationship** between Bitcoin price movements and social media sentiment.


## 📈 Features
- **Bitcoin Sentiment Analysis**: Investigates the correlation between social media sentiment and Bitcoin price fluctuations using Twitter data.
- **Historical Price Trends**: Analyzes Bitcoin’s price movements over time to identify patterns related to trading volume and volatility.
- **Sentiment Classification**: Implements sentiment analysis to classify tweets as **positive, neutral, or negative** using lexicon-based techniques.
- **Correlation Analysis**: Computes statistical correlations between sentiment trends and Bitcoin price metrics.
- **Machine Learning Regression**: Develops predictive models to test the influence of social media sentiment on Bitcoin price behavior.
- **Data Visualization**: Utilizes interactive plots and visual analytics to illustrate trends and relationships in the dataset.


## 🛠 Technologies Used
- **Python**: Primary programming language for data analysis and modeling.
- **Pandas & NumPy**: Used for data manipulation and numerical computations.
- **Matplotlib & Seaborn**: Enables visualization of sentiment trends and price fluctuations.
- **Scikit-learn**: Used for machine learning models, including regression and feature scaling.
- **NLTK**: Natural Language Processing toolkit for sentiment analysis.
- **Jupyter Notebook**: Development environment for exploratory data analysis and modeling.


## ❓ Research Question
**Can I identify a measurable correlation between specific social media sentiment indicators (such as counts of positive, neutral, and negative posts) on platforms like X (formerly Twitter), and key Bitcoin price metrics?**  

To answer this, I:
- Collected Bitcoin price data and Twitter sentiment data.
- Processed and cleaned the datasets.
- Conducted sentiment analysis on tweets.
- Performed statistical tests to identify potential correlations.

## 📂 Data Sources
Two main datasets were used:

1. **[Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/data?select=btcusd_1-min_data.csv)**  
   - Contains historical Bitcoin market data at 1-minute intervals from 2012 onwards.
   - Features include **Open, Close, Volume**, and other key trading statistics.

2. **[Bitcoin Tweets Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)**  
   - Contains tweets mentioning Bitcoin using hashtags like `#Bitcoin` and `#Btc`.
   - Includes variables like **date** (when the tweet was posted) and **text** (content of the tweet).
   - Used for sentiment analysis to categorize tweets as **positive, neutral, or negative**.

## 🛠️ Technologies Used
This project was implemented using Python with the following libraries:

```python
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

## 🔬 Methodology
### 1. **Data Collection**
- Collected two primary datasets: **Bitcoin Historical Data** and **Bitcoin Tweets Dataset**.
- Merged these datasets using date and time as common keys.

### 2. **Data Cleaning**
- Removed irrelevant tweets that were not directly related to Bitcoin price discussions.
- Eliminated duplicate entries to ensure consistency.
- Standardized date and time formats for accurate merging of datasets.
- Filtered Bitcoin price data to focus on the same timeframe as the tweet dataset (2021 - present).
- Handled missing values by imputing with appropriate statistical methods.

### 3. **Sentiment Analysis**
- Used a **lexicon-based sentiment analysis** approach to classify tweets as:
  - **Positive** (bullish sentiment)
  - **Neutral** (no opinion or mixed sentiment)
  - **Negative** (bearish sentiment)
- Applied **preprocessing techniques** such as:
  - Tokenization and lowercasing
  - Removing special characters, URLs, and hashtags
  - Stopword removal and lemmatization
- Compared sentiment trends with Bitcoin price movements over different time intervals.

### 4. **Statistical Analysis**
- Computed correlation coefficients (Pearson and Spearman) to measure relationships between tweet sentiment and Bitcoin price fluctuations.
- Built **regression models** (Random Forest, Linear Regression) to assess predictive power.
- Evaluated model performance using:
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**
  - **R-Squared (R²) Score**

## 📈 Key Findings
- **No significant correlation** was found between Bitcoin price movements and Twitter sentiment.
- Unlike previous studies, our dataset did not show conclusive evidence of social media influencing Bitcoin's value.
- Market sentiment on social media does not appear to be a reliable predictor of price volatility.

## 📌 Conclusion
While some studies suggest social media sentiment can drive asset prices, this project found **no strong relationship between Bitcoin's value and Twitter sentiment trends**. This could be due to:
- High market efficiency reducing the impact of public sentiment.
- Other factors (institutional trading, regulations) playing a bigger role in price movements.
- Sentiment analysis limitations (e.g., sarcasm in tweets being misclassified).

## 🚀 How to Run the Project
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/Bitcoin-Sentiment-Analysis.git
   cd Bitcoin-Sentiment-Analysis
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook FinalProject_Group017-FA24.ipynb
   ```

---

🛠 Maintained by **Varun Naik**  
📌 *This project was made for the UC San Diego course COGS 108 (Data Science in Practice)*
