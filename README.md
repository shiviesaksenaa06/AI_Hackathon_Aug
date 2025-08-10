FinSight GPT is an AI-powered financial analysis system that transforms how investment decisions are made. By combining Google's Gemini Pro AI, advanced ML forecasting models, and real-time market data, it provides instant insights, predictions, and actionable investment recommendations for Microsoft (MSFT) stock.

# Features
## Stage 1: Document Q&A & Analysis
•⁠  ⁠💬 AI-Powered Q&A: Ask questions about Microsoft's financials in natural language

•⁠  ⁠📊 Sentiment Analysis: Real-time market sentiment scoring

•⁠  ⁠🔍 Anomaly Detection: Automatic flagging of unusual financial metrics

•⁠  ⁠📈 Key Insights: Revenue growth, cloud performance, profitability metrics

## Stage 2: Financial Forecasting
•⁠  ⁠📉 Multi-Model Predictions: Prophet, Random Forest, and Linear Regression

•⁠  ⁠📅 Flexible Timeframes: 7, 30, and 90-day forecasts

•⁠  ⁠📊 Technical Indicators: RSI, SMA20, SMA50, trend analysis

•⁠  ⁠🎯 Confidence Intervals: Upper and lower bounds for predictions

## Stage 3: Investment Strategy
•⁠  ⁠🤖 Automated Recommendations: Clear BUY/SELL/HOLD decisions

•⁠  ⁠💯 Confidence Scoring: 0-100% confidence in recommendations

•⁠  ⁠📋 Investment Rationale: AI-generated explanations

•⁠  ⁠⚠️ Risk Assessment: Identified risks with severity levels

#  Technology Stack

•⁠  ⁠Backend: FastAPI, Python 3.11

•⁠  ⁠AI/ML: Google Gemini Pro, Prophet, scikit-learn

•⁠  ⁠Frontend: HTML5, JavaScript, Chart.js, Tailwind CSS

•⁠  ⁠Data Sources: Yahoo Finance, Alpha Vantage

•⁠  ⁠NLP: TextBlob for sentiment analysis

# Prerequisites
•⁠  ⁠Python 3.11+

•⁠  ⁠API Keys:

    Google Gemini API Key (openly available)
    
    Alpha Vantage API Key (openly available)

# Quick Start: 

## 1.⁠ ⁠Clone the Repository

bash
git clone [https://github.com/yourusername/findocgpt.git](https://github.com/shiviesaksenaa06/AI_Hackathon_Aug.git)
cd findocgpt

##2.⁠ ⁠Set Up Backend

bash
### Navigate to backend
cd backend

### Create virtual environment
python -m venv venv

### Activate virtual environment
### On Mac/Linux:
source venv/bin/activate
### On Windows:
venv\Scripts\activate

## Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
echo "ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here" >> .env

## Run the backend
python main.py

# Set Up Frontend


bash
## Open a new terminal and navigate to frontend
cd frontend

## Serve the frontend
### Option 1: Using Python
python -m http.server 3000

### Option 2: Using Node.js
npx serve -p 3000

### Option 3: Simply open in browser
open index.html  # Mac
start index.html # Windows

# Configuration

## Environment Variables
Create a .env file in the backend directory:


### env

GEMINI_API_KEY=your_gemini_api_key_here

ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here  # Optional

# 📊 Performance Metrics

•⁠  ⁠Q&A Accuracy: 100% for predefined financial queries

•⁠  ⁠Prediction Models: 3 different algorithms with fallbacks

•⁠  ⁠API Response Time: <2 seconds average

•⁠  ⁠Frontend Load Time: <1 second

•⁠  ⁠Auto-refresh: Every 30 seconds

# 🤝 Contributing

1.⁠ ⁠Fork the repository

2.⁠ ⁠Create your feature branch (git checkout -b feature/AmazingFeature)

3.⁠ ⁠Commit your changes (git commit -m 'Add some AmazingFeature')

4.⁠ ⁠Push to the branch (git push origin feature/AmazingFeature)

5.⁠ ⁠Open a Pull Request


# 🏆 Acknowledgments

•⁠  ⁠Built for the AkashX.ai Global AI Hackathon 2025

•⁠  ⁠Powered by Google Gemini Pro AI

•⁠  ⁠Financial data from Yahoo Finance and Alpha Vantage

•⁠  ⁠UI components from Tailwind CSS and Chart.js

👥 Team
