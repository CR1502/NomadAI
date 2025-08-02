# 🌍 Lifestyle Discovery Assistant - Nomad AI

A comprehensive lifestyle discovery platform that combines **real Reddit community insights** with **verified business data** to provide intelligent travel recommendations. Built with advanced data engineering, machine learning, and a beautiful user interface.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-green.svg)
![AWS](https://img.shields.io/badge/AWS-S3-orange.svg)
![Google](https://img.shields.io/badge/Google-Places%20API-red.svg)

## ✨ Key Features

### 🧠 Smart AI Trip Planner (100% Free)
- **Personalized itineraries** generated using local ML algorithms
- **Reddit community insights** as RAG context for recommendations
- **No external AI API costs** - completely self-contained
- **User preference customization** (budget, interests, travel style)

### 📊 Advanced Data Pipeline
- **2,451+ Reddit posts** extracted from 25 top destinations
- **Real-time Reddit API integration** with rate limiting and error handling
- **AWS S3 storage** for scalable data management
- **Apache Airflow orchestration** for automated data collection

### 🏪 Verified Business Data
- **Google Places API integration** for restaurants and attractions
- **Real ratings, prices, and contact information**
- **Address verification and business hours**
- **Cross-referenced with community recommendations**

### 🤖 Machine Learning Features
- **Advanced sentiment analysis** using TextBlob + custom travel sentiment models
- **Named Entity Recognition** with spaCy for location and business extraction
- **Duplicate detection** using TF-IDF similarity and fuzzy matching
- **Quality scoring** with multiple ML factors

### 💰 Cost Intelligence
- **Realistic budget estimates** for all 25 destinations
- **Daily cost breakdowns** (accommodation, food, transport, attractions)
- **Trip duration planning** (3 days to 2 weeks)
- **Price-level integration** from Google Places data

## 🗺️ Supported Destinations

**25 Top Global Destinations:**
Paris • London • New York • Tokyo • Rome • Barcelona • Amsterdam • Prague • Vienna • Berlin • Istanbul • Dubai • Bangkok • Singapore • Hong Kong • Sydney • Los Angeles • Chicago • Las Vegas • Miami • San Francisco • Venice • Florence • Athens • Lisbon

## 🛠️ Technology Stack

### **Backend**
- **Python 3.12** - Core application logic
- **Streamlit** - Web application framework
- **Reddit API (PRAW)** - Community data extraction
- **Google Places API** - Verified business data
- **AWS S3** - Cloud data storage
- **PostgreSQL** - User data and caching

### **Machine Learning**
- **scikit-learn** - TF-IDF similarity and clustering
- **TextBlob** - Sentiment analysis
- **spaCy** - Named entity recognition
- **sentence-transformers** - Text embeddings
- **fuzzywuzzy** - Duplicate detection

### **Data Processing**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **Apache Airflow** - Data pipeline orchestration
- **BeautifulSoup** - Web scraping capabilities

### **Visualization**
- **Plotly** - Interactive charts and analytics
- **Custom CSS** - Responsive design and dark theme
- **Streamlit Components** - Enhanced UI elements

## 📈 Project Statistics

- **2,451+ Reddit Posts** analyzed across all destinations
- **896 Travel insights** from community experiences
- **803 Food recommendations** from local experts
- **752 Event suggestions** for cultural experiences
- **25 Global destinations** with comprehensive data
- **100% Free AI** with no external API dependencies

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Git
- Docker (optional)

### 1. Clone the Repository
```bash
git clone https://github.com/CR1502/lifestyle_RAG.git
cd lifestyle-discovery-assistant
```
### 2. Set Up Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for enhanced features
python -m spacy download en_core_web_sm
```
### 3. Configure API Keys
Create docker/.env file with your credentials:
```bash
# Reddit API (Free)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=lifestyle_discovery_bot_v1.0

# Google Places API
GOOGLE_PLACES_API_KEY=your_google_places_api_key

# AWS S3 (Free Tier)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```
### 4. Extract Reddit Data
```bash
# Extract fresh Reddit data (takes 30-45 minutes)
python3 -m src.data_pipeline.reddit_extractor
```
### 5. Run the Application
```bash
# Start the web application
streamlit run src/webapp.py
```
-----

# 📁 Project Structure
```bash
lifestyle-discovery-assistant/
├── src/
│   ├── data_pipeline/           # Data extraction and processing
│   │   ├── reddit_extractor.py  # Reddit API integration with comments
│   │   ├── data_processor.py    # ML-powered data enhancement
│   │   └── airflow_dag.py       # Automated data pipeline
│   ├── models/                  # Machine learning models
│   │   ├── embedding_model.py   # Text embeddings and similarity
│   │   └── model_compressor.py  # Model optimization
│   ├── web_app/                 # Streamlit application
│   │   ├── enhanced_app.py      # Main application with all features
│   │   └── components/          # Reusable UI components
│   └── utils/                   # Helper functions and utilities
├── data/
│   ├── by_location/             # Reddit data organized by destination
│   ├── processed/               # ML-enhanced and cleaned data
│   └── summaries/               # Analytics and extraction summaries
├── config/                      # Configuration files
├── docker/                      # Docker configuration and environment
├── docs/                        # Documentation and guides
└── requirements.txt             # Python dependencies
```
----

## 🙏 Acknowledgments

Reddit API for providing access to community discussions
Google Places API for verified business data
AWS for reliable cloud storage
Streamlit for the amazing web framework
spaCy & TextBlob for natural language processing
Open source community for the incredible tools and libraries


