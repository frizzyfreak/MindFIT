# MindFIT - Defense Personnel Mental Health Analysis

A comprehensive mental health analysis tool designed for defense personnel assessment using advanced AI techniques.

> **Project Background**: Developed during a 2-month internship at DRDO, this tool creates a Mental Health Assessment Questionnaire for Defence Personnel in India using K-means clustering to classify data, Random Forest Classifier (building up to 100 trees), and TextBlob for Semantic Analysis. It provides diagnosis along with radar charts & personalized feedback.

## 🚀 Live Demo

**[Try the app here](https://mindfit-drdo.streamlit.app/)**

## 📋 Features

- **Sentiment Analysis**: Natural language processing to analyze textual responses
- **Machine Learning Clustering**: K-means algorithm to identify mental health groups
- **Predictive Modeling**: Random Forest classifier for mental readiness prediction
- **Statistical Analysis**: Demographic patterns and correlations
- **Individual Assessment**: Comprehensive mental health evaluation tool including PHQ-9 screening

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **NLP**: TextBlob

## 📊 Key Capabilities

### Data Analysis
- Personnel search and filtering
- Demographic analysis
- Sentiment score distribution
- Statistical summaries

### Mental Health Assessment
- 6-dimensional assessment framework
- PHQ-9 depression screening
- Risk level categorization
- Personalized recommendations

### Visualizations
- Interactive charts and graphs
- Cluster analysis plots
- Demographic distributions
- Feature importance analysis

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/frizzyfreak/MindFIT.git
cd MindFIT
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
MindFIT/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── assets/                       # Visualization assets
│   ├── mental health groups distribution.png
│   ├── elbow method.png
│   ├── gender distribution.png
│   └── ...
└── MindFIT - Form Responses 1 (1).csv  # Data file
```

## 🔧 Usage

### Navigation Sections

1. **Overview**: General information about the analysis tool
2. **Data Analysis**: Explore dataset and search personnel
3. **Clustering Results**: View mental health group classifications
4. **Predictive Modeling**: Machine learning model performance
5. **Visualizations**: Charts and graphs from analysis
6. **Individual Assessment**: Complete mental health evaluation

### Individual Assessment Features

- **Personal Information**: Basic demographic data
- **Stress Response & Crisis Management**: Pressure handling capabilities
- **Daily Stress & Anxiety Assessment**: Regular stress patterns
- **Recovery & Coping Mechanisms**: Resilience evaluation
- **Team Dynamics & Leadership**: Interpersonal skills
- **PHQ-9 Depression Screening**: Clinical assessment tool
- **Support Systems**: Social and professional support evaluation

## 📈 Assessment Framework

The tool evaluates personnel across 6 key dimensions:

1. **Stress Response** - Reaction to high-pressure situations
2. **Anxiety Management** - Control over worry and fear
3. **Coping Skills** - Personal resilience mechanisms
4. **Overall Wellbeing** - General mental health status
5. **Support Systems** - Social and professional networks
6. **Clinical Risk** - Depression and risk indicators

## 🎯 Risk Classification

- **Good Mental Health**: Continue current practices
- **Some Concerns**: Monitor and consider preventive measures
- **Moderate Risk**: Consider professional counseling
- **High Risk**: Immediate professional attention required

## 🔒 Privacy & Security

- No personal data is stored permanently
- All assessments are processed locally
- Results are for guidance only and not medical diagnosis

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is developed for DRDO-SSPL research purposes.

## 👥 Contact

For questions or support regarding this mental health analysis tool, please contact the development team.

---

**Developed for DRDO-SSPL | 2025**
