# MindFIT - Defense Personnel Mental Health Analysis

A comprehensive mental health analysis tool designed for defense personnel assessment using advanced AI techniques including K-means clustering, Random Forest classification, and sentiment analysis.

> **Project Background**: Developed during a 2-month internship at DRDO, this tool creates a Mental Health Assessment Questionnaire for Defence Personnel in India. The system uses K-means clustering to classify data, Random Forest Classifier (building up to 100 trees), and TextBlob for Semantic Analysis, providing diagnosis along with radar charts & personalized feedback.

##  Live Demo

**[Try the app here](https://mindfit-drdo.streamlit.app/)**

---

##  Key Features

###  **Advanced Analytics**
- **Sentiment Analysis**: Natural language processing using TextBlob to analyze textual responses
- **Machine Learning Clustering**: K-means algorithm to identify mental health groups (85% accuracy)
- **Predictive Modeling**: Random Forest classifier with 100 trees for mental readiness prediction
- **Statistical Analysis**: Comprehensive demographic patterns and correlations

###  **Interactive Assessment**
- **5-Dimensional Core Assessment**: Balanced evaluation framework
- **PHQ-9 Depression Screening**: Separate clinical risk evaluation
- **Real-time Risk Classification**: Immediate feedback with personalized recommendations
- **Visual Analytics**: Interactive radar charts and demographic visualizations

###  **Data Exploration**
- **Personnel Search & Filtering**: Advanced search capabilities
- **Demographic Analysis**: Gender, age, and role distribution patterns
- **Cluster Visualization**: Interactive charts showing mental health group distributions

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (K-means, Random Forest)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **NLP**: TextBlob for sentiment analysis
- **Deployment**: Streamlit Cloud

---

## 📊 Research Findings & Analysis

### 🎯 **Cluster Analysis Results**

Our analysis successfully identified three distinct mental health groups:

![Mental Health Groups Distribution](assets/mental%20health%20groups%20distribution.png)
<p align="center"><em>Distribution of personnel across High Health, Moderate Health, and Needs Support clusters</em></p>

![Elbow Method](assets/elbow%20method.png)
<p align="center"><em>Elbow method validation showing optimal k=3 clusters for mental health classification</em></p>

### 🔍 **Key Differentiating Factors**

The analysis revealed critical questions that effectively distinguish between mental health groups:

![Questions That Classified the Most](assets/questions%20that%20classified%20the%20most.png)
<p align="center"><em>Most discriminatory questions for cluster classification</em></p>

**Primary Indicators:**
- **Worry and Anxiety Patterns**: "Do you worry about different things more than most people?"
- **Self-Perception**: "Do you think your mental health is as good as most people's?"
- **Crisis Decision-Making**: Responses to teammate violations and mistake handling
- **Emotional Regulation**: Ability to identify and manage emotional reactions

### 👥 **Demographic Insights**

![Gender Distribution](assets/gender%20distribution.png) ![Role Distribution Within Clusters](assets/Role%20distribution%20withi%20clusters.png)
<p align="center"><em>Gender and role distribution patterns across mental health clusters</em></p>

![Age Distribution with Clusters](assets/Age%20Distribution%20with%20clusters.png)
<p align="center"><em>Age independence in mental health classification</em></p>

**Key Findings:**
- **Gender**: Slightly higher male representation in 'Needs Support' group
- **Leadership Roles**: Captains and Majors more prevalent in 'High Health' cluster
- **Age Factor**: Mental readiness shows minimal age correlation, indicating psychological rather than demographic determinants

### 🎯 **Model Performance**
- **Accuracy**: 85% on test set
- **High Precision & Recall**: Across all clusters
- **Robust Classification**: Validated clustering methodology

---

## 📈 Assessment Framework

### 🎨 **Core Mental Health Dimensions** (Radar Chart)
1. **Stress Response** - Reaction to high-pressure situations
2. **Anxiety Management** - Control over worry and fear
3. **Coping Skills** - Personal resilience mechanisms
4. **Overall Wellbeing** - General mental health status
5. **Support Systems** - Social and professional networks

### 🏥 **Clinical Assessment** (Separate Evaluation)
- **PHQ-9 Depression Screening** - Professional clinical indicators
- **Risk Level Classification** - Immediate attention requirements

### 🚨 **Risk Classification Levels**
- **🟢 Good Mental Health**: Continue current wellness practices
- **🟡 Some Concerns**: Monitor and consider preventive measures
- **🟠 Moderate Risk**: Consider professional counseling
- **🔴 High Risk**: Immediate professional attention required

---

## 🚀 Installation & Setup

### 1. **Clone Repository**
```bash
git clone https://github.com/frizzyfreak/MindFIT.git
cd MindFIT
```

### 2. **Create Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Run Application**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
MindFIT/
├── 📱 app.py                          # Main Streamlit application
├── 📦 requirements.txt                # Python dependencies
├── 📚 README.md                      # Project documentation
├── ⚙️ .streamlit/
│   └── config.toml                   # Streamlit configuration
├── 🖼️ assets/                        # Analysis visualizations
│   ├── mental health groups distribution.png
│   ├── elbow method.png
│   ├── gender distribution.png
│   ├── Role distribution withi clusters.png
│   ├── Age Distribution with clusters.png
│   └── questions that classified the most.png
├── 📊 MindFIT - Form Responses 1 (1).csv  # Assessment data
├── 📄 Clinical_Validation_Methodology.md
├── 📄 Technical_Implementation_Details.md
├── 📄 DEPLOYMENT.md
└── 📄 analysis_summary.md
```

---

## 🔧 Application Features

### 📋 **Navigation Sections**

1. **📊 Overview**: Project introduction and key metrics
2. **🔍 Data Analysis**: Dataset exploration with personnel search
3. **🎯 Clustering Results**: Mental health group classifications
4. **🤖 Predictive Modeling**: Machine learning model performance
5. **📈 Visualizations**: Research findings and charts
6. **🧠 Individual Assessment**: Complete mental health evaluation tool

### 🩺 **Assessment Components**

- **Personal Information**: Demographics and consent
- **Stress Response & Crisis Management**: Pressure handling capabilities
- **Daily Stress & Anxiety**: Regular stress patterns evaluation
- **Recovery & Coping Mechanisms**: Resilience assessment
- **Team Dynamics & Leadership**: Interpersonal skills evaluation
- **PHQ-9 Depression Screening**: Clinical assessment tool
- **Support Systems**: Social and professional network evaluation

---

## 🎯 Research Conclusions & Recommendations

### ✅ **Validated Findings**
- **Reliable Classification**: 85% accuracy in predicting mental health groups
- **Behavioral Indicators**: Specific response patterns distinguish mental readiness levels
- **Demographic Independence**: Mental health primarily determined by psychological factors

### 🎯 **Targeted Interventions**
- **'Needs Support' Group**: Focus on anxiety management and emotional regulation programs
- **Leadership Development**: Incorporate 'High Health' cluster response patterns in training
- **Proactive Approach**: Early identification enables preventive mental health strategies

### 🛡️ **Organizational Benefits**
- **Mission Readiness**: Enhanced force resilience through targeted support
- **Resource Optimization**: Efficient allocation of mental health resources
- **Data-Driven Decisions**: Evidence-based personnel management strategies

---

## 🔒 Privacy & Security

- ✅ **No Permanent Storage**: Personal data processed locally only
- ✅ **Confidential Assessment**: Individual results remain private
- ✅ **Guidance Only**: Results supplement, not replace, professional diagnosis
- ✅ **Consent-Based**: Voluntary participation with clear consent requirements

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is developed for **DRDO-SSPL research purposes** and defense personnel mental health assessment.

---

## 👥 Contact & Support

For questions, technical support, or collaboration inquiries regarding this mental health analysis tool, please contact the development team.

**Research Institution**: Defence Research and Development Organisation (DRDO)  
**Laboratory**: Solid State Physics Laboratory (SSPL)  
**Project Type**: Internship Research Project  
**Year**: 2025

---

<div align="center">

**🛡️ Developed for DRDO-SSPL | 2025 🛡️**

*Enhancing Defense Personnel Mental Readiness Through Advanced Analytics*

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mindfit-drdo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>
