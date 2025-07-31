# Defense Personnel Mental Health Analysis - Streamlit App

This Streamlit application provides an interactive web interface for the AI-driven mental health analysis of defense personnel.

## Features

- **Overview**: Project introduction and key metrics
- **Data Analysis**: Dataset exploration and summary statistics
- **Clustering Results**: Mental health group analysis and demographics
- **Predictive Modeling**: Machine learning model performance and insights
- **Visualizations**: Interactive charts and saved analysis plots
- **Individual Assessment**: Real-time mental health assessment tool

## Installation and Setup

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DRDO-SSPL/SUKHMANI.git
   cd SUKHMANI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   Open your browser and go to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub** (already done!)

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `DRDO-SSPL/SUKHMANI`
   - Set main file path: `app.py`
   - Click "Deploy"

### Alternative Deployment Options

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku (requires Heroku CLI)
heroku create your-app-name
git push heroku main
```

#### Docker
```dockerfile
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## File Structure

```
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── MindFIT - Form Responses 1 (1).csv # Dataset
├── assets/                            # Visualization images
│   ├── mental health groups distribution.png
│   ├── elbow method.png
│   ├── gender distribution.png
│   ├── Role distribution withi clusters.png
│   ├── Age Distribution with clusters.png
│   └── questions that classified the most.png
├── Defense_Personnel_Mental_Health_EDA.ipynb # Jupyter notebook
└── analysis_summary.md               # Analysis documentation
```

## Usage

1. **Navigation**: Use the sidebar to switch between different sections
2. **Data Analysis**: Explore the dataset and view summary statistics
3. **Clustering**: Analyze mental health groups and demographic patterns
4. **Modeling**: View machine learning model performance and feature importance
5. **Visualizations**: Browse interactive charts and saved plots
6. **Assessment**: Use the individual assessment tool for real-time evaluation

## Data Requirements

The application expects a CSV file named `MindFIT - Form Responses 1 (1).csv` with the following structure:
- Timestamp, Name, Gender, Role columns for metadata
- Numeric columns for Likert scale responses (1-5)
- Text columns for open-ended responses

## Technical Details

- **Framework**: Streamlit
- **Machine Learning**: Scikit-learn (K-means clustering, Random Forest)
- **NLP**: TextBlob for sentiment analysis
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## Support

For issues or questions, please create an issue in the GitHub repository.

## License

This project is developed for DRDO-SSPL research purposes.
