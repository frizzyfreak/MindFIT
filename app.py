import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Defense Personnel Mental Health Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">Defense Personnel Mental Health Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "Overview",
    "Data Analysis",
    "Clustering Results",
    "Predictive Modeling",
    "Visualizations",
    "Individual Assessment"
])

@st.cache_data
def load_data():
    """Load and preprocess the mental health data"""
    try:
        # Try to load the CSV file
        df = pd.read_csv("MindFIT - Form Responses 1 (1).csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'MindFIT - Form Responses 1 (1).csv' is in the same directory.")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess the data and perform sentiment analysis"""
    if df is None:
        return None, None, None
    
    # Identify numeric and textual columns
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    textual_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Remove non-response columns
    non_response_columns = ["Timestamp", "Name", "Gender", "Role"]
    textual_columns = [col for col in textual_columns if col not in non_response_columns]
    
    # Sentiment analysis for textual columns
    sentiment_results_text = {}
    for col in textual_columns:
        if col in df.columns:
            sentiments = df[col].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
            sentiment_results_text[col] = sentiments
    
    sentiment_df_text = pd.DataFrame(sentiment_results_text)
    
    # Map numeric responses to sentiment scores
    def map_numeric_sentiment(value):
        if pd.isna(value):
            return 0
        if value <= 2:
            return -1
        elif value == 3:
            return 0
        else:
            return 1
    
    numeric_sentiment_df = df[numeric_columns].applymap(map_numeric_sentiment)
    
    # Combine sentiment scores
    combined_sentiment_df = pd.concat([sentiment_df_text, numeric_sentiment_df], axis=1)
    
    # Calculate total sentiment score
    df["Total_Sentiment_Score"] = combined_sentiment_df.sum(axis=1)
    
    return df, combined_sentiment_df, sentiment_df_text

@st.cache_data
def perform_clustering(combined_sentiment_df):
    """Perform K-means clustering"""
    if combined_sentiment_df is None:
        return None, None
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_sentiment_df.fillna(0))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    return clusters, kmeans

def create_cluster_distribution_plot(df):
    """Create cluster distribution plot"""
    cluster_counts = df["Sentiment_Cluster"].value_counts().sort_index()
    
    # Map cluster numbers to meaningful names
    cluster_names = ['High Health', 'Moderate Health', 'Needs Support']
    
    fig = px.bar(
        x=cluster_names,
        y=cluster_counts.values,
        title="Mental Health Groups Distribution",
        labels={'x': 'Health Group', 'y': 'Number of Personnel'},
        color=cluster_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

def create_demographic_plots(df):
    """Create demographic analysis plots"""
    plots = {}
    
    # Create a mapping for cluster names
    cluster_mapping = {0: 'High Health', 1: 'Moderate Health', 2: 'Needs Support'}
    df_mapped = df.copy()
    df_mapped['Cluster_Name'] = df_mapped['Sentiment_Cluster'].map(cluster_mapping)
    
    # Gender distribution
    if 'Gender' in df.columns:
        gender_cluster = pd.crosstab(df_mapped['Gender'], df_mapped['Cluster_Name'])
        fig_gender = px.bar(
            gender_cluster,
            title="Gender Distribution Across Clusters",
            labels={'value': 'Count', 'index': 'Gender'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        plots['gender'] = fig_gender
    
    # Role distribution
    if 'Role' in df.columns:
        role_cluster = pd.crosstab(df_mapped['Role'], df_mapped['Cluster_Name'])
        fig_role = px.bar(
            role_cluster,
            title="Role Distribution Across Clusters",
            labels={'value': 'Count', 'index': 'Role'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_role.update_xaxes(tickangle=45)
        plots['role'] = fig_role
    
    return plots

# Load and preprocess data
df = load_data()
if df is not None:
    df_processed, combined_sentiment_df, sentiment_df_text = preprocess_data(df)
    
    if df_processed is not None and combined_sentiment_df is not None:
        # Perform clustering
        clusters, kmeans_model = perform_clustering(combined_sentiment_df)
        if clusters is not None:
            df_processed["Sentiment_Cluster"] = clusters
            # Add meaningful cluster names
            cluster_mapping = {0: 'High Health', 1: 'Moderate Health', 2: 'Needs Support'}
            df_processed["Cluster_Name"] = df_processed["Sentiment_Cluster"].map(cluster_mapping)

# Page content based on selection
if page == "Overview":
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Analysis
        
        This comprehensive mental health analysis tool is designed for defense personnel assessment. 
        The system uses advanced AI techniques including:
        
        - **Sentiment Analysis**: Natural language processing to analyze textual responses
        - **Machine Learning Clustering**: K-means algorithm to identify mental health groups
        - **Predictive Modeling**: Random Forest classifier for mental readiness prediction
        - **Statistical Analysis**: Demographic patterns and correlations
        
        ### Key Features
        - Real-time data processing and visualization
        - Interactive dashboards and charts
        - Individual assessment capabilities
        - Comprehensive reporting system
        """)
    
    with col2:
        if df is not None:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Personnel", len(df))
            st.metric("Assessment Questions", len(df.columns) - 4)  # Excluding metadata columns
            if 'Sentiment_Cluster' in df.columns:
                st.metric("Mental Health Groups", df['Sentiment_Cluster'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Data Analysis":
    st.markdown('<h2 class="sub-header">Data Analysis</h2>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Sample Data:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Add search functionality
            st.subheader("🔍 Personnel Search")
            search_name = st.text_input("Search by Name:", placeholder="Enter personnel name...")
            
            if search_name:
                # Search for matching names (case-insensitive)
                matching_personnel = df[df['Name'].str.contains(search_name, case=False, na=False)]
                
                if not matching_personnel.empty:
                    st.success(f"Found {len(matching_personnel)} matching personnel:")
                    
                    # Display matching personnel with key information
                    display_cols = ['Name', 'Age', 'Gender', 'Role']
                    if df_processed is not None and 'Total_Sentiment_Score' in df_processed.columns:
                        # Add sentiment score if available
                        matching_with_scores = matching_personnel.copy()
                        matching_indices = matching_personnel.index
                        matching_with_scores['Total_Sentiment_Score'] = df_processed.loc[matching_indices, 'Total_Sentiment_Score']
                        display_cols.append('Total_Sentiment_Score')
                        
                        # Add cluster information if available
                        if 'Sentiment_Cluster' in df_processed.columns:
                            cluster_mapping = {0: 'High Health', 1: 'Moderate Health', 2: 'Needs Support'}
                            matching_with_scores['Mental_Health_Group'] = df_processed.loc[matching_indices, 'Sentiment_Cluster'].map(cluster_mapping)
                            display_cols.append('Mental_Health_Group')
                        
                        st.dataframe(matching_with_scores[display_cols], use_container_width=True)
                    else:
                        st.dataframe(matching_personnel[display_cols], use_container_width=True)
                        
                    # Show detailed view for exact matches
                    exact_matches = df[df['Name'].str.lower() == search_name.lower()]
                    if not exact_matches.empty:
                        st.subheader("📋 Detailed Information")
                        selected_person = exact_matches.iloc[0]
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Name", selected_person['Name'])
                            st.metric("Age", selected_person['Age'])
                        with col_b:
                            st.metric("Gender", selected_person['Gender'])
                            st.metric("Role", selected_person['Role'])
                        with col_c:
                            if df_processed is not None and 'Total_Sentiment_Score' in df_processed.columns:
                                person_idx = exact_matches.index[0]
                                sentiment_score = df_processed.loc[person_idx, 'Total_Sentiment_Score']
                                st.metric("Sentiment Score", f"{sentiment_score:.2f}")
                                
                                if 'Sentiment_Cluster' in df_processed.columns:
                                    cluster = df_processed.loc[person_idx, 'Sentiment_Cluster']
                                    cluster_name = {0: 'High Health', 1: 'Moderate Health', 2: 'Needs Support'}[cluster]
                                    color = {'High Health': 'green', 'Moderate Health': 'orange', 'Needs Support': 'red'}[cluster_name]
                                    st.markdown(f"**Mental Health Group:** <span style='color:{color}'>{cluster_name}</span>", unsafe_allow_html=True)
                else:
                    st.warning(f"No personnel found with name containing '{search_name}'. Please check the spelling or try a different search term.")
                    
                    # Show suggestions - names that are similar
                    all_names = df['Name'].str.lower().tolist()
                    search_lower = search_name.lower()
                    suggestions = [name for name in df['Name'].unique() if search_lower in name.lower()]
                    
                    if suggestions:
                        st.info("**Suggestions:**")
                        for suggestion in suggestions[:5]:  # Show max 5 suggestions
                            st.write(f"• {suggestion}")
        
        with col2:
            st.subheader("Data Summary")
            if df_processed is not None and 'Total_Sentiment_Score' in df_processed.columns:
                st.write("**Sentiment Score Statistics:**")
                st.write(df_processed['Total_Sentiment_Score'].describe())
                
                # Sentiment score distribution
                fig_hist = px.histogram(
                    df_processed, 
                    x='Total_Sentiment_Score',
                    title="Distribution of Total Sentiment Scores",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Sentiment analysis in progress... Please wait for data processing to complete.")
    else:
        st.error("No data available for analysis.")

elif page == "Clustering Results":
    st.markdown('<h2 class="sub-header">Clustering Results</h2>', unsafe_allow_html=True)
    
    if df_processed is not None and 'Sentiment_Cluster' in df_processed.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            fig_cluster = create_cluster_distribution_plot(df_processed)
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        with col2:
            # Cluster characteristics
            st.subheader("Cluster Characteristics")
            cluster_summary = df_processed.groupby('Sentiment_Cluster')['Total_Sentiment_Score'].agg(['mean', 'count', 'std']).round(2)
            cluster_summary.index = ['High Health', 'Moderate Health', 'Needs Support']
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Show cluster distribution with names
            if 'Cluster_Name' in df_processed.columns:
                st.subheader("Cluster Distribution")
                cluster_dist = df_processed['Cluster_Name'].value_counts()
                st.dataframe(cluster_dist.to_frame('Count'), use_container_width=True)
        
        # Demographic analysis
        demographic_plots = create_demographic_plots(df_processed)
        
        if demographic_plots:
            st.subheader("Demographic Analysis")
            for plot_name, plot_fig in demographic_plots.items():
                st.plotly_chart(plot_fig, use_container_width=True)
    else:
        st.error("Clustering analysis not available. Please check the data.")

elif page == "Predictive Modeling":
    st.markdown('<h2 class="sub-header">Predictive Modeling</h2>', unsafe_allow_html=True)
    
    if df_processed is not None and combined_sentiment_df is not None and 'Sentiment_Cluster' in df_processed.columns:
        # Train Random Forest model
        X = combined_sentiment_df.fillna(0)
        y = df_processed['Sentiment_Cluster']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("Classification Report")
            class_report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(class_report).transpose().round(2)
            # Update index names for better readability
            if '0' in report_df.index:
                index_mapping = {'0': 'High Health', '1': 'Moderate Health', '2': 'Needs Support'}
                report_df.index = report_df.index.map(lambda x: index_mapping.get(x, x))
            st.dataframe(report_df, use_container_width=True)
    else:
        st.error("Predictive modeling not available. Please check the data.")

elif page == "Visualizations":
    st.markdown('<h2 class="sub-header">Visualizations</h2>', unsafe_allow_html=True)
    
    # Display saved visualizations from assets folder
    asset_files = {
        "Mental Health Groups Distribution": "mental health groups distribution.png",
        "Elbow Method": "elbow method.png",
        "Gender Distribution": "gender distribution.png",
        "Role Distribution": "Role distribution withi clusters.png",
        "Age Distribution": "Age Distribution with clusters.png",
        "Key Questions": "questions that classified the most.png"
    }
    
    for title, filename in asset_files.items():
        filepath = f"assets/{filename}"
        if os.path.exists(filepath):
            st.subheader(title)
            image = Image.open(filepath)
            st.image(image, use_column_width=True)
        else:
            st.warning(f"Visualization '{title}' not found at {filepath}")

elif page == "Individual Assessment":
    st.markdown('<h2 class="sub-header">Individual Assessment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Comprehensive Mental Health Assessment Tool
    
    This tool allows for individual assessment based on the trained model using all survey questions.
    Please answer all questions honestly to get an accurate mental health readiness prediction.
    """)
    
    # Create comprehensive assessment form
    with st.form("assessment_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, value=25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            role = st.selectbox("Role", [
                "Corporal", "Junior Warrant Officer", "Lieutenant Commander", 
                "Flight Lieutenant", "Major", "Captain", "Sergeant", "Other"
            ])
            consent = st.checkbox("Do you consent to participate in this self-assessment survey?", value=True)
        
        st.subheader("Current Mental Health Status")
        under_care = st.selectbox("Are you currently under any professional mental health care?", ["No", "Yes"])
        
        st.subheader("Stress Response & Crisis Management")
        reaction_call = st.slider("What's your usual reaction when someone suddenly calls your name from behind? (1=Very startled, 5=Calm)", 1, 5, 3)
        crisis_handling = st.slider("Would you rather deal with a crisis alone or involve others quickly? (1=Alone, 5=Involve others)", 1, 5, 3)
        crisis_priority = st.slider("In a crisis, what do you do first? (1=Assist others, 5=Control own reaction)", 1, 5, 3)
        pressure_reaction = st.slider("How do you react when under pressure? (1=Poor, 5=Excellent)", 1, 5, 3)
        anxiety_handling = st.slider("What do you do when you feel anxious or overwhelmed? (1=Poor coping, 5=Excellent coping)", 1, 5, 3)
        
        st.subheader("Daily Stress & Anxiety Assessment")
        daily_stress = st.slider("Do you handle daily stress and problems effectively? (1=Never, 5=Always)", 1, 5, 3)
        duty_overwhelm = st.slider("How often do you feel overwhelmed during duty? (1=Always, 5=Never)", 1, 5, 3)
        worry_control = st.slider("Do you find it difficult to stop or control your worrying? (1=Always difficult, 5=Never difficult)", 1, 5, 3)
        worry_level = st.slider("Do you worry about different things more than most people? (1=Much more, 5=Much less)", 1, 5, 3)
        avoidance = st.slider("Do you avoid certain situations because of anxiety or fear? (1=Always, 5=Never)", 1, 5, 3)
        panic_attacks = st.slider("Do you experience sudden panic or intense fear? (1=Very often, 5=Never)", 1, 5, 3)
        relaxation = st.slider("Do you struggle to relax most days? (1=Always struggle, 5=Never struggle)", 1, 5, 3)
        worry_prevention = st.slider("Do you believe worrying prevents bad things from happening? (1=Strongly believe, 5=Don't believe)", 1, 5, 3)
        
        st.subheader("Recovery & Coping Mechanisms")
        unwind_ability = st.slider("Can you relax and unwind after duty without help? (1=Never, 5=Always)", 1, 5, 3)
        recovery_time = st.slider("How long does it take you to recover from high-stress operations? (1=Very long, 5=Very quick)", 1, 5, 3)
        coping_mechanisms = st.slider("Do you use personal coping mechanisms when under pressure? (1=None, 5=Very effective)", 1, 5, 3)
        
        st.subheader("Emotional Regulation & Decision Making")
        emotional_management = st.slider("Can you identify and manage your emotional reactions in crisis? (1=Never, 5=Always)", 1, 5, 3)
        decision_confidence = st.slider("Do you feel confident making quick decisions under pressure? (1=Never, 5=Always)", 1, 5, 3)
        
        st.subheader("Team Dynamics & Leadership")
        team_conflicts = st.slider("Have you had recent interpersonal conflicts within your unit? (1=Many, 5=None)", 1, 5, 3)
        teammate_violation = st.selectbox("What would you do if you see a teammate violating orders during a critical mission?", 
                                        ["Ignore it", "Confront immediately", "Report after mission", "Discuss privately later", "Seek guidance"])
        mistake_handling = st.selectbox("What would you do if you made a mistake affecting team performance?", 
                                      ["Hide it", "Blame others", "Take responsibility immediately", "Fix quietly", "Seek help"])
        
        st.subheader("Purpose & Motivation")
        sense_of_purpose = st.slider("Do you have a strong sense of purpose or mission? (1=None, 5=Very strong)", 1, 5, 3)
        motivation = st.text_area("What motivates you to keep going when things get tough?", placeholder="Enter your motivation...")
        
        st.subheader("Overall Mental Health Perception")
        mental_health_comparison = st.slider("Do you think your mental health is as good as most people's? (1=Much worse, 5=Much better)", 1, 5, 3)
        future_hope = st.slider("Do you feel hopeful about your future most of the time? (1=Never, 5=Always)", 1, 5, 3)
        relationships = st.slider("Are you satisfied with your relationships and social connections? (1=Very unsatisfied, 5=Very satisfied)", 1, 5, 3)
        community_contribution = st.slider("Are you able to contribute to your community or society? (1=Never, 5=Always)", 1, 5, 3)
        emotional_stability = st.slider("Do you feel emotionally stable today? (1=Very unstable, 5=Very stable)", 1, 5, 3)
        
        st.subheader("Communication & Support")
        open_communication = st.slider("Can you talk openly with peers about emotions? (1=Never, 5=Always)", 1, 5, 3)
        
        st.subheader("Depression Screening (PHQ-9 Style)")
        little_interest = st.selectbox("Little interest or pleasure in doing things", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        feeling_down = st.selectbox("Feeling down, depressed, or hopeless", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        sleep_trouble = st.selectbox("Trouble sleeping or sleeping too much", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        tired_energy = st.selectbox("Feeling tired or having little energy", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        appetite_changes = st.selectbox("Changes in appetite", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        feeling_failure = st.selectbox("Feeling bad about yourself or like a failure", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        concentration = st.selectbox("Trouble concentrating", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        slow_restless = st.selectbox("Moving or speaking unusually slowly or being fidgety", ["Not at all", "Several days", "More than half the days", "Nearly every day"])
        
        st.subheader("Critical Mental Health Indicators")
        persistent_sadness = st.selectbox("Have you had a persistently sad, empty mood for the past two weeks?", ["No", "Yes", "Maybe"])
        appetite_weight = st.selectbox("Have you experienced significant appetite or weight changes?", ["No", "Yes", "Maybe"])
        suicidal_thoughts = st.selectbox("Are you currently experiencing any suicidal thoughts?", ["No", "Yes", "Prefer not to answer"])
        
        st.subheader("Support Systems")
        emotional_support = st.slider("Do you feel emotionally supported by those around you? (1=Never, 5=Always)", 1, 5, 3)
        team_support = st.slider("Do you feel supported by your team/unit? (1=Never, 5=Always)", 1, 5, 3)
        isolation = st.slider("Do you feel isolated or disconnected? (1=Always, 5=Never)", 1, 5, 3)
        
        submitted = st.form_submit_button("Get Comprehensive Assessment", type="primary")
        
        if submitted and consent:
            # Calculate scores for different categories
            stress_scores = [reaction_call, crisis_handling, crisis_priority, pressure_reaction, anxiety_handling]
            anxiety_scores = [daily_stress, duty_overwhelm, worry_control, worry_level, avoidance, panic_attacks, relaxation]
            coping_scores = [unwind_ability, recovery_time, coping_mechanisms, emotional_management, decision_confidence]
            wellbeing_scores = [mental_health_comparison, future_hope, relationships, community_contribution, emotional_stability]
            support_scores = [emotional_support, team_support, isolation, open_communication]
            
            # Convert text responses to numeric scores for depression screening
            response_to_score = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
            depression_items = [little_interest, feeling_down, sleep_trouble, tired_energy, appetite_changes, 
                              feeling_failure, concentration, slow_restless]
            depression_score = sum([response_to_score.get(item, 0) for item in depression_items])
            
            # Create Clinical Risk Indicator (inverted scale: lower scores = higher risk)
            clinical_risk_score = max(0, 5 - (depression_score / 4))  # Inverted relationship
            
            # Calculate overall assessment (6 dimensions including clinical risk)
            avg_stress = np.mean(stress_scores)
            avg_anxiety = np.mean(anxiety_scores)
            avg_coping = np.mean(coping_scores)
            avg_wellbeing = np.mean(wellbeing_scores)
            avg_support = np.mean(support_scores)
            
            # Overall score includes the 5 core dimensions (clinical risk assessed separately)
            overall_score = np.mean([avg_stress, avg_anxiety, avg_coping, avg_wellbeing, avg_support])
            
            # Determine risk level
            if depression_score >= 15 or suicidal_thoughts == "Yes":
                risk_level = "High Risk"
                color = "red"
                recommendation = "🚨 **IMMEDIATE ATTENTION REQUIRED** - Please seek professional help immediately."
            elif depression_score >= 10 or overall_score <= 2.5:
                risk_level = "Moderate Risk"
                color = "orange"
                recommendation = "⚠️ Consider seeking professional counseling and support services."
            elif overall_score <= 3.5:
                risk_level = "Some Concerns"
                color = "yellow"
                recommendation = "💡 Monitor mental health closely and consider preventive measures."
            else:
                risk_level = "Good Mental Health"
                color = "green"
                recommendation = "✅ Continue maintaining current wellness practices."
            
            # Display results
            st.markdown("---")
            st.markdown("## Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Risk Level", risk_level)
                st.markdown(f"<div style='color:{color}; font-weight:bold;'>{recommendation}</div>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Depression Score (PHQ-9)", f"{depression_score}/24")
                if depression_score >= 15:
                    st.error("Severe depression symptoms detected")
                elif depression_score >= 10:
                    st.warning("Moderate depression symptoms")
                elif depression_score >= 5:
                    st.info("Mild depression symptoms")
                else:
                    st.success("Minimal depression symptoms")
            
            with col3:
                st.metric("Overall Wellness Score", f"{overall_score:.1f}/5.0")
                st.metric("Clinical Risk Indicator", f"{clinical_risk_score:.1f}/5.0")
            
            # Detailed breakdown (Core 5 categories + Clinical Risk displayed separately)
            st.subheader("Core Mental Health Dimensions")
            categories_df = pd.DataFrame({
                'Category': ['Stress Response', 'Anxiety Management', 'Coping Skills', 'Overall Wellbeing', 'Support Systems'],
                'Score': [avg_stress, avg_anxiety, avg_coping, avg_wellbeing, avg_support],
                'Status': [
                    'Good' if avg_stress >= 3.5 else 'Needs Attention',
                    'Good' if avg_anxiety >= 3.5 else 'Needs Attention',
                    'Good' if avg_coping >= 3.5 else 'Needs Attention',
                    'Good' if avg_wellbeing >= 3.5 else 'Needs Attention',
                    'Good' if avg_support >= 3.5 else 'Needs Attention'
                ]
            })
            st.dataframe(categories_df, use_container_width=True)
            
            # Clinical Risk displayed separately
            st.subheader("Clinical Assessment")
            clinical_df = pd.DataFrame({
                'Assessment': ['Clinical Risk Indicator (Depression Screening)'],
                'Score': [clinical_risk_score],
                'Status': ['Good' if clinical_risk_score >= 3.5 else 'Needs Clinical Attention']
            })
            st.dataframe(clinical_df, use_container_width=True)
            
            # Enhanced 5-Dimension Radar Chart (balanced without clinical risk)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[avg_stress, avg_anxiety, avg_coping, avg_wellbeing, avg_support],
                theta=['Stress Response', 'Anxiety Management', 'Coping Skills', 'Overall Wellbeing', 'Support Systems'],
                fill='toself',
                name='Your Scores',
                line_color='rgb(32, 146, 230)',
                fillcolor='rgba(32, 146, 230, 0.25)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 5],
                        tickfont=dict(size=10),
                        gridcolor='lightgray'
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11)
                    )
                ),
                showlegend=True,
                title={
                    'text': "Mental Health Assessment Radar Chart - Core Dimensions",
                    'x': 0.5,
                    'font': {'size': 16}
                },
                height=600
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Additional recommendations based on specific areas
            st.subheader("Personalized Recommendations")
            if avg_stress < 3.5:
                st.write("🔹 **Stress Management**: Consider stress reduction techniques like deep breathing, meditation, or physical exercise.")
            if avg_anxiety < 3.5:
                st.write("🔹 **Anxiety Support**: Practice mindfulness techniques and consider talking to a counselor about anxiety management strategies.")
            if avg_coping < 3.5:
                st.write("🔹 **Coping Skills**: Develop a toolkit of healthy coping mechanisms such as journaling, physical activity, or creative outlets.")
            if avg_wellbeing < 3.5:
                st.write("🔹 **Overall Wellbeing**: Focus on work-life balance, maintain social connections, and engage in activities you enjoy.")
            if avg_support < 3.5:
                st.write("🔹 **Support Systems**: Strengthen relationships with colleagues, friends, and family. Don't hesitate to reach out for help.")
            if clinical_risk_score < 3.5:
                st.write("🔹 **Clinical Attention**: PHQ-9 indicators suggest potential depression symptoms. Consider professional mental health evaluation.")
            
            # Show breakdown of PHQ-9 responses
            if depression_score > 0:
                st.subheader("Depression Screening Breakdown")
                phq9_breakdown = pd.DataFrame({
                    'Symptom': ['Interest/Pleasure', 'Feeling Down', 'Sleep Issues', 'Energy/Fatigue', 
                               'Appetite Changes', 'Self-Worth', 'Concentration', 'Psychomotor'],
                    'Response': [little_interest, feeling_down, sleep_trouble, tired_energy, 
                               appetite_changes, feeling_failure, concentration, slow_restless],
                    'Score': [response_to_score.get(item, 0) for item in depression_items]
                })
                st.dataframe(phq9_breakdown, use_container_width=True)
        
        elif submitted and not consent:
            st.error("Please provide consent to participate in the assessment.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Defense Personnel Mental Health Analysis | Developed for DRDO-SSPL | 2025
</div>
""", unsafe_allow_html=True)
