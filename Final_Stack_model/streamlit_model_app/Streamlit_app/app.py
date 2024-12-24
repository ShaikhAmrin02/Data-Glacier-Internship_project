import streamlit as st 
import pandas as pd
import joblib 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the trained pipeline
pipeline = joblib.load(pipeline = joblib.load("streamlit_model_app/Streamlit_app/preprocessing_pipeline.joblib")

# Load the data file directly
file_path = "D:/DataGlacier/Final Stack model/streamlit_model_app/Streamlit_app/Healthcare_dataset.xlsx"
input_df = pd.read_excel(file_path)

# List of selected features
selected_features = ['Concom_Macrolides_And_Similar_Types', 'Idn_Indicator', 'Risk_Segment_Prior_Ntm',
    'Change_Risk_Segment', 'Comorb_Encounter_For_Screening_For_Malignant_Neoplasms',
    'Concom_Viral_Vaccines', 'Gluco_Record_Prior_Ntm', 'Comorb_Encounter_For_Immunization',
    'Risk_Segment_During_Rx', 'Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx',
    'Region', 'Ntm_Specialist_Flag', 'Tscore_Bucket_During_Rx', 'Comorb_Other_Disorders_Of_Bone_Density_And_Structure',
    'Comorb_Dorsalgia', 'Dexa_Freq_During_Rx', 'Concom_Cephalosporins', 'Comorb_Personal_history_of_malignant_neoplasm',
    'Comorb_Long_Term_Current_Drug_Therapy', 'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx',
    'Gluco_Record_During_Rx', 'Comorb_Vitamin_D_Deficiency', 'Dexa_During_Rx',
    'Comorb_Personal_History_Of_Other_Diseases_And_Conditions', 'Concom_Anaesthetics_General',
    'Comorb_Gastro_esophageal_reflux_disease', 'Count_Of_Risks', 'Concom_Broad_Spectrum_Penicillins',
    'Comorb_Osteoporosis_without_current_pathological_fracture', 'Concom_Fluoroquinolones',
    'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified', 'Ntm_Speciality_Bucket']

# Streamlit UI
st.set_page_config(page_title='Healthcare Persistency Model', layout='wide')
st.title('Healthcare Persistency Model: Predictions')

# Display the uploaded data
st.subheader("Dataset Preview")
st.dataframe(input_df.head())

# Preprocess and predict
X = input_df[selected_features]
predictions = pipeline.predict(X)
prediction_probs = pipeline.predict_proba(X)[:, 1]

# Create DataFrame with predictions and probabilities
prediction_df = pd.DataFrame({'prediction': predictions,
                              'Probability': prediction_probs})
prediction_df = pd.concat([input_df[selected_features], prediction_df], axis=1)

# Display top 10 Persistent ana non-persistent Cases
st.subheader("Top 10 persistent and Non-Persistent Cases")
top_10_persistent = prediction_df.sort_values(by='Probability', ascending=False).head(10)
top_10_non_persistent = prediction_df.sort_values(by='Probability', ascending=True).head(10)

col1, col2 = st.columns(2)

with col1:
    st.write("Top 10 Persistent Cases")
    st.dataframe(top_10_persistent[['Probability'] + selected_features].style.format({'Probability': '{:.2%}'}))

with col2:
    st.write("Top 10 Non-Persistent Cases")
    st.dataframe(top_10_non_persistent[['Probability'] + selected_features].style.format({'Probability': '{:.2%}'}))

# Interactive visualizations
st.subheader("Interactive Visualizations")

    # Create small subplots side by side
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Histogram for Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(prediction_probs, kde=True, ax=ax)
    ax.set_title('Histogram with KDE')
    st.pyplot(fig)

with col2:
    st.write("Scatter Plot for Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(5, 3))
    scatter = ax.scatter(range(len(prediction_probs)), prediction_probs, c=prediction_probs, cmap='viridis', alpha=0.75)
    ax.set_title('Scatter Plot of Prediction Probabilities')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Probability')
    plt.colorbar(scatter)
    st.pyplot(fig)

with col3:
    st.write("Top 10 Prediction Probabilities")
    top_10_indices = np.argsort(prediction_probs)[-10:]
    top_10_probs = prediction_probs[top_10_indices]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(range(10), top_10_probs, color='skyblue')
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_10_indices)
    ax.set_xlabel('Probability')
    ax.set_ylabel('Sample Index')
    ax.set_title('Top 10 Prediction Probabilities')
    st.pyplot(fig)

# Dynamic feature selection and visualization
st.subheader("Feature Impact on Prediction Probabilities")
feature_to_plot = st.selectbox('Select a feature to visualize', selected_features)
feature_values = input_df[feature_to_plot]

# Bar chart visualization for the selected feature
st.write(f"Bar Chart of {feature_to_plot} vs Prediction Probabilities")
fig, ax = plt.subplots(figsize=(5, 3))
feature_probs = pd.DataFrame({'Feature': feature_values, 'Probability': prediction_probs})
feature_probs_mean = feature_probs.groupby('Feature')['Probability'].mean().sort_values(ascending=False)
feature_probs_mean.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title(f'{feature_to_plot} vs Prediction Probabilities')
ax.set_xlabel(feature_to_plot)
ax.set_ylabel('Mean Probability')
st.pyplot(fig)
