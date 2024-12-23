import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load raw data
df = pd.read_excel("D:\DataGlacier\Week 12 Final Stack model\Healthcare_dataset.xlsx") 

# List of selected features
selected_features = ['Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx', 'Comorb_Encounter_For_Screening_For_Malignant_Neoplasms',
                       'Comorb_Dorsalgia', 'Concom_Viral_Vaccines', 'Comorb_Osteoporosis_without_current_pathological_fracture',
                       'Risk_Segment_During_Rx', 'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified', 'Risk_Segment_Prior_Ntm', 
                       'Comorb_Vitamin_D_Deficiency', 'Concom_Anaesthetics_General', 'Comorb_Encounter_For_Immunization', 
                       'Concom_Broad_Spectrum_Penicillins', 'Comorb_Encntr_For_Oth_Sp_Exam_W_O_Complaint_Suspected_Or_Reprtd_Dx', 
                       'Comorb_Personal_history_of_malignant_neoplasm', 'Dexa_During_Rx', 'Idn_Indicator', 'Concom_Fluoroquinolones', 
                       'Dexa_Freq_During_Rx', 'Gluco_Record_During_Rx', 'Comorb_Long_Term_Current_Drug_Therapy', 'Gluco_Record_Prior_Ntm', 
                       'Comorb_Other_Disorders_Of_Bone_Density_And_Structure', 'Change_Risk_Segment', 'Concom_Cephalosporins', 
                       'Comorb_Personal_History_Of_Other_Diseases_And_Conditions', 'Count_Of_Risks', 'Region', 'Tscore_Bucket_During_Rx', 
                       'Comorb_Gastro_esophageal_reflux_disease', 'Ntm_Specialist_Flag', 'Concom_Macrolides_And_Similar_Types', 'Ntm_Speciality_Bucket']

X = df[selected_features]
y = df['Persistency_Flag'] 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), X.select_dtypes(include=['object']).columns)])

# Stacking model
stack_model = StackingClassifier(
    estimators=[('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(max_depth=5)),
        ('rf', RandomForestClassifier(n_estimators=100))],
    final_estimator=RandomForestClassifier(n_estimators=100))

# Full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),('stack_model', stack_model)])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(pipeline, 'preprocessing_pipeline.joblib')
print("Model and pipeline saved successfully.")