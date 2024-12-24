import os

print(f"Current working directory: {os.getcwd()}")
print(f"File exists: {os.path.exists('D:/DataGlacier/Final Stack model/streamlit_model_app/Streamlit_app/preprocessing_pipeline.joblib')}")

from joblib import load
import traceback

try:
    pipeline = load(r"d:/DataGlacier/Final Stack model/streamlit_model_app/Streamlit_app/preprocessing_pipeline.joblib")
    print("Pipeline loaded successfully.")
except ModuleNotFoundError as e:
    print("ModuleNotFoundError:", e)
    print(traceback.format_exc())

from joblib import load

pipeline = load("preprocessing_pipeline.joblib")
print("Pipeline loaded successfully!")
