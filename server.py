import pandas as pd
import os
import joblib
from dotenv import load_dotenv # Import this library
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from groq import Groq
from classify import classify_logs

# Load environment variables from the .env file
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize models at the global scope
try:
    log_classifier = joblib.load(os.path.join("models", "log_classifier.joblib"))
except FileNotFoundError:
    raise FileNotFoundError("The model file 'models/log_classifier.joblib' was not found. Please ensure the path is correct.")

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

@app.post("/classify/")
async def classify_logs_endpoint(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    
    try:
        df = pd.read_csv(file.file)
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        # Pass the pre-loaded models and the groq_client to the classification logic
        df["target_label"] = classify_logs(df, model_embedding, log_classifier, groq_client)
        
        print("Dataframe:", df.to_dict())

        output_file = "output.csv"
        df.to_csv(output_file, index=False)
        print("File saved to output.csv")
        return FileResponse(output_file, media_type='text/csv')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()