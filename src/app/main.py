""" 
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving.

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation.
- Gradio: User-friendly web UI for manual testing and demonstrations.
- Pydantic: Data validation and automatic API documentation.

"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict # core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title= "Telco Customer Churn Prediction API",
    description= "ML API for predicting customer churn in telcom industry",
    version= "1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """ 
    Health check endpoint for monitoring and load balancer health checks.

    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validatioin and API documentation

class CustomerData(BaseModel):
    """ 
    Customer data schema for churn prediction.

    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.

    """
    # Demographies
    gender: str               # "Male" or "Female"
    Partner: str              # "Yes" or "No"
    Dependents: str           # "Yes", "No" - has partner

    # Phone service
    PhoneService: str         # "Yes", "No"
    MultipleLines: str        # "Yes", "No", or "No phone service"

    # Internet services
    InternetService: str      # "DSL", "Fiber optic", pr "No"
    OnlineSecurity: str       # "Yes", "No", or "No internet service"
    OnlineBackup: str         # "Yes", "No", or "No internet service" 
    DeviceProtection: str     # "Yes", "No", or "No internet service"
    TechSupport: str          # "Yes", "No", or "No internet service"
    StreamingTV: str          # "Yes", "No", or "No internet service"
    StreamingMovies: str      # "Yes", "No", or "No internet service"

    #Account Information
    Contract: str             # "Month-to-Month", "One-year", "Two-years"
    PaperlessBilling: str     # "Yes" or "No"
    PaymentMethod: str        # "Electronic check"

    # Numeric features
    tenure: int               # Number of months with company
    MonthlyCharges: float     # Monthly charges in dollars
    TotalCharges: float       # Total charges to date

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """ 
    Main prediciton endpoint for customer churn prediction.

    This endpoint:
    1. Receives validated customer data via Pydantic.
    2. Calls the inference pipeline to transform features and predict
    3. Returns  churn prediction in JSON format.
    
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction ": result}
    except Exception as e:
        # Return error details for debugging
        return {"error": str(e)}