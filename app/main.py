import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Set up base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and necessary assets
model = joblib.load(os.path.join(BASE_DIR, "..", "assets", "rfc.pkl"))
feature_columns = joblib.load(open(os.path.join(BASE_DIR, "..", "assets", "feature_columns.pkl"), "rb"))
age_scaler = joblib.load(open(os.path.join(BASE_DIR, "..", "assets", "age_scaler.pkl"), "rb"))
neighborhood_encoder = joblib.load(open(os.path.join(BASE_DIR, "..", "assets", "neighborhood_encoder.pkl"), "rb"))
neighborhood_mapping = joblib.load(open(os.path.join(BASE_DIR, "..", "assets", "neighborhood_mapping.pkl"), "rb"))
day_difference_scaler = joblib.load(open(os.path.join(BASE_DIR, "..", "assets", "day_difference_scaler.pkl"), "rb"))
previous_no_shows_scaler = joblib.load(open(os.path.join(BASE_DIR, "..", "assets", "previous_no_shows_scaler.pkl"), "rb"))

# FastAPI app setup
app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    neighborhoods = sorted(neighborhood_mapping.keys())
    return templates.TemplateResponse("index.html", {"request": request, "neighborhoods": neighborhoods})


@app.post("/predict")
def predict(
    gender: str = Form(...),
    age: int = Form(...),
    neighborhood: str = Form(...),
    scholarship: int = Form(...),
    hypertension: int = Form(...),
    diabetes: int = Form(...),
    alcoholism: int = Form(...),
    handicap: int = Form(...),
    scheduled_day: str = Form(...),
    appointment_day: str = Form(...),
    previous_no_shows: int = Form(...),
    sms_received: int = Form(...)
):
    # Validate neighborhood
    if neighborhood not in neighborhood_mapping:
        return JSONResponse(status_code=400, content={"detail": "Invalid neighborhood selected."})

    # Convert dates
    scheduled_day = pd.to_datetime(scheduled_day)
    appointment_day = pd.to_datetime(appointment_day)
    
    # Ensure scheduled date is before or equal to appointment date
    if scheduled_day > appointment_day:
        return JSONResponse(status_code=400, content={"detail": "Scheduled day must be before or equal to appointment day."})

    day_difference = (appointment_day - scheduled_day).days
    
    appointment_month = appointment_day.strftime("%B")
    appointment_day_of_week = appointment_day.strftime("%A")

    valid_months = ['april', 'may', 'june']
    valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']

    if appointment_month.lower() not in valid_months or appointment_day_of_week.lower() not in valid_days:
        return JSONResponse(status_code=400, content={"detail": "Invalid appointment day or month."})

    # Construct input dict with proper scaling and encoding
    input_data = {
        'female': 1 if gender.lower() == 'f' else 0,
        'male': 1 if gender.lower() == 'm' else 0,
        'age': age_scaler.transform(pd.DataFrame({'age': [age]}))[0][0],
        'scholarship': scholarship,
        'hypertension': hypertension,
        'diabetes': diabetes,
        'alcoholism': alcoholism,
        'handicap': handicap,
        'previous_no_shows': previous_no_shows_scaler.transform(pd.DataFrame({'previous_no_shows': [previous_no_shows]}))[0][0],
        'day_difference': day_difference_scaler.transform(pd.DataFrame({'day_difference': [day_difference]}))[0][0],
        'sms_received': sms_received
    }

    # One-hot encode month
    for month in valid_months:
        input_data[month] = 1 if appointment_month.lower() == month else 0

    # One-hot encode day of week
    for day in valid_days:
        input_data[day] = 1 if appointment_day_of_week.lower() == day else 0

    # Encode neighborhood
    encoded_neigh = neighborhood_encoder.transform(
        pd.DataFrame({'neighborhood': [neighborhood_mapping[neighborhood]]})
    )
    encoded_neigh.columns = [f'neighborhood_{i}' for i in range(encoded_neigh.shape[1])]
    input_data.update(encoded_neigh.iloc[0].to_dict())

    # Match to expected feature columns
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Predict
    prediction = model.predict(input_df)[0]
    return JSONResponse(content={"prediction": int(prediction)})