import os
import logging

import joblib
import pandas as pd
import numpy as np
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, '..', 'assets')

try:
    model = joblib.load(os.path.join(ASSETS_DIR, 'rfc.pkl'))
    feature_columns = joblib.load(os.path.join(ASSETS_DIR, 'feature_columns.pkl'))
    age_scaler = joblib.load(os.path.join(ASSETS_DIR, 'age_scaler.pkl'))
    neighborhood_encoder = joblib.load(os.path.join(ASSETS_DIR, 'neighborhood_encoder.pkl'))
    neighborhood_mapping = joblib.load(os.path.join(ASSETS_DIR, 'neighborhood_mapping.pkl'))
    day_difference_scaler = joblib.load(os.path.join(ASSETS_DIR, 'day_difference_scaler.pkl'))
    previous_no_shows_scaler = joblib.load(os.path.join(ASSETS_DIR, 'previous_no_shows_scaler.pkl'))
    logger.info('All model assets loaded successfully')
except Exception as e:
    logger.error(f'Failed to load model assets: {e}')
    raise RuntimeError(f'Model initialization failed: {e}')

app = FastAPI(
    title='No-Show Prediction',
    version='1.0.0'
)

# In production, set CORS_ORIGIN environment variable to restrict origins
CORS_ORIGIN = os.getenv('CORS_ORIGIN', '*').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGIN,
    allow_credentials=True,
    allow_methods=['POST'],
    allow_headers=['*'],
)

class Input(BaseModel):
    gender: Literal['M', 'F'] = Field(..., description='Patient gender')
    age: int = Field(..., ge=0, le=120, description='Patient age')
    neighborhood: str = Field(..., min_length=1, description='Patient neighborhood (see /model-info for common values)')
    scholarship: int = Field(..., ge=0, le=1, description='Has scholarship (0 = No, 1 = Yes)')
    hypertension: int = Field(..., ge=0, le=1, description='Has hypertension (0 = No, 1 = Yes)')
    diabetes: int = Field(..., ge=0, le=1, description='Has diabetes (0 = No, 1 = Yes)')
    alcoholism: int = Field(..., ge=0, le=1, description='Has alcoholism (0 = No, 1 = Yes)')
    handicap: int = Field(..., ge=0, le=4, description='Number of handicaps')
    scheduled_day: str = Field(..., description='YYYY-MM-DD format')
    appointment_day: str = Field(..., description='YYYY-MM-DD format')
    previous_no_shows: int = Field(..., ge=0, description='Number of previous no-shows')
    sms_received: int = Field(..., ge=0, le=1, description='SMS notification received (0 = No, 1 = Yes)')

    @field_validator('scheduled_day', 'appointment_day')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

@app.get('/model-info', operation_id='', tags=['Endpoints'])
async def model_info():
    '''Information about the loaded model and features.'''
    return {
        'model_type': type(model).__name__,
        'feature_count': len(feature_columns),
        'features': feature_columns,
        'neighborhoods': list(neighborhood_mapping.keys()),
        'valid_months': ['April', 'May', 'June'],
        'valid_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    }

@app.post('/predict', operation_id='', tags=['Endpoints'], status_code=status.HTTP_200_OK)
async def predict(data: Input):
    '''Prediction (Show / No-Show) and no_show_probability (0.0 - 1.0).'''
    try:
        logger.info(f'Processing prediction request for patient with age {data.age}, gender {data.gender}')
        
        scheduled_day = datetime.strptime(data.scheduled_day, '%Y-%m-%d').date()
        appointment_day = datetime.strptime(data.appointment_day, '%Y-%m-%d').date()
        
        # Business rule: appointments cannot be scheduled after the appointment date
        if scheduled_day > appointment_day:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Scheduled day must be before or equal to appointment day'
            )

        day_difference = (appointment_day - scheduled_day).days
        
        appointment_datetime = datetime.strptime(data.appointment_day, '%Y-%m-%d')
        appointment_month = appointment_datetime.strftime('%B').lower()
        appointment_day_of_week = appointment_datetime.strftime('%A').lower()
        
        # Business constraints: model only trained on April-June, Monday-Friday appointments
        valid_months = ['april', 'may', 'june']
        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        
        if appointment_month not in valid_months:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid appointment month. Must be one of: {valid_months}'
            )
        if appointment_day_of_week not in valid_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Invalid appointment day. Must be one of: {valid_days}'
            )

        # Apply log transformation before scaling to match training preprocessing
        log_age = np.log1p(data.age)
        log_previous_no_shows = np.log1p(data.previous_no_shows)
        log_day_difference = np.log1p(day_difference)

        input_data = {
            'female': 1 if data.gender == 'F' else 0,
            'male': 1 if data.gender == 'M' else 0,
            'age': age_scaler.transform(pd.DataFrame({'age': [log_age]}))[0][0],
            'scholarship': data.scholarship,
            'hypertension': data.hypertension,
            'diabetes': data.diabetes,
            'alcoholism': data.alcoholism,
            'handicap': data.handicap,
            'previous_no_shows': previous_no_shows_scaler.transform(
                pd.DataFrame({'previous_no_shows': [log_previous_no_shows]})
            )[0][0],
            'day_difference': day_difference_scaler.transform(
                pd.DataFrame({'day_difference': [log_day_difference]})
            )[0][0],
            'sms_received': data.sms_received
        }

        for month in valid_months:
            input_data[month] = 1 if appointment_month == month else 0

        for day in valid_days:
            input_data[day] = 1 if appointment_day_of_week == day else 0

        # Handle unknown neighborhoods by mapping to 'Other' category (same as training preprocessing)
        mapped_neighborhood = neighborhood_mapping.get(data.neighborhood, 'Other')
        encoded_neighborhood = neighborhood_encoder.transform(
            pd.DataFrame({'neighborhood': [mapped_neighborhood]})
        )
        encoded_neighborhood.columns = [f'neighborhood_{i}' for i in range(encoded_neighborhood.shape[1])]
        input_data.update(encoded_neighborhood.iloc[0].to_dict())

        input_df = pd.DataFrame([input_data], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            no_show_probability = float(prediction_proba[1])
        except AttributeError:
            no_show_probability = None

        logger.info(f'Prediction completed successfully. Result: {prediction}')
        
        return {
            'prediction': 'No-Show' if prediction == 1 else 'Show',
            'no_show_probability': no_show_probability
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Unexpected error during prediction: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred during prediction'
        )

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes
    )
    
    # Remove validation error schemas and their references
    for schema in ['HTTPValidationError', 'ValidationError']:
        openapi_schema['components']['schemas'].pop(schema, None)

    for path in openapi_schema.get('paths', {}).values():
        for method in path.values():
            if isinstance(method, dict):
                if 'responses' in method and '422' in method['responses']:
                    method['responses'].pop('422', None)
                if 'summary' in method:
                    method.pop('summary', None)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(
        'main:app',
        host='0.0.0.0',
        port=port,
        log_level='info'
    )