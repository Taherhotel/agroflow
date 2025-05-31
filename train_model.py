import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import traceback

def load_and_preprocess_data(df):
    """Load and preprocess the dataset"""
    try:
        print("Starting data preprocessing...")
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Prepare features
        X = df[['Plant', 'pH', 'TDS', 'Turbidity']]
        y_fertilizer = df['Fertilizer']
        y_supplements = df['Supplements']
        y_ph_adjustment = df['pH_Adjustment']
        
        # Convert plant names to numeric values
        print("Encoding plant names...")
        plant_encoder = LabelEncoder()
        X['Plant'] = plant_encoder.fit_transform(X['Plant'])
        
        # Convert dosage to float
        print("Converting dosage values...")
        y_dosage = pd.to_numeric(df['Dosage'], errors='coerce')
        if y_dosage.isnull().any():
            print("Warning: Some dosage values could not be converted to numeric")
            y_dosage = y_dosage.fillna(y_dosage.mean())
        
        print(f"Feature ranges:")
        print(f"pH: {X['pH'].min():.2f} - {X['pH'].max():.2f}")
        print(f"TDS: {X['TDS'].min():.2f} - {X['TDS'].max():.2f}")
        print(f"Turbidity: {X['Turbidity'].min():.2f} - {X['Turbidity'].max():.2f}")
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print("Data preprocessing completed successfully")
        return X_scaled, y_fertilizer, y_supplements, y_ph_adjustment, y_dosage, scaler, plant_encoder
        
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

def train_models(X, y_fertilizer, y_supplements, y_ph_adjustment, y_dosage):
    """Train all models"""
    try:
        print("\nTraining fertilizer classifier...")
        fertilizer_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        fertilizer_model.fit(X, y_fertilizer)
        print("Fertilizer classifier trained successfully")
        
        print("\nTraining supplements classifier...")
        supplements_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        supplements_model.fit(X, y_supplements)
        print("Supplements classifier trained successfully")
        
        print("\nTraining pH adjustment classifier...")
        ph_adjustment_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        ph_adjustment_model.fit(X, y_ph_adjustment)
        print("pH adjustment classifier trained successfully")
        
        print("\nTraining dosage regressor...")
        dosage_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        dosage_model.fit(X, y_dosage)
        print("Dosage regressor trained successfully")
        
        return fertilizer_model, supplements_model, ph_adjustment_model, dosage_model
        
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

def train_and_save_model(df, model_path, scaler_path):
    """Train and save all models and scaler"""
    try:
        print("\nStarting model training process...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Model directory created/verified: {os.path.dirname(model_path)}")
        
        # Load and preprocess data
        X, y_fertilizer, y_supplements, y_ph_adjustment, y_dosage, scaler, plant_encoder = load_and_preprocess_data(df)
        
        # Train models
        fertilizer_model, supplements_model, ph_adjustment_model, dosage_model = train_models(
            X, y_fertilizer, y_supplements, y_ph_adjustment, y_dosage
        )
        
        # Save models and encoders
        print("\nSaving models...")
        models = {
            'fertilizer_model': fertilizer_model,
            'supplements_model': supplements_model,
            'ph_adjustment_model': ph_adjustment_model,
            'dosage_model': dosage_model,
            'plant_encoder': plant_encoder
        }
        with open(model_path, 'wb') as f:
            joblib.dump(models, f)
        print(f"Models saved to: {model_path}")
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        print("\nModel training and saving completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError in train_and_save_model: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

def main():
    try:
        print("Loading dataset...")
        df = pd.read_csv('fertilizer_data.csv')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Train and save model
        model_path = 'models/model.pkl'
        scaler_path = 'models/scaler.pkl'
        
        if train_and_save_model(df, model_path, scaler_path):
            print("\nModel training process completed successfully!")
        else:
            print("\nModel training process failed!")
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 