from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import joblib
import os
from models import db, User, SoilData
from datetime import datetime
from sqlalchemy import select
from flask_sock import Sock
import json
import random  # For demo purposes, replace with actual sensor readings
import time
import qrcode
from io import BytesIO
import base64
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agroflow.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
sock = Sock(app)

# Store active WebSocket connections
active_connections = set()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def get_user_model_path(user_id):
    return os.path.join('models', f'user_{user_id}_model.pkl')

def get_user_scaler_path(user_id):
    return os.path.join('models', f'user_{user_id}_scaler.pkl')

def ensure_user_model_exists(user):
    """Ensure user has a trained model, if not, create one with default data"""
    model_path = get_user_model_path(user.id)
    scaler_path = get_user_scaler_path(user.id)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Generate default training data
            from create_dataset import create_synthetic_dataset
            df = create_synthetic_dataset(n_samples=1000)
            
            # Train and save model
            from train_model import train_and_save_model
            success = train_and_save_model(df, model_path, scaler_path)
            
            if not success:
                print("Failed to create initial model")
                return False
                
            print(f"Created initial model for user {user.id}")
            return True
            
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            return False
    return True

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        # Check if username exists
        stmt = select(User).where(User.username == username)
        if db.session.execute(stmt).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        # Check if email exists
        stmt = select(User).where(User.email == email)
        if db.session.execute(stmt).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            
            # Create user's model
            if not ensure_user_model_exists(user):
                flash('Error creating model. Please try again.', 'warning')
                return redirect(url_for('register'))
            
            login_user(user)
            flash('Registration successful!', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        stmt = select(User).where(User.email == email)
        user = db.session.execute(stmt).scalar_one_or_none()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Successfully logged in!', 'success')
            return redirect(url_for('home'))
        
        flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get values from the form
        plant = request.form['plant']
        ph = float(request.form['ph'])
        tds = float(request.form['tds'])
        turbidity = float(request.form['turbidity'])
        notes = request.form.get('notes', '')

        # Validate input ranges
        if not (0 <= ph <= 14):
            flash("pH value must be between 0 and 14")
            return redirect(url_for('home'))
        if tds < 0:
            flash("TDS value cannot be negative")
            return redirect(url_for('home'))
        if turbidity < 0:
            flash("Turbidity value cannot be negative")
            return redirect(url_for('home'))

        # Ensure user has a model
        if not ensure_user_model_exists(current_user):
            flash("Error loading model. Please try again.")
            return redirect(url_for('home'))

        # Load user's model and scaler
        model_path = get_user_model_path(current_user.id)
        scaler_path = get_user_scaler_path(current_user.id)
        
        try:
            # Load models
            with open(model_path, 'rb') as f:
                models = joblib.load(f)
            fertilizer_model = models['fertilizer_model']
            supplements_model = models['supplements_model']
            ph_adjustment_model = models['ph_adjustment_model']
            dosage_model = models['dosage_model']
            plant_encoder = models['plant_encoder']
            
            # Load scaler
            scaler = joblib.load(scaler_path)
            
            # Create DataFrame with feature names
            import pandas as pd
            features = pd.DataFrame([[plant, ph, tds, turbidity]], 
                                 columns=['Plant', 'pH', 'TDS', 'Turbidity'])
            
            # Encode plant name
            features['Plant'] = plant_encoder.transform([plant])[0]
            
            # Scale the features
            features_scaled = scaler.transform(features)
            
            # Make predictions
            fertilizer_prediction = fertilizer_model.predict(features_scaled)[0]
            supplements_prediction = supplements_model.predict(features_scaled)[0]
            ph_adjustment_prediction = ph_adjustment_model.predict(features_scaled)[0]
            dosage_value = dosage_model.predict(features_scaled)[0]
            
            # Format dosage prediction
            dosage_prediction = f"{dosage_value:.1f} g/L"
            
            print(f"Prediction made for {plant}:")
            print(f"Fertilizer: {fertilizer_prediction}")
            print(f"Supplements: {supplements_prediction}")
            print(f"pH Adjustment: {ph_adjustment_prediction}")
            print(f"Dosage: {dosage_prediction}")
            
        except Exception as e:
            print(f"Error during model loading/prediction: {str(e)}")
            flash("Error making prediction. Please try again.")
            return redirect(url_for('home'))
        
        # Save the data point
        soil_data = SoilData(
            user_id=current_user.id,
            plant=plant,
            ph=ph,
            tds=tds,
            turbidity=turbidity,
            fertilizer=fertilizer_prediction,
            supplements=supplements_prediction,
            ph_adjustment=ph_adjustment_prediction,
            dosage=dosage_value,
            notes=notes
        )
        db.session.add(soil_data)
        db.session.commit()
        
        # Get user's soil data
        stmt = select(SoilData).where(SoilData.user_id == current_user.id).order_by(SoilData.created_at.desc())
        soil_data_list = db.session.execute(stmt).scalars().all()
        
        return render_template('index.html', 
                            plant=plant,
                            result=fertilizer_prediction,
                            supplements=supplements_prediction,
                            ph_adjustment=ph_adjustment_prediction,
                            dosage=dosage_prediction,
                            ph=ph,
                            tds=tds,
                            turb=turbidity,
                            soil_data=soil_data_list)

    except ValueError as e:
        flash("Please enter valid numeric values")
        return redirect(url_for('home'))
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        flash("An error occurred. Please try again.")
        return redirect(url_for('home'))

@app.route('/retrain', methods=['POST'])
@login_required
def retrain():
    try:
        # Get all user's data
        stmt = select(SoilData).where(SoilData.user_id == current_user.id)
        soil_data = db.session.execute(stmt).scalars().all()
        
        if len(soil_data) < 10:
            flash('Need at least 10 data points to retrain the model')
            return redirect(url_for('home'))
        
        # Convert to DataFrame
        import pandas as pd
        data = {
            'pH': [d.ph for d in soil_data],
            'TDS': [d.tds for d in soil_data],
            'Turbidity': [d.turbidity for d in soil_data],
            'Fertilizer': [d.fertilizer for d in soil_data]
        }
        df = pd.DataFrame(data)
        
        # Train new model
        from train_model import train_and_save_model
        model_path = get_user_model_path(current_user.id)
        scaler_path = get_user_scaler_path(current_user.id)
        
        if train_and_save_model(df, model_path, scaler_path):
            flash('Model retrained successfully!')
        else:
            flash('Error retraining model')
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"Error during retraining: {str(e)}")
        flash('Error retraining model')
        return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's soil data
    stmt = select(SoilData).where(SoilData.user_id == current_user.id).order_by(SoilData.created_at.desc())
    soil_data = db.session.execute(stmt).scalars().all()
    return render_template('dashboard.html', soil_data=soil_data)

@app.route('/generate_summary', methods=['POST'])
@login_required
def generate_summary():
    try:
        # Get user's soil data
        stmt = select(SoilData).where(SoilData.user_id == current_user.id).order_by(SoilData.created_at.desc())
        soil_data = db.session.execute(stmt).scalars().all()
        
        if not soil_data:
            return jsonify({'summary': '<div class="alert alert-info">No data available for analysis.</div>'})
        
        # Calculate statistics
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame([{
            'plant': d.plant,
            'ph': d.ph,
            'tds': d.tds,
            'turbidity': d.turbidity,
            'fertilizer': d.fertilizer,
            'supplements': d.supplements,
            'ph_adjustment': d.ph_adjustment,
            'dosage': d.dosage
        } for d in soil_data])
        
        # Generate summary
        summary_parts = []
        
        # Plant distribution
        plant_counts = df['plant'].value_counts()
        most_common_plant = plant_counts.index[0]
        summary_parts.append(f"<h6>Plant Analysis</h6>")
        summary_parts.append(f"<p>You've grown {len(plant_counts)} different types of plants. Your most common plant is <strong>{most_common_plant}</strong> ({plant_counts[most_common_plant]} entries).</p>")
        
        # pH Analysis
        avg_ph = df['ph'].mean()
        ph_std = df['ph'].std()
        summary_parts.append(f"<h6>pH Analysis</h6>")
        summary_parts.append(f"<p>Average pH: <strong>{avg_ph:.2f}</strong> (±{ph_std:.2f})</p>")
        if avg_ph < 5.5:
            summary_parts.append("<p class='text-warning'>Your average pH is slightly low. Consider using pH Up more frequently.</p>")
        elif avg_ph > 6.5:
            summary_parts.append("<p class='text-warning'>Your average pH is slightly high. Consider using pH Down more frequently.</p>")
        
        # TDS Analysis
        avg_tds = df['tds'].mean()
        tds_std = df['tds'].std()
        summary_parts.append(f"<h6>Nutrient Analysis</h6>")
        summary_parts.append(f"<p>Average TDS: <strong>{avg_tds:.0f} ppm</strong> (±{tds_std:.0f})</p>")
        
        # Fertilizer Analysis
        fertilizer_counts = df['fertilizer'].value_counts()
        most_common_fertilizer = fertilizer_counts.index[0]
        summary_parts.append(f"<h6>Fertilizer Usage</h6>")
        summary_parts.append(f"<p>Most used fertilizer: <strong>{most_common_fertilizer}</strong> ({fertilizer_counts[most_common_fertilizer]} times)</p>")
        
        # Supplements Analysis
        supplement_counts = df['supplements'].value_counts()
        most_common_supplement = supplement_counts.index[0]
        summary_parts.append(f"<h6>Supplement Usage</h6>")
        summary_parts.append(f"<p>Most used supplement combination: <strong>{most_common_supplement}</strong></p>")
        
        # Dosage Analysis
        avg_dosage = df['dosage'].mean()
        dosage_std = df['dosage'].std()
        summary_parts.append(f"<h6>Dosage Analysis</h6>")
        summary_parts.append(f"<p>Average dosage: <strong>{avg_dosage:.2f} g/L</strong> (±{dosage_std:.2f})</p>")
        
        # Recommendations
        summary_parts.append(f"<h6>Recommendations</h6>")
        if len(soil_data) < 5:
            summary_parts.append("<p>Continue collecting data to get more detailed insights.</p>")
        else:
            if avg_ph < 5.5:
                summary_parts.append("<p>• Consider increasing your pH levels slightly for better nutrient absorption.</p>")
            elif avg_ph > 6.5:
                summary_parts.append("<p>• Consider decreasing your pH levels slightly for better nutrient absorption.</p>")
            
            if avg_tds < 800:
                summary_parts.append("<p>• Your nutrient levels are on the lower side. Consider increasing your fertilizer dosage slightly.</p>")
            elif avg_tds > 2000:
                summary_parts.append("<p>• Your nutrient levels are on the higher side. Consider decreasing your fertilizer dosage slightly.</p>")
        
        # Combine all parts
        summary = "\n".join(summary_parts)
        return jsonify({'summary': summary})
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({'summary': '<div class="alert alert-danger">Error generating summary. Please try again.</div>'})

@app.route('/predict')
@login_required
def predict_page():
    return render_template('index.html')

# Global variable to store latest sensor values
sensor_data = {
    'ph': None,
    'tds': None,
    'turbidity': None
}

# Set of active WebSocket clients
active_connections = set()

@app.route('/sensors')
@login_required
def sensors():
    return render_template('sensors.html')

@app.route('/api/push_sensor_data', methods=['POST'])
def push_sensor_data():
    global sensor_data
    try:
        # Get data from Raspberry Pi
        data = request.json
        
        # Validate and convert sensor data
        sensor_data = {
            'ph': float(data.get('ph', 0)),
            'tds': float(data.get('tds', 0)),
            'turbidity': float(data.get('turbidity', 0))
        }
        
        # Log the received data
        print(f"Received sensor data from Raspberry Pi: {sensor_data}")
        
        # Store in database if needed
        try:
            soil_data = SoilData(
                user_id=current_user.id if current_user.is_authenticated else None,
                plant='Auto',  # You can modify this based on your needs
                ph=sensor_data['ph'],
                tds=sensor_data['tds'],
                turbidity=sensor_data['turbidity'],
                notes='Automated sensor reading'
            )
            db.session.add(soil_data)
            db.session.commit()
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Continue even if database storage fails
        
        return jsonify({'status': 'success', 'data': sensor_data}), 200
    except Exception as e:
        print(f"Error processing sensor data: {str(e)}")
        return jsonify({'error': str(e)}), 400

@sock.route('/ws/sensors')
def sensor_ws(ws):
    active_connections.add(ws)
    try:
        while True:
            # Send current sensor data or default values if None
            data = {
                'ph': sensor_data['ph'] if sensor_data['ph'] is not None else 0,
                'tds': sensor_data['tds'] if sensor_data['tds'] is not None else 0,
                'turbidity': sensor_data['turbidity'] if sensor_data['turbidity'] is not None else 0
            }
            ws.send(json.dumps(data))
            time.sleep(1)  # Update every second
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        active_connections.remove(ws)

def get_ngrok_url():
    try:
        # Try to get the ngrok URL from the ngrok API
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json()["tunnels"]
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
    except:
        # If ngrok is not running or API is not accessible
        return None

@app.route('/qr')
def qr_code():
    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    # Try to get ngrok URL first, fallback to local URL
    ngrok_url = get_ngrok_url()
    if ngrok_url:
        qr.add_data(f"{ngrok_url}/info")
    else:
        qr.add_data(request.url_root + 'info')
    
    qr.make(fit=True)
    
    # Create an image from the QR Code
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64 for embedding in HTML
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return render_template('qr.html', 
                         qr_code_url=f"data:image/png;base64,{img_str}",
                         current_url=ngrok_url if ngrok_url else request.url_root)

@app.route('/info')
def info():
    return render_template('info.html')

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)