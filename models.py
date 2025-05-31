from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    soil_data = db.relationship('SoilData', backref='user', lazy=True)
    model_path = db.Column(db.String(200))  # Path to user's trained model
    scaler_path = db.Column(db.String(200))  # Path to user's scaler

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SoilData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plant = db.Column(db.String(50), nullable=False)
    ph = db.Column(db.Float, nullable=False)
    tds = db.Column(db.Float, nullable=False)
    turbidity = db.Column(db.Float, nullable=False)
    fertilizer = db.Column(db.String(100), nullable=False)
    supplements = db.Column(db.String(200), nullable=True)
    ph_adjustment = db.Column(db.String(100), nullable=True)
    dosage = db.Column(db.Float, nullable=True)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'ph': self.ph,
            'tds': self.tds,
            'turbidity': self.turbidity,
            'fertilizer': self.fertilizer,
            'created_at': self.created_at.isoformat(),
            'notes': self.notes
        } 