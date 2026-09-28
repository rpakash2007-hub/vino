from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User account entity with hashed password security."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationship with diagnosis history
    diagnoses = db.relationship('DiagnosisHistory', backref='patient_user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password: str):
        """Hash and store the given password securely."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify the plain password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'


class DiagnosisHistory(db.Model):
    """Medical scan diagnostic decision-support record."""
    __tablename__ = 'diagnosis_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # File details
    filename = db.Column(db.String(256), nullable=False)
    original_filename = db.Column(db.String(256), nullable=False)
    image_dimensions = db.Column(db.String(64), default='Unknown')
    image_data = db.Column(db.Text, nullable=True)  # Base64 data URI for serverless display persistence
    
    # Ensemble diagnosis outcome
    prediction = db.Column(db.String(32), nullable=False)  # NORMAL, STONE, CYST, TUMOR
    confidence = db.Column(db.Float, nullable=False)        # Overall percentage, e.g. 94.82
    
    # Individual model outcomes
    svm_prediction = db.Column(db.String(32), nullable=False)
    svm_confidence = db.Column(db.Float, nullable=False)
    
    decision_tree_prediction = db.Column(db.String(32), nullable=False)
    decision_tree_confidence = db.Column(db.Float, nullable=False)
    
    is_low_confidence = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<DiagnosisHistory #{self.id} User={self.user_id} Prediction={self.prediction} Conf={self.confidence}%>'
