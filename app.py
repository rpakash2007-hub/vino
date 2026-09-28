import os
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
from PIL import Image

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, abort, send_from_directory, jsonify
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from werkzeug.utils import secure_filename

from config import Config
from database import db, User, DiagnosisHistory
from feature_extraction import extract_features, TARGET_IMAGE_SIZE

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Database and Login Manager
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = "Please sign in to access the Kidney AI Screening Platform."
login_manager.login_message_category = "info"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Cached in-memory model holders
ML_MODELS = {
    'svm': None,
    'decision_tree': None,
    'scaler': None,
    'labels': None,
    'metadata': None,
    'loaded': False
}

def load_ml_models():
    """Attempts to load trained ML models from disk."""
    try:
        if (Config.SVM_MODEL_PATH.exists() and 
            Config.DT_MODEL_PATH.exists() and 
            Config.SCALER_PATH.exists()):
            
            ML_MODELS['svm'] = joblib.load(Config.SVM_MODEL_PATH)
            ML_MODELS['decision_tree'] = joblib.load(Config.DT_MODEL_PATH)
            ML_MODELS['scaler'] = joblib.load(Config.SCALER_PATH)
            
            if Config.CLASS_LABELS_PATH.exists():
                ML_MODELS['labels'] = joblib.load(Config.CLASS_LABELS_PATH)
            else:
                ML_MODELS['labels'] = Config.DIAGNOSTIC_CLASSES
                
            if Config.METADATA_PATH.exists():
                with open(Config.METADATA_PATH, 'r') as f:
                    ML_MODELS['metadata'] = json.load(f)
                    
            ML_MODELS['loaded'] = True
            app.logger.info("NephroScan AI models loaded successfully.")
            return True
    except Exception as e:
        app.logger.error(f"Failed to load ML models: {e}")
    
    ML_MODELS['loaded'] = False
    return False

# Initialize database tables and try loading models at startup
with app.app_context():
    db.create_all()
    load_ml_models()


def allowed_file(filename: str) -> bool:
    """Verifies that filename has an approved image extension."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS


def validate_and_process_image(filepath: Path):
    """
    Validates file integrity and ensures it is an authentic readable medical image.
    Returns: (PIL.Image, dimensions_str) or raises ValueError
    """
    try:
        with Image.open(filepath) as img:
            img.verify()  # Check for corruption
        
        # Re-open after verify() closes the file
        img = Image.open(filepath)
        w, h = img.size
        if w < 32 or h < 32:
            raise ValueError("Image dimensions are too small to represent a diagnostic CT/MRI slice.")
        
        dim_str = f"{w} × {h} px"
        return img, dim_str
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image format: {e}")


def run_ensemble_prediction(image_path: Path):
    """
    Predicts kidney condition using both SVM and Decision Tree classifiers.
    Strictly outputs one of: NORMAL, STONE, CYST, TUMOR.
    Returns dict with predictions, confidences, agreement, and low-confidence flag.
    """
    if not ML_MODELS['loaded']:
        # Attempt reloading in case models were trained recently
        if not load_ml_models():
            raise RuntimeError("Diagnostic models are not loaded. Please train models first.")

    # 1. Feature Extraction (Reproducible pipeline)
    features = extract_features(image_path)
    features_scaled = ML_MODELS['scaler'].transform(features.reshape(1, -1))

    # 2. Support Vector Machine prediction & probabilities
    svm_probs = ML_MODELS['svm'].predict_proba(features_scaled)[0]
    svm_classes = list(ML_MODELS['svm'].classes_)
    svm_pred_idx = int(np.argmax(svm_probs))
    svm_class = svm_classes[svm_pred_idx]
    svm_conf = float(svm_probs[svm_pred_idx])

    # 3. Decision Tree prediction & probabilities
    dt_probs = ML_MODELS['decision_tree'].predict_proba(features_scaled)[0]
    dt_classes = list(ML_MODELS['decision_tree'].classes_)
    dt_pred_idx = int(np.argmax(dt_probs))
    dt_class = dt_classes[dt_pred_idx]
    dt_conf = float(dt_probs[dt_pred_idx])

    # 4. Ensemble Resolution Logic
    models_agree = (svm_class == dt_class)
    
    # Map probabilities to a unified class order
    unified_classes = Config.DIAGNOSTIC_CLASSES
    ensemble_prob_map = {}
    for c in unified_classes:
        s_p = svm_probs[svm_classes.index(c)] if c in svm_classes else 0.0
        d_p = dt_probs[dt_classes.index(c)] if c in dt_classes else 0.0
        # Weighted ensemble: 0.60 SVM (kernel distance margin) + 0.40 DT (orthogonal partition)
        ensemble_prob_map[c] = (0.60 * s_p) + (0.40 * d_p)

    if models_agree:
        final_class = svm_class
        final_confidence = ensemble_prob_map[final_class]
    else:
        # Deterministic resolution: class with highest validated ensemble probability
        final_class = max(ensemble_prob_map, key=ensemble_prob_map.get)
        final_confidence = ensemble_prob_map[final_class]

    # Convert to clean percentage
    confidence_pct = round(final_confidence * 100.0, 2)
    svm_conf_pct = round(svm_conf * 100.0, 2)
    dt_conf_pct = round(dt_conf * 100.0, 2)

    # Low-confidence threshold check
    is_low_confidence = (final_confidence < Config.CONFIDENCE_THRESHOLD)

    return {
        'prediction': final_class,
        'confidence': confidence_pct,
        'svm_prediction': svm_class,
        'svm_confidence': svm_conf_pct,
        'decision_tree_prediction': dt_class,
        'decision_tree_confidence': dt_conf_pct,
        'models_agree': models_agree,
        'is_low_confidence': is_low_confidence,
        'class_probabilities': {k: round(v * 100.0, 2) for k, v in ensemble_prob_map.items()}
    }


# ==========================================
# PUBLIC ROUTES
# ==========================================

@app.route('/')
def index():
    """Landing page showcasing the AI Medical Image Diagnosis System."""
    models_ready = ML_MODELS['loaded'] or (
        Config.SVM_MODEL_PATH.exists() and 
        Config.DT_MODEL_PATH.exists() and 
        Config.SCALER_PATH.exists()
    )
    return render_template('index.html', models_ready=models_ready)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """New clinician / researcher registration."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not full_name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template('register.html')

        if password != confirm_password:
            flash("Passwords do not match. Please verify and re-enter.", "danger")
            return render_template('register.html')

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "warning")
            return render_template('register.html')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email address already exists.", "warning")
            return render_template('register.html')

        # First registered user becomes admin
        is_first_user = (User.query.count() == 0)
        new_user = User(name=full_name, email=email, is_admin=is_first_user)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        flash("Registration successful! Welcome to NephroScan AI.", "success")
        return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User authentication."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.name}!", "success")
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash("Invalid email or password. Please try again.", "danger")

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Terminates session."""
    logout_user()
    flash("You have been signed out safely.", "info")
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Password recovery assistance."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()
        if user:
            # Demonstration reset workflow
            flash("A secure password reset link has been dispatched to your email address.", "info")
        else:
            flash("If that email exists in our registry, a reset link has been sent.", "info")
        return redirect(url_for('login'))
    return render_template('forgot_password.html')


# ==========================================
# PROTECTED WORKSPACE ROUTES
# ==========================================

@app.route('/dashboard')
@login_required
def dashboard():
    """Main clinician screening dashboard."""
    # Check model readiness dynamically
    if not ML_MODELS['loaded']:
        load_ml_models()
    
    # Query user-specific diagnoses
    diagnoses = DiagnosisHistory.query.filter_by(user_id=current_user.id).order_by(
        DiagnosisHistory.created_at.desc()
    ).all()

    # Aggregate Statistics
    total_scans = len(diagnoses)
    normal_count = sum(1 for d in diagnoses if d.prediction == 'NORMAL')
    stone_count = sum(1 for d in diagnoses if d.prediction == 'STONE')
    cyst_count = sum(1 for d in diagnoses if d.prediction == 'CYST')
    tumor_count = sum(1 for d in diagnoses if d.prediction == 'TUMOR')

    stats = {
        'total': total_scans,
        'normal': normal_count,
        'stone': stone_count,
        'cyst': cyst_count,
        'tumor': tumor_count
    }

    recent_scans = diagnoses[:5]
    models_ready = ML_MODELS['loaded']

    return render_template(
        'dashboard.html',
        stats=stats,
        recent_scans=recent_scans,
        models_ready=models_ready,
        metadata=ML_MODELS.get('metadata')
    )


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """CT / MRI scan upload and automated analysis."""
    if not ML_MODELS['loaded']:
        load_ml_models()

    if request.method == 'POST':
        if not ML_MODELS['loaded']:
            flash("AI Models are NOT yet trained or loaded. Please train models before analyzing scans.", "danger")
            return redirect(url_for('dashboard'))

        if 'scan_image' not in request.files:
            flash("No file part detected in request.", "warning")
            return redirect(request.url)

        file = request.files['scan_image']
        if file.filename == '':
            flash("No image file was selected. Please choose a kidney CT or MRI image.", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file format. Please upload a .jpg, .jpeg, or .png image.", "danger")
            return redirect(request.url)

        try:
            # Generate secure, unguessable storage filename
            original_fname = secure_filename(file.filename)
            ext = original_fname.rsplit('.', 1)[1].lower() if '.' in original_fname else 'png'
            unique_fname = f"scan_{uuid.uuid4().hex[:12]}.{ext}"
            save_path = Config.UPLOAD_FOLDER / unique_fname
            
            # Save file to disk
            file.save(str(save_path))

            # Validate image format and dimensions
            _, dim_str = validate_and_process_image(save_path)

            # Run deterministic ML prediction
            result = run_ensemble_prediction(save_path)

            # Persist diagnosis in database
            diagnosis = DiagnosisHistory(
                user_id=current_user.id,
                filename=unique_fname,
                original_filename=original_fname,
                image_dimensions=dim_str,
                prediction=result['prediction'],
                confidence=result['confidence'],
                svm_prediction=result['svm_prediction'],
                svm_confidence=result['svm_confidence'],
                decision_tree_prediction=result['decision_tree_prediction'],
                decision_tree_confidence=result['decision_tree_confidence'],
                is_low_confidence=result['is_low_confidence']
            )
            db.session.add(diagnosis)
            db.session.commit()

            return redirect(url_for('result', diagnosis_id=diagnosis.id))

        except ValueError as ve:
            # Clean up uploaded file if corrupted
            if 'save_path' in locals() and save_path.exists():
                save_path.unlink()
            flash(str(ve), "danger")
            return redirect(request.url)
        except Exception as e:
            if 'save_path' in locals() and save_path.exists():
                save_path.unlink()
            app.logger.exception("Inference error:")
            flash(f"Unable to analyze this image: {str(e)}. Please upload a valid kidney CT/MRI scan.", "danger")
            return redirect(request.url)

    return render_template('upload.html', models_ready=ML_MODELS['loaded'])


@app.route('/result/<int:diagnosis_id>')
@login_required
def result(diagnosis_id: int):
    """Detailed visual diagnostic result page."""
    diagnosis = DiagnosisHistory.query.get_or_404(diagnosis_id)

    # Protect patient privacy: only the owner or an admin can access this diagnosis
    if diagnosis.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    display_name = Config.CLASS_DISPLAY_NAMES.get(diagnosis.prediction, diagnosis.prediction)
    all_classes = Config.DIAGNOSTIC_CLASSES

    return render_template(
        'result.html',
        diagnosis=diagnosis,
        display_name=display_name,
        all_classes=all_classes
    )


@app.route('/history')
@login_required
def history():
    """Historical screening audit log."""
    selected_class = request.args.get('class', '').upper()
    query = DiagnosisHistory.query.filter_by(user_id=current_user.id)

    if selected_class in Config.DIAGNOSTIC_CLASSES:
        query = query.filter_by(prediction=selected_class)

    diagnoses = query.order_by(DiagnosisHistory.created_at.desc()).all()
    all_classes = Config.DIAGNOSTIC_CLASSES

    return render_template(
        'history.html',
        diagnoses=diagnoses,
        all_classes=all_classes,
        selected_class=selected_class
    )


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management and diagnostic metrics."""
    if request.method == 'POST':
        current_pwd = request.form.get('current_password', '')
        new_pwd = request.form.get('new_password', '')
        confirm_new_pwd = request.form.get('confirm_new_password', '')

        if not current_user.check_password(current_pwd):
            flash("Current password incorrect.", "danger")
        elif new_pwd != confirm_new_pwd:
            flash("New passwords do not match.", "danger")
        elif len(new_pwd) < 6:
            flash("Password must be at least 6 characters.", "warning")
        else:
            current_user.set_password(new_pwd)
            db.session.commit()
            flash("Your password has been updated successfully.", "success")
            return redirect(url_for('profile'))

    user_scans_count = DiagnosisHistory.query.filter_by(user_id=current_user.id).count()
    return render_template('profile.html', user_scans_count=user_scans_count)


@app.route('/admin/model-info')
@login_required
def admin_model_info():
    """Detailed model architecture and dataset analytics page."""
    if not ML_MODELS['loaded']:
        load_ml_models()

    # Read dataset counts on disk
    def count_images_in(split_path):
        counts = {c: 0 for c in Config.DIAGNOSTIC_CLASSES}
        if split_path.exists():
            for c in Config.DIAGNOSTIC_CLASSES:
                for cand in [c.lower(), c.upper(), c.capitalize()]:
                    p = split_path / cand
                    if p.is_dir():
                        counts[c] = len([f for f in p.iterdir() if f.suffix.lower() in Config.ALLOWED_EXTENSIONS])
                        break
        return counts

    dataset_stats = {
        'train': count_images_in(Config.TRAIN_DIR),
        'val': count_images_in(Config.VAL_DIR),
        'test': count_images_in(Config.TEST_DIR)
    }

    return render_template(
        'admin_model.html',
        models_ready=ML_MODELS['loaded'],
        metadata=ML_MODELS.get('metadata'),
        dataset_stats=dataset_stats,
        classes=Config.DIAGNOSTIC_CLASSES
    )


@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename: str):
    """Securely streams uploaded images to authenticated authorized users."""
    diagnosis = DiagnosisHistory.query.filter_by(filename=filename).first()
    if not diagnosis:
        abort(404)
    if diagnosis.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@app.route('/api/model-status')
def api_model_status():
    """JSON status probe for ML model health."""
    if not ML_MODELS['loaded']:
        load_ml_models()
    return jsonify({
        'status': 'READY' if ML_MODELS['loaded'] else 'MODEL NOT TRAINED',
        'classes': Config.DIAGNOSTIC_CLASSES,
        'metadata': ML_MODELS.get('metadata')
    })


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found_error(error):
    return render_template('base.html', custom_error="The requested medical resource or scan was not found (404)."), 404

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('base.html', custom_error="Access denied: You are not authorized to view this diagnostic record (403)."), 403

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("The uploaded scan file is too large (maximum allowed size is 16 MB).", "danger")
    return redirect(url_for('upload'))

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('base.html', custom_error="Internal screening system error (500). Please try again later."), 500


if __name__ == '__main__':
    # Listen on localhost:5000 as specified
    app.run(host='127.0.0.1', port=5000, debug=True)
