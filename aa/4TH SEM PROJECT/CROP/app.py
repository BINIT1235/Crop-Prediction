import os
import pickle
import numpy as np
import pymysql

from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

pymysql.install_as_MySQLdb()

app = Flask(__name__)

# ------------------ CONFIG ------------------
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost/crop'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ------------------ LOGIN ------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ------------------ LOAD MODEL ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

# ------------------ MODELS ------------------

class User(db.Model, UserMixin):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)

    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    email = db.Column(db.String(150), unique=True)

    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    is_admin = db.Column(db.Boolean, default=False)

    predictions = db.relationship('Prediction', backref='user', lazy=True)


class Prediction(db.Model):
    __tablename__ = 'prediction'

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    N = db.Column(db.Float)
    P = db.Column(db.Float)
    K = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    ph = db.Column(db.Float)
    rainfall = db.Column(db.Float)

    result = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# ------------------ LOAD USER ------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ------------------ ROUTES ------------------

@app.route('/')
@login_required
def home():
    return render_template('index.html')


# ------------------ PREDICT ------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get probabilities for all crops
        probabilities = model.predict_proba(features)[0]
        
        # Get class labels
        crop_classes = model.classes_
        
        # Create list of (crop, probability) tuples and sort by probability
        crop_predictions = list(zip(crop_classes, probabilities))
        crop_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 5 predictions
        top_predictions = crop_predictions[:5]

        # Store the top prediction in database
        new_prediction = Prediction(
            user_id=current_user.id,
            N=N, P=P, K=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            result=prediction
        )

        db.session.add(new_prediction)
        db.session.commit()

        return render_template("result.html", 
                             crop=prediction,
                             top_predictions=top_predictions,
                             input_data={
                                 'N': N, 'P': P, 'K': K,
                                 'temperature': temperature,
                                 'humidity': humidity,
                                 'ph': ph,
                                 'rainfall': rainfall
                             })

    except Exception as e:
        return f"Error occurred: {e}"


# ------------------ SIGNUP ------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        # MATCH YOUR HTML NAMES
        first_name = request.form.get('firstname')
        last_name = request.form.get('lastname')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm')

        if not all([first_name, last_name, email, username, password, confirm_password]):
            return "All fields are required"

        if password != confirm_password:
            return "Passwords do not match"

        if User.query.filter_by(username=username).first():
            return "Username already exists"

        if User.query.filter_by(email=email).first():
            return "Email already exists"

        hashed_password = generate_password_hash(password)

        user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            username=username,
            password=hashed_password
        )

        db.session.add(user)
        db.session.commit()

        return redirect('/login')

    return render_template('signup.html')


# ------------------ LOGIN ------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if not user:
            return "User not found"

        if user.is_admin:
            return "Use admin login page"

        if not check_password_hash(user.password, password):
            return "Wrong password"

        login_user(user)
        return redirect('/')

    return render_template('login.html')


# ------------------ LOGOUT ------------------
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect('/login')


# ------------------ ADMIN ------------------
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if not user:
            return "Admin not found"

        if not user.is_admin:
            return "Not an admin account"

        if not check_password_hash(user.password, password):
            return "Wrong password"

        login_user(user)
        return redirect('/admin')

    return render_template('admin_login.html')


@app.route('/admin')
def admin():
    if not current_user.is_authenticated or not current_user.is_admin:
        return "Access Denied"

    users = User.query.all()

    # 🔥 FORCE LOAD USER RELATIONSHIP
    predictions = Prediction.query.options(db.joinedload(Prediction.user)).order_by(Prediction.created_at.desc()).all()

    return render_template('admin.html', users=users, predictions=predictions)


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)