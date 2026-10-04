"""
app.py – Main Flask Application for LearnFlow AI
Cognitive Drift Detection & Personalized Learning Platform

Routes:
    /               → Landing page
    /login          → Login
    /logout         → Logout
    /register       → Registration
    /dashboard      → Student dashboard (protected)
    /learning       → Learning path (protected)
    /quiz           → Quiz module (protected)
    /quiz/submit    → Quiz submission handler (protected)
    /checkin        → Daily check-in (protected)
    /analytics      → Analytics & charts (protected)
    /recommendations→ Recommendation history (protected)
    /profile        → Student profile (protected)
    /api/chart-data → JSON chart data (protected)
"""

import os
import json
from datetime import datetime, date, timedelta

from flask import (
    Flask, render_template, redirect, url_for,
    request, flash, jsonify, session
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)

from database import db, User, LearningActivity, QuizResult, Recommendation, DailyCheckin, seed_demo_data
from recommendation_engine import predict_drift, get_recommendation, compute_student_features

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY']                = 'learnflow-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI']   = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'learnflow.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'),   exist_ok=True)

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view    = 'login'
login_manager.login_message = 'Please log in to access your dashboard.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ─────────────────────────────────────────────
# Quiz Question Bank
# ─────────────────────────────────────────────

QUIZ_QUESTIONS = {
    "Python": {
        "easy": [
            {"q": "What is the output of print(type(5))?", "options": ["<class 'float'>", "<class 'int'>", "<class 'str'>", "<class 'num'>"], "answer": 1},
            {"q": "Which keyword is used to define a function in Python?", "options": ["func", "function", "def", "define"], "answer": 2},
            {"q": "What does len([1, 2, 3]) return?", "options": ["1", "2", "3", "4"], "answer": 2},
            {"q": "Which of these creates a valid list in Python?", "options": ["(1, 2, 3)", "{1, 2, 3}", "1, 2, 3", "[1, 2, 3]"], "answer": 3},
            {"q": "What is the correct way to assign a variable?", "options": ["var x = 5", "int x = 5", "let x = 5", "x = 5"], "answer": 3},
        ],
        "medium": [
            {"q": "What does 'self' refer to inside a class method?", "options": ["The class itself", "The parent class", "The current instance", "None"], "answer": 2},
            {"q": "What is a lambda function?", "options": ["A named function", "A recursive function", "A built-in function", "An anonymous function"], "answer": 3},
            {"q": "Which module handles regular expressions in Python?", "options": ["regex", "regexp", "pattern", "re"], "answer": 3},
            {"q": "What is the difference between '==' and 'is'?", "options": ["No difference", "'is' checks value, '==' checks identity", "Both check identity", "'==' checks value, 'is' checks identity"], "answer": 3},
            {"q": "What does a list comprehension do?", "options": ["Sorts a list", "Filters duplicates", "Creates a dictionary", "Creates a list from an iterable"], "answer": 3},
        ],
        "hard": [
            {"q": "What is a decorator in Python?", "options": ["A design pattern", "A class attribute", "A built-in module", "A function that modifies another function"], "answer": 3},
            {"q": "What does *args do in a function definition?", "options": ["Passes keyword arguments", "Passes a dictionary", "Passes a list", "Accepts variable positional arguments"], "answer": 3},
            {"q": "What is the purpose of __init__.py?", "options": ["Initialize variables", "Import modules", "Set class attributes", "Mark a directory as a Python package"], "answer": 3},
            {"q": "What does the 'yield' keyword do?", "options": ["Returns a value and ends the function", "Raises an exception", "Imports a module", "Creates a generator function"], "answer": 3},
            {"q": "What is GIL in Python?", "options": ["Global Input Library", "General Interface Layer", "Global Inline List", "Global Interpreter Lock"], "answer": 3},
        ],
    },
    "Machine Learning": {
        "easy": [
            {"q": "Which algorithm is used for classification?", "options": ["Linear Regression", "K-Means", "PCA", "Random Forest"], "answer": 3},
            {"q": "What does ML stand for?", "options": ["Multi Learning", "Model Learning", "Max Learning", "Machine Learning"], "answer": 3},
            {"q": "Which is a supervised learning algorithm?", "options": ["K-Means", "DBSCAN", "PCA", "Decision Tree"], "answer": 3},
            {"q": "What is overfitting?", "options": ["Model performs well on test data", "Model is too simple", "Model has no features", "Model performs too well on training data only"], "answer": 3},
            {"q": "What does 'training data' mean?", "options": ["Test data", "Raw data", "Validation data", "Data used to train the model"], "answer": 3},
        ],
        "medium": [
            {"q": "What metric evaluates classification models?", "options": ["MSE", "RMSE", "R-squared", "Accuracy"], "answer": 3},
            {"q": "What is cross-validation?", "options": ["Validating with one dataset", "A type of algorithm", "Splitting data 50/50", "Assessing model performance on unseen data"], "answer": 3},
            {"q": "What is the purpose of a confusion matrix?", "options": ["Show model architecture", "Plot feature importance", "Show training loss", "Visualize classification results"], "answer": 3},
            {"q": "What is regularization?", "options": ["Increasing model complexity", "Feature scaling", "Data augmentation", "Technique to prevent overfitting"], "answer": 3},
            {"q": "What is feature engineering?", "options": ["Selecting an algorithm", "Removing duplicates", "Normalizing data", "Creating new features from existing data"], "answer": 3},
        ],
        "hard": [
            {"q": "What is the kernel trick in SVM?", "options": ["Feature selection", "Reducing dimensionality", "Normalizing features", "Mapping data to a higher dimension"], "answer": 3},
            {"q": "What is gradient boosting?", "options": ["Training models in parallel", "Averaging model predictions", "Random sampling", "Building models sequentially to correct errors"], "answer": 3},
            {"q": "What is the bias-variance tradeoff?", "options": ["Accuracy vs speed", "Training vs testing", "Precision vs recall", "Balance between underfitting and overfitting"], "answer": 3},
            {"q": "What does SMOTE do?", "options": ["Feature selection", "Removes outliers", "Scales features", "Handles class imbalance by oversampling"], "answer": 3},
            {"q": "What is an ensemble method?", "options": ["Single powerful model", "Data preprocessing", "Feature extraction", "Combining multiple models for better predictions"], "answer": 3},
        ],
    },
    "Data Structures": {
        "easy": [
            {"q": "What is an array?", "options": ["A linked list", "A tree structure", "A hash map", "Collection of elements of the same type"], "answer": 3},
            {"q": "What is a stack?", "options": ["FIFO data structure", "Sorted list", "Binary tree", "LIFO data structure"], "answer": 3},
            {"q": "What does FIFO stand for?", "options": ["Fast In Fast Out", "First Item Final Output", "Full Input Final Output", "First In First Out"], "answer": 3},
            {"q": "Which data structure uses key-value pairs?", "options": ["Stack", "Queue", "Array", "Dictionary / Hash Map"], "answer": 3},
            {"q": "What is a linked list?", "options": ["An array", "A sorted list", "A tree", "Elements connected by pointers"], "answer": 3},
        ],
        "medium": [
            {"q": "What is the time complexity of binary search?", "options": ["O(n)", "O(n²)", "O(1)", "O(log n)"], "answer": 3},
            {"q": "What is a binary tree?", "options": ["Tree with two nodes", "Sorted tree", "Balanced tree", "Each node has at most two children"], "answer": 3},
            {"q": "What is hashing?", "options": ["Sorting data", "Encrypting data", "Compressing data", "Mapping data to fixed-size values"], "answer": 3},
            {"q": "What is the average time complexity of hash table insertion?", "options": ["O(n)", "O(log n)", "O(n²)", "O(1)"], "answer": 3},
            {"q": "What is a queue used for?", "options": ["Undo operations", "Recursive algorithms", "Searching", "Scheduling and order processing"], "answer": 3},
        ],
        "hard": [
            {"q": "What is a Red-Black tree?", "options": ["Colored binary tree", "Hash tree", "Ordered set", "Self-balancing BST"], "answer": 3},
            {"q": "What is dynamic programming?", "options": ["Runtime programming", "Using dynamic memory", "Parallel computing", "Solving overlapping subproblems optimally"], "answer": 3},
            {"q": "What is the space complexity of merge sort?", "options": ["O(1)", "O(log n)", "O(n²)", "O(n)"], "answer": 3},
            {"q": "What is a trie data structure used for?", "options": ["Sorting numbers", "Graph traversal", "Hashing", "String prefix searching"], "answer": 3},
            {"q": "DFS vs BFS: which uses a queue?", "options": ["DFS", "Both", "Neither", "BFS"], "answer": 3},
        ],
    },
    "Statistics": {
        "easy": [
            {"q": "What is the mean?", "options": ["Middle value", "Most frequent value", "Difference between max and min", "Sum divided by count"], "answer": 3},
            {"q": "What is the median?", "options": ["Average value", "Most common value", "Standard deviation", "Middle value when sorted"], "answer": 3},
            {"q": "What is the mode?", "options": ["Average", "Middle value", "Range", "Most frequent value"], "answer": 3},
            {"q": "What does standard deviation measure?", "options": ["Average value", "Minimum value", "Data range", "Spread of data from the mean"], "answer": 3},
            {"q": "What is a histogram?", "options": ["Line chart", "Pie chart", "Scatter plot", "Bar chart showing frequency distribution"], "answer": 3},
        ],
        "medium": [
            {"q": "What is a normal distribution?", "options": ["Skewed distribution", "Uniform distribution", "Bimodal distribution", "Bell-shaped symmetric distribution"], "answer": 3},
            {"q": "What is correlation?", "options": ["Causation", "Average of variables", "Difference between variables", "Statistical relationship between variables"], "answer": 3},
            {"q": "What is variance?", "options": ["Mean squared", "Standard deviation", "Median deviation", "Average squared deviation from the mean"], "answer": 3},
            {"q": "What is a confidence interval?", "options": ["Exact parameter value", "Sample size", "Test statistic", "Range likely to contain the true parameter"], "answer": 3},
            {"q": "What is a p-value?", "options": ["Correlation coefficient", "Mean value", "Sample size", "Probability of the result occurring by chance under the null hypothesis"], "answer": 3},
        ],
        "hard": [
            {"q": "What does the Central Limit Theorem state?", "options": ["All distributions are normal", "Mean equals median", "Large samples are always accurate", "Sample means approach a normal distribution as n increases"], "answer": 3},
            {"q": "What is Type I error?", "options": ["Failing to reject a false null hypothesis", "Sample bias", "Measurement error", "Rejecting a true null hypothesis"], "answer": 3},
            {"q": "What is Bayesian inference?", "options": ["Frequency-based statistics", "Hypothesis testing", "Regression analysis", "Updating beliefs with new evidence using Bayes' theorem"], "answer": 3},
            {"q": "What is multicollinearity?", "options": ["Multiple dependent variables", "Non-linear relationship", "Missing data", "High correlation between independent variables"], "answer": 3},
            {"q": "What is the chi-squared test used for?", "options": ["Comparing means", "Checking normality", "Measuring correlation", "Testing independence between categorical variables"], "answer": 3},
        ],
    },
    "Mathematics": {
        "easy": [
            {"q": "What is a derivative?", "options": ["Integral of a function", "Area under curve", "Sum of functions", "Rate of change of a function"], "answer": 3},
            {"q": "What is a matrix?", "options": ["Single number", "Vector", "Polynomial", "A 2D array of numbers"], "answer": 3},
            {"q": "What is log(1)?", "options": ["1", "-1", "Undefined", "0"], "answer": 3},
            {"q": "What does Σ (sigma) represent?", "options": ["Product", "Difference", "Ratio", "Sum"], "answer": 3},
            {"q": "What is e approximately equal to?", "options": ["3.14", "1.41", "1.73", "2.71"], "answer": 3},
        ],
        "medium": [
            {"q": "What is the gradient in calculus?", "options": ["Maximum value", "Second derivative", "Integral", "Vector of partial derivatives"], "answer": 3},
            {"q": "What is an eigenvalue?", "options": ["Matrix determinant", "Vector length", "Matrix trace", "Scalar associated with a linear transformation"], "answer": 3},
            {"q": "What is the chain rule?", "options": ["Product rule", "Integration technique", "Limit definition", "Derivative of a composite function"], "answer": 3},
            {"q": "What is a dot product?", "options": ["Matrix multiplication", "Cross multiplication", "Vector addition", "Sum of products of corresponding vector elements"], "answer": 3},
            {"q": "What is a convex function?", "options": ["Decreasing function", "Non-differentiable function", "Periodic function", "A function where line segment between two points lies above the curve"], "answer": 3},
        ],
        "hard": [
            {"q": "What is the Jacobian matrix?", "options": ["Hessian of a function", "Covariance matrix", "Transition matrix", "Matrix of all first-order partial derivatives"], "answer": 3},
            {"q": "What is the purpose of Lagrange multipliers?", "options": ["Finding derivatives", "Integration", "Matrix decomposition", "Optimization under constraints"], "answer": 3},
            {"q": "What is SVD?", "options": ["Matrix addition", "Eigenvalue calculation", "Determinant expansion", "Factoring a matrix into U, Σ, V^T"], "answer": 3},
            {"q": "What is a Fourier transform?", "options": ["Data normalization", "Matrix transformation", "Probability distribution", "Decomposing a signal into frequency components"], "answer": 3},
            {"q": "What does the softmax function do?", "options": ["Normalize to [-1, 1]", "Scale features", "Clip activations", "Convert raw scores to probabilities summing to 1"], "answer": 3},
        ],
    },
}

# ─────────────────────────────────────────────
# Learning Path Definition
# ─────────────────────────────────────────────

LEARNING_PATH = [
    {"id": 1, "topic": "Python Basics",         "description": "Variables, data types, control flow, functions", "status": "completed", "icon": "🐍"},
    {"id": 2, "topic": "Data Preprocessing",    "description": "Data cleaning, normalization, encoding",         "status": "completed", "icon": "🔧"},
    {"id": 3, "topic": "Supervised Learning",   "description": "Classification and regression fundamentals",     "status": "current",   "icon": "🎯"},
    {"id": 4, "topic": "Classification",        "description": "Decision Trees, Random Forest, SVM",             "status": "locked",    "icon": "🌳"},
    {"id": 5, "topic": "Regression",            "description": "Linear, Polynomial, Ridge Regression",           "status": "locked",    "icon": "📈"},
    {"id": 6, "topic": "Model Evaluation",      "description": "Metrics, cross-validation, confusion matrix",   "status": "locked",    "icon": "📊"},
    {"id": 7, "topic": "Unsupervised Learning", "description": "K-Means clustering, PCA",                        "status": "locked",    "icon": "🔍"},
]

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def compute_streak(user_id):
    """Calculate the student's current consecutive study-day streak."""
    activities = (
        LearningActivity.query
        .filter_by(user_id=user_id)
        .order_by(LearningActivity.date.desc())
        .all()
    )
    if not activities:
        return 0

    unique_dates = sorted(set(a.date for a in activities), reverse=True)
    streak = 0
    check  = date.today()

    for d in unique_dates:
        if d == check or d == check - timedelta(days=1):
            streak += 1
            check   = d
        else:
            break

    return streak


def check_and_award_badges(user):
    """Check badge eligibility and award new badges. Returns list of newly earned badges."""
    newly_earned = []
    quiz_count   = QuizResult.query.filter_by(user_id=user.id).count()
    streak       = compute_streak(user.id)
    top_quizzes  = QuizResult.query.filter(
        QuizResult.user_id == user.id,
        QuizResult.score >= 80
    ).count()
    perfect_quiz = QuizResult.query.filter(
        QuizResult.user_id == user.id,
        QuizResult.score >= 100
    ).count()

    badge_rules = [
        (quiz_count >= 1,       '🏆 First Quiz'),
        (streak >= 7,           '🔥 7-Day Streak'),
        (streak >= 5,           '🔥 5-Day Streak'),
        (top_quizzes >= 3,      '🎯 80% Quiz Master'),
        (perfect_quiz >= 1,     '⭐ Perfect Score'),
        (quiz_count >= 5,       '📚 5 Quizzes Done'),
    ]

    for condition, badge in badge_rules:
        if condition:
            if user.add_badge(badge):
                newly_earned.append(badge)

    if newly_earned:
        db.session.commit()

    return newly_earned


def get_drift_data_for_user(user_id):
    """Run ML prediction for a user and return drift info + recommendation."""
    models_dict = {
        'LearningActivity': LearningActivity,
        'QuizResult':       QuizResult,
        'DailyCheckin':     DailyCheckin,
    }
    features    = compute_student_features(user_id, models_dict)
    drift_label = predict_drift(features)

    # Find weak topic from quiz history
    recent_quizzes = (
        QuizResult.query
        .filter_by(user_id=user_id)
        .order_by(QuizResult.date.desc())
        .limit(3)
        .all()
    )
    weak_topic = None
    if recent_quizzes:
        worst = min(recent_quizzes, key=lambda q: q.score)
        if worst.score < 70:
            weak_topic = worst.topic

    rec = get_recommendation(drift_label, topic=weak_topic)
    return drift_label, features, rec


# ─────────────────────────────────────────────
# Routes – Public Pages
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Landing page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.name}! 👋', 'success')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')
        subject  = request.form.get('subject', 'Machine Learning')

        # Validation
        if not name or not email or not password:
            flash('Please fill in all required fields.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('An account with this email already exists.', 'danger')
        else:
            user = User(name=name, email=email, subject=subject, xp=0, streak=0)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash(f'Account created! Welcome to LearnFlow AI, {name}! 🎉', 'success')
            return redirect(url_for('dashboard'))

    return render_template('register.html')


# ─────────────────────────────────────────────
# Routes – Protected Pages
# ─────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    """Main student dashboard with ML drift prediction."""
    user = current_user

    # Stats
    week_ago        = date.today() - timedelta(days=7)
    recent_acts     = LearningActivity.query.filter(
        LearningActivity.user_id == user.id,
        LearningActivity.date    >= week_ago
    ).all()
    weekly_hours    = round(sum(a.study_hours for a in recent_acts), 1)

    all_quizzes     = QuizResult.query.filter_by(user_id=user.id).order_by(QuizResult.date.desc()).limit(10).all()
    avg_score       = round(sum(q.score for q in all_quizzes) / len(all_quizzes), 1) if all_quizzes else 0
    streak          = compute_streak(user.id)

    # ML prediction
    drift_label, features, rec_dict = get_drift_data_for_user(user.id)

    now_formatted = date.today().strftime('%B %d, %Y')

    # Save recommendation if none today
    today_rec = Recommendation.query.filter(
        Recommendation.user_id == user.id,
        db.func.date(Recommendation.date) == date.today().isoformat()
    ).first()

    if not today_rec:
        today_rec = Recommendation(
            user_id        = user.id,
            recommendation = rec_dict['recommendation'],
            reason         = rec_dict['reason'],
            suggested_time = rec_dict['suggested_time'],
            drift_status   = rec_dict['drift_status'],
            next_action    = rec_dict['next_action'],
        )
        db.session.add(today_rec)
        db.session.commit()

    # Recent feed
    recent_feed = (
        LearningActivity.query
        .filter_by(user_id=user.id)
        .order_by(LearningActivity.date.desc())
        .limit(5)
        .all()
    )
    recent_quizzes = (
        QuizResult.query
        .filter_by(user_id=user.id)
        .order_by(QuizResult.date.desc())
        .limit(3)
        .all()
    )

    badges = user.get_badges()

    return render_template('dashboard.html',
        user           = user,
        now            = now_formatted,
        streak         = streak,
        weekly_hours   = weekly_hours,
        avg_score      = avg_score,
        drift_label    = drift_label,
        drift_status   = rec_dict['drift_status'],
        drift_color    = rec_dict['drift_color'],
        drift_icon     = rec_dict['drift_icon'],
        drift_emoji    = rec_dict['drift_emoji'],
        badge_color    = rec_dict['badge_color'],
        recommendation = today_rec,
        learning_path  = LEARNING_PATH,
        recent_feed    = recent_feed,
        recent_quizzes = recent_quizzes,
        badges         = badges,
        features       = features,
    )


@app.route('/learning')
@login_required
def learning():
    """Personalized learning path page."""
    # Determine completed topics from quiz history (score ≥ 70)
    strong_topics = set()
    quizzes = QuizResult.query.filter_by(user_id=current_user.id).all()
    topic_scores = {}
    for q in quizzes:
        if q.topic not in topic_scores:
            topic_scores[q.topic] = []
        topic_scores[q.topic].append(q.score)

    for topic, scores in topic_scores.items():
        if sum(scores) / len(scores) >= 70:
            strong_topics.add(topic)

    return render_template('learning.html',
        user          = current_user,
        learning_path = LEARNING_PATH,
        strong_topics = strong_topics,
        topic_scores  = topic_scores,
    )


@app.route('/quiz')
@login_required
def quiz():
    """Quiz selection and question page."""
    topics     = list(QUIZ_QUESTIONS.keys())
    topic      = request.args.get('topic', topics[0])
    difficulty = request.args.get('difficulty', 'easy')

    if topic not in QUIZ_QUESTIONS:
        topic = topics[0]
    if difficulty not in ('easy', 'medium', 'hard'):
        difficulty = 'easy'

    questions  = QUIZ_QUESTIONS[topic][difficulty]

    return render_template('quiz.html',
        user       = current_user,
        topics     = topics,
        topic      = topic,
        difficulty = difficulty,
        questions  = questions,
    )


@app.route('/quiz/submit', methods=['POST'])
@login_required
def quiz_submit():
    """Process quiz answers, calculate score, save results, generate recommendation."""
    topic      = request.form.get('topic', 'Python')
    difficulty = request.form.get('difficulty', 'easy')

    if topic not in QUIZ_QUESTIONS or difficulty not in QUIZ_QUESTIONS[topic]:
        flash('Invalid quiz parameters.', 'danger')
        return redirect(url_for('quiz'))

    questions = QUIZ_QUESTIONS[topic][difficulty]
    correct   = 0
    results   = []

    for i, q in enumerate(questions):
        user_answer = request.form.get(f'q{i}')
        is_valid = user_answer is not None and user_answer.strip().isdigit()
        user_ans_val = int(user_answer) if is_valid else -1
        is_correct  = is_valid and user_ans_val == q['answer']
        if is_correct:
            correct += 1
        results.append({
            'question':      q['q'],
            'options':       q['options'],
            'user_answer':   user_ans_val,
            'correct_answer': q['answer'],
            'is_correct':    is_correct,
        })

    score_pct = round((correct / len(questions)) * 100, 1)

    # Identify weak areas
    wrong_indices = [i for i, r in enumerate(results) if not r['is_correct']]
    weak_areas    = topic if len(wrong_indices) >= 3 else ''

    # Save quiz result
    quiz_result = QuizResult(
        user_id       = current_user.id,
        topic         = topic,
        score         = score_pct,
        difficulty    = difficulty,
        num_questions = len(questions),
        correct       = correct,
        weak_areas    = weak_areas,
    )
    db.session.add(quiz_result)

    # Log as learning activity
    activity = LearningActivity(
        user_id         = current_user.id,
        date            = date.today(),
        study_hours     = 0.5,
        quiz_score      = score_pct,
        response_time   = 45.0,
        engagement_level= 4,
        topics_studied  = topic,
    )
    db.session.add(activity)

    # Award XP
    xp_earned = 10 + (correct * 2)
    current_user.add_xp(xp_earned)
    db.session.commit()

    # Check badges
    new_badges = check_and_award_badges(current_user)

    # Generate ML-based recommendation
    drift_label, _, rec_dict = get_drift_data_for_user(current_user.id)

    # Determine strong/weak
    strong = topic if score_pct >= 70 else None
    weak   = topic if score_pct < 70  else None

    return render_template('quiz_result.html',
        user        = current_user,
        topic       = topic,
        difficulty  = difficulty,
        score       = score_pct,
        correct     = correct,
        total       = len(questions),
        results     = results,
        strong      = strong,
        weak        = weak,
        xp_earned   = xp_earned,
        new_badges  = new_badges,
        drift_status= rec_dict['drift_status'],
        drift_color = rec_dict['drift_color'],
        rec_action  = rec_dict['recommendation'],
        rec_reason  = rec_dict['reason'],
        rec_time    = rec_dict['suggested_time'],
    )


@app.route('/checkin', methods=['GET', 'POST'])
@login_required
def checkin():
    """Daily study session check-in."""
    if request.method == 'POST':
        try:
            study_duration      = float(request.form.get('study_duration', 0) or 0)
            topics_studied      = request.form.get('topics_studied', '')
            raw_quiz_score      = request.form.get('quiz_score', '').strip()
            quiz_score          = float(raw_quiz_score) if raw_quiz_score else 0.0
            questions_attempted = int(request.form.get('questions_attempted', 0) or 0)
            response_time       = float(request.form.get('response_time', 45) or 45)
            difficulty_level    = request.form.get('difficulty_level', 'medium')
            engagement_level    = int(request.form.get('engagement_level', 3) or 3)
            note                = request.form.get('note', '')

            # Save check-in
            ci = DailyCheckin(
                user_id             = current_user.id,
                date                = date.today(),
                study_duration      = study_duration,
                topics_studied      = topics_studied,
                quiz_score          = quiz_score,
                questions_attempted = questions_attempted,
                response_time       = response_time,
                difficulty_level    = difficulty_level,
                engagement_level    = engagement_level,
                note                = note,
            )
            db.session.add(ci)

            # Log as learning activity
            activity = LearningActivity(
                user_id          = current_user.id,
                date             = date.today(),
                study_hours      = study_duration,
                quiz_score       = quiz_score if raw_quiz_score else 75.0,
                response_time    = response_time,
                engagement_level = engagement_level,
                topics_studied   = topics_studied,
            )
            db.session.add(activity)

            # Award XP
            current_user.add_xp(5)
            db.session.commit()

            flash('Check-in saved! Your dashboard has been updated. 🎉', 'success')
            return redirect(url_for('dashboard'))

        except (ValueError, TypeError) as e:
            flash('Please fill in all fields correctly.', 'danger')

    topics = list(QUIZ_QUESTIONS.keys())
    return render_template('checkin.html', user=current_user, topics=topics)


@app.route('/analytics')
@login_required
def analytics():
    """Analytics page with performance overview."""
    user = current_user

    # Last 30 days of activity
    month_ago   = date.today() - timedelta(days=30)
    activities  = (
        LearningActivity.query
        .filter(LearningActivity.user_id == user.id, LearningActivity.date >= month_ago)
        .order_by(LearningActivity.date)
        .all()
    )

    all_quizzes = (
        QuizResult.query
        .filter_by(user_id=user.id)
        .order_by(QuizResult.date)
        .all()
    )

    # Topic-wise average score
    topic_scores = {}
    for q in all_quizzes:
        if q.topic not in topic_scores:
            topic_scores[q.topic] = []
        topic_scores[q.topic].append(q.score)

    topic_avg = {t: round(sum(s)/len(s), 1) for t, s in topic_scores.items()}

    # Streak
    streak      = compute_streak(user.id)
    total_hours = round(sum(a.study_hours for a in activities), 1)
    avg_score   = round(sum(q.score for q in all_quizzes) / len(all_quizzes), 1) if all_quizzes else 0

    # Completed topics (avg ≥ 70)
    completed_topics = [t for t, avg in topic_avg.items() if avg >= 70]
    weak_topics      = [t for t, avg in topic_avg.items() if avg < 70]

    return render_template('analytics.html',
        user             = user,
        streak           = streak,
        total_hours      = total_hours,
        avg_score        = avg_score,
        topic_avg        = topic_avg,
        completed_topics = completed_topics,
        weak_topics      = weak_topics,
        quiz_count       = len(all_quizzes),
        activity_count   = len(activities),
    )


@app.route('/recommendations')
@login_required
def recommendations():
    """Full recommendation history page."""
    all_recs = (
        Recommendation.query
        .filter_by(user_id=current_user.id)
        .order_by(Recommendation.date.desc())
        .all()
    )

    # Run fresh prediction for today
    drift_label, features, rec_dict = get_drift_data_for_user(current_user.id)

    return render_template('recommendations.html',
        user         = current_user,
        all_recs     = all_recs,
        drift_status = rec_dict['drift_status'],
        drift_color  = rec_dict['drift_color'],
        drift_icon   = rec_dict['drift_icon'],
        badge_color  = rec_dict['badge_color'],
        rec_dict     = rec_dict,
        features     = features,
    )


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """Student profile page."""
    user = current_user

    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        subject = request.form.get('subject', user.subject)

        if name:
            user.name    = name
            user.subject = subject
            db.session.commit()
            flash('Profile updated successfully! ✅', 'success')
        else:
            flash('Name cannot be empty.', 'danger')

        return redirect(url_for('profile'))

    streak     = compute_streak(user.id)
    quiz_count = QuizResult.query.filter_by(user_id=user.id).count()
    badges     = user.get_badges()
    all_badges = [
        {'icon': '🏆', 'name': 'First Quiz',       'desc': 'Completed your first quiz',    'earned': '🏆 First Quiz'       in badges},
        {'icon': '🔥', 'name': '5-Day Streak',      'desc': '5 consecutive study days',     'earned': '🔥 5-Day Streak'      in badges},
        {'icon': '🔥', 'name': '7-Day Streak',      'desc': '7 consecutive study days',     'earned': '🔥 7-Day Streak'      in badges},
        {'icon': '🎯', 'name': '80% Quiz Master',   'desc': 'Scored 80%+ on 3+ quizzes',   'earned': '🎯 80% Quiz Master'   in badges},
        {'icon': '📚', 'name': '5 Quizzes Done',    'desc': 'Completed 5 quizzes',          'earned': '📚 5 Quizzes Done'    in badges},
        {'icon': '⭐', 'name': 'Perfect Score',     'desc': 'Scored 100% on a quiz',        'earned': '⭐ Perfect Score'     in badges},
        {'icon': '📚', 'name': '3 Topics Completed','desc': '3 topics with avg score ≥ 70', 'earned': '📚 3 Topics Completed' in badges},
    ]

    topics = list(QUIZ_QUESTIONS.keys())
    return render_template('profile.html',
        user       = user,
        streak     = streak,
        quiz_count = quiz_count,
        badges     = badges,
        all_badges = all_badges,
        topics     = topics,
    )


# ─────────────────────────────────────────────
# API – Chart Data
# ─────────────────────────────────────────────

@app.route('/api/chart-data')
@login_required
def chart_data():
    """Return JSON data for Chart.js charts on the analytics page."""
    user_id   = current_user.id
    month_ago = date.today() - timedelta(days=30)

    # Weekly study hours (last 7 days)
    labels_7   = []
    hours_7    = []
    for i in range(6, -1, -1):
        d = date.today() - timedelta(days=i)
        labels_7.append(d.strftime('%a'))
        acts = LearningActivity.query.filter(
            LearningActivity.user_id == user_id,
            LearningActivity.date    == d
        ).all()
        hours_7.append(round(sum(a.study_hours for a in acts), 1))

    # Quiz score trend (last 10 quizzes)
    quizzes = (
        QuizResult.query
        .filter_by(user_id=user_id)
        .order_by(QuizResult.date)
        .limit(10)
        .all()
    )
    quiz_labels = [f'Quiz {i+1}' for i in range(len(quizzes))]
    quiz_scores = [q.score for q in quizzes]
    quiz_topics = [q.topic for q in quizzes]

    # Topic-wise performance
    all_quizzes = QuizResult.query.filter_by(user_id=user_id).all()
    topic_scores = {}
    for q in all_quizzes:
        if q.topic not in topic_scores:
            topic_scores[q.topic] = []
        topic_scores[q.topic].append(q.score)
    topic_labels = list(topic_scores.keys())
    topic_avgs   = [round(sum(v)/len(v), 1) for v in topic_scores.values()]

    # Engagement trend (last 7 days)
    eng_data = []
    for i in range(6, -1, -1):
        d    = date.today() - timedelta(days=i)
        acts = LearningActivity.query.filter(
            LearningActivity.user_id == user_id,
            LearningActivity.date    == d
        ).all()
        eng_data.append(round(sum(a.engagement_level for a in acts) / max(len(acts), 1), 1) if acts else 0)

    return jsonify({
        'weekly_hours':  {'labels': labels_7,    'data': hours_7},
        'quiz_trend':    {'labels': quiz_labels,  'data': quiz_scores, 'topics': quiz_topics},
        'topic_perf':    {'labels': topic_labels, 'data': topic_avgs},
        'engagement':    {'labels': labels_7,    'data': eng_data},
    })


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_demo_data(app)
    app.run(debug=True, port=5000)
