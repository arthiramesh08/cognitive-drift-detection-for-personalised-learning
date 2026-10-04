# 🧠 LearnFlow AI – Cognitive Drift Detection & Personalized Learning

LearnFlow AI is an AI-powered personalized learning platform for college students. It monitors observable learning behavior (such as quiz performance, study duration, response times, login frequency, and engagement) to detect **Cognitive Drift** (Stable, Mild Drift, High Drift) using a Random Forest Classifier and recommends actionable next study steps.

---

## 🌟 Key Features

1. **Cognitive Drift Detection**:
   - Machine Learning model classifies student learning behavior into 3 states: `Stable`, `Mild Drift`, or `High Drift`.
   - Uses 9 observable indicators without any medical diagnostic claims.

2. **Personalized Recommendations**:
   - Provides tailored recommendations featuring **WHAT** to do, **WHY** it is recommended, and **HOW LONG** to spend.

3. **Smart Quiz Module**:
   - Multiple-choice quizzes covering CS/ML topics (Python, Machine Learning, Data Structures, Statistics, Mathematics).
   - 3 difficulty levels (Easy, Medium, Hard), real-time countdown timer, immediate score breakdown, and weak-area analysis.

4. **Daily Study Check-in**:
   - Allows students to log daily study duration, topics covered, quiz performance, response time, and engagement levels.

5. **Learning Path Roadmap**:
   - Adaptive step-by-step curriculum showing completed, current, and locked modules.

6. **Progress Analytics & Charts**:
   - Interactive Chart.js visualizations for weekly study hours, quiz score trends, topic mastery breakdown, and 7-day engagement metrics.

7. **Gamification**:
   - Earn XP points, maintain consecutive daily study streaks, and unlock achievement badges.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Flask, Flask-SQLAlchemy, Flask-Login, Werkzeug
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib (Random Forest, Decision Tree, Logistic Regression)
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6+), Chart.js
- **Database**: SQLite (SQLAlchemy ORM)

---

## 📁 Project Structure

```
LearnFlowAI/
├── app.py                      # Main Flask application & routes
├── database.py                 # Database models & demo seeding script
├── recommendation_engine.py    # ML prediction & recommendation logic
├── train_model.py              # ML model training pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── dataset/
│   └── student_learning_data.csv # Synthetic student dataset
├── models/
│   └── cognitive_drift_model.pkl # Trained Random Forest model
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── learning.html
│   ├── quiz.html
│   ├── quiz_result.html
│   ├── checkin.html
│   ├── analytics.html
│   ├── recommendations.html
│   └── profile.html
├── static/
│   ├── css/
│   │   └── style.css           # Design tokens & UI styles
│   └── js/
│       └── script.js           # Client-side interactions & Chart.js setup
└── database/
    └── learnflow.db            # SQLite database instance
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
Open your terminal in the `LearnFlowAI` directory and run:
```bash
pip install -r requirements.txt
```

### 3. Train the ML Model
Before running the application for the first time, execute the training script:
```bash
python train_model.py
```
This trains Random Forest, Decision Tree, and Logistic Regression models on the dataset, evaluates performance metrics, and exports `models/cognitive_drift_model.pkl`.

### 4. Run the Application
Start the Flask development server:
```bash
python app.py
```
Access the application in your browser at `http://127.0.0.1:5000/`.

---

## 🔑 Demo Account Credentials

The application automatically seeds a demo student profile (**Arthi**) with 30 days of historical activity data:

- **Email**: `arthi@demo.com`
- **Password**: `demo123`

You can also create a new student account using the Registration page.

---

## 🔮 Future Enhancements

- Integration of LLM-generated explanations for complex quiz questions.
- Collaborative study groups and peer comparison dashboards.
- Dynamic quiz generation from custom PDF course notes.
