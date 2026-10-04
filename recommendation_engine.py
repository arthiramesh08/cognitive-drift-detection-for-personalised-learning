"""
recommendation_engine.py – Cognitive Drift Detection & Personalized Recommendation Engine.

This module:
  1. Loads the trained Random Forest model (if available).
  2. Predicts the student's cognitive drift level: Stable / Mild Drift / High Drift.
  3. Generates a personalized, student-friendly recommendation with WHAT / WHY / HOW LONG.
  4. Falls back to a rule-based approach if the model file is not yet generated.

NOTE: This system only detects changes in observable LEARNING BEHAVIOR.
      It is NOT a medical or mental-health diagnostic tool.
"""

import os
import joblib
import numpy as np
import random
from datetime import date, timedelta

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'cognitive_drift_model.pkl')

DRIFT_LABELS = {0: 'Stable', 1: 'Mild Drift', 2: 'High Drift'}

# Recommendation bank: keyed by drift label (0, 1, 2)
RECOMMENDATIONS = {
    0: {  # Stable
        'label':  'Stable',
        'color':  'success',
        'icon':   '✅',
        'emoji':  '🌟',
        'badge_color': '#10B981',
        'actions': [
            {
                'action': 'Continue to the next topic',
                'reason': 'Your learning pattern is consistent and strong. You are ready to advance.',
                'time':   '45–60 minutes',
                'next':   'Take a practice quiz on your next topic to confirm readiness.'
            },
            {
                'action': 'Try advanced-level questions',
                'reason': 'Your quiz scores and engagement are high. Challenge yourself with harder problems!',
                'time':   '30 minutes',
                'next':   'Attempt the hard difficulty quiz on your current topic.'
            },
            {
                'action': 'Explore a related concept',
                'reason': 'You have mastered the current content. Broadening your knowledge will deepen understanding.',
                'time':   '40 minutes',
                'next':   'Read about one new concept and summarize it in your own words.'
            },
        ]
    },
    1: {  # Mild Drift
        'label':  'Mild Drift',
        'color':  'warning',
        'icon':   '⚠️',
        'emoji':  '📖',
        'badge_color': '#F59E0B',
        'actions': [
            {
                'action': 'Revise the previous topic',
                'reason': 'Your recent quiz performance shows a slight dip. A quick revision session will help consolidate your understanding.',
                'time':   '20–30 minutes',
                'next':   'Take a short 5-question practice quiz afterward to test retention.'
            },
            {
                'action': 'Watch a concept explanation video',
                'reason': 'Your response time is increasing, which may mean some concepts need more clarity.',
                'time':   '15–20 minutes',
                'next':   'Write down 3 key takeaways from the video.'
            },
            {
                'action': 'Try the Pomodoro study technique',
                'reason': 'Shorter, focused study intervals often improve concentration and retention.',
                'time':   '25 min study + 5 min break',
                'next':   'Complete two full Pomodoro sessions on your current topic.'
            },
        ]
    },
    2: {  # High Drift
        'label':  'High Drift',
        'color':  'danger',
        'icon':   '🔔',
        'emoji':  '💪',
        'badge_color': '#EF4444',
        'actions': [
            {
                'action': 'Review your weak concepts from the beginning',
                'reason': 'Your learning indicators show a significant change. Going back to fundamentals will build a stronger foundation.',
                'time':   '30 minutes',
                'next':   'Take a beginner-level quiz on your weakest topic to rebuild confidence.'
            },
            {
                'action': 'Set a smaller, achievable daily study goal',
                'reason': 'Shorter, consistent study sessions are more effective than long, irregular ones.',
                'time':   '20–25 minutes maximum',
                'next':   'Study one concept thoroughly rather than rushing through multiple topics.'
            },
            {
                'action': 'Reach out to your instructor or a study group',
                'reason': 'Getting support when you are struggling is a sign of strength. A fresh explanation often unlocks understanding.',
                'time':   'As needed',
                'next':   'Prepare 2–3 specific questions about topics you find difficult.'
            },
        ]
    }
}


# ─────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────

def predict_drift(features: dict) -> int:
    """
    Predict cognitive drift level from a student's learning features.

    Parameters
    ----------
    features : dict
        Keys: quiz_score, assignment_score, attendance, study_hours,
              response_time, login_frequency, previous_performance,
              engagement_level, learning_consistency

    Returns
    -------
    int  –  0 = Stable, 1 = Mild Drift, 2 = High Drift
    """
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            feature_vector = np.array([[
                features.get('quiz_score',           75.0),
                features.get('assignment_score',     75.0),
                features.get('attendance',           80.0),
                features.get('study_hours',           2.0),
                features.get('response_time',        45.0),
                features.get('login_frequency',       5.0),
                features.get('previous_performance', 75.0),
                features.get('engagement_level',      3.0),
                features.get('learning_consistency', 70.0),
            ]])
            return int(model.predict(feature_vector)[0])
    except Exception as e:
        print(f"[ML] Falling back to rule-based prediction. Reason: {e}")

    # Rule-based fallback when model is not yet trained
    return _rule_based_prediction(features)


def _rule_based_prediction(features: dict) -> int:
    """
    Rule-based fallback prediction when the ML model is unavailable.
    Uses weighted scoring of key learning indicators.
    """
    score = 0

    quiz_score   = features.get('quiz_score',           75.0)
    engagement   = features.get('engagement_level',      3.0)
    study_hours  = features.get('study_hours',           2.0)
    resp_time    = features.get('response_time',        45.0)
    consistency  = features.get('learning_consistency', 70.0)

    # Quiz score
    if quiz_score < 50:   score += 2
    elif quiz_score < 65: score += 1

    # Engagement
    if engagement <= 2:   score += 2
    elif engagement <= 3: score += 1

    # Study hours per day
    if study_hours < 1.0: score += 1

    # Response time (higher = slower = worse)
    if resp_time > 80:    score += 2
    elif resp_time > 60:  score += 1

    # Consistency (% of days active)
    if consistency < 40:  score += 2
    elif consistency < 60: score += 1

    if score >= 5:   return 2  # High Drift
    elif score >= 2: return 1  # Mild Drift
    else:            return 0  # Stable


def get_recommendation(drift_label: int,
                       topic: str = None,
                       weak_topics: list = None,
                       student_name: str = 'Student') -> dict:
    """
    Generate a personalized, student-friendly recommendation.

    Parameters
    ----------
    drift_label   : int  (0, 1, or 2)
    topic         : str  – current topic or weak topic
    weak_topics   : list – list of topics needing improvement
    student_name  : str

    Returns
    -------
    dict  – {recommendation, reason, suggested_time, next_action,
             drift_status, drift_color, drift_icon, drift_emoji, badge_color}
    """
    rec_data    = RECOMMENDATIONS.get(drift_label, RECOMMENDATIONS[0])
    action_data = random.choice(rec_data['actions'])

    action_text = action_data['action']
    reason_text = action_data['reason']

    # Personalize with topic name when available
    if topic:
        if drift_label == 0:
            action_text = f'Advance to the next topic after {topic}'
            reason_text = (
                f'You have shown strong performance in {topic}. '
                'Your learning indicators confirm you are ready to move forward.'
            )
        elif drift_label == 1:
            action_text = f'Revise {topic}'
            reason_text = (
                f'Your recent quiz performance in {topic} has slightly declined. '
                'A focused 20-minute revision session will help consolidate your understanding.'
            )
        elif drift_label == 2:
            action_text = f'Review {topic} from the beginning'
            reason_text = (
                f'Your learning indicators for {topic} suggest going back to fundamentals '
                'will build a stronger foundation before moving on.'
            )

    return {
        'recommendation': action_text,
        'reason':         reason_text,
        'suggested_time': action_data['time'],
        'next_action':    action_data['next'],
        'drift_status':   rec_data['label'],
        'drift_color':    rec_data['color'],
        'drift_icon':     rec_data['icon'],
        'drift_emoji':    rec_data['emoji'],
        'badge_color':    rec_data['badge_color'],
    }


def compute_student_features(user_id: int, models: dict) -> dict:
    """
    Compute the ML feature vector for a student from their recent DB activity.

    Parameters
    ----------
    user_id  : int
    models   : dict  – {'LearningActivity': ..., 'QuizResult': ..., 'DailyCheckin': ...}

    Returns
    -------
    dict of features ready for predict_drift()
    """
    LearningActivity = models['LearningActivity']
    QuizResult       = models['QuizResult']

    week_ago      = date.today() - timedelta(days=7)
    two_weeks_ago = date.today() - timedelta(days=14)

    recent_activities = LearningActivity.query.filter(
        LearningActivity.user_id == user_id,
        LearningActivity.date >= week_ago
    ).all()

    recent_quizzes = QuizResult.query.filter(
        QuizResult.user_id == user_id
    ).order_by(QuizResult.date.desc()).limit(5).all()

    # Average quiz score from last 5 attempts
    if recent_quizzes:
        avg_quiz_score = sum(q.score for q in recent_quizzes) / len(recent_quizzes)
        prev_score     = sum(q.score for q in recent_quizzes[-2:]) / max(len(recent_quizzes[-2:]), 1)
    else:
        avg_quiz_score = 75.0
        prev_score     = 75.0

    # Averages from recent activity
    if recent_activities:
        avg_study_hours  = sum(a.study_hours      for a in recent_activities) / len(recent_activities)
        avg_engagement   = sum(a.engagement_level for a in recent_activities) / len(recent_activities)
        avg_response     = sum(a.response_time    for a in recent_activities) / len(recent_activities)
    else:
        avg_study_hours = 2.0
        avg_engagement  = 3.0
        avg_response    = 45.0

    # Login frequency: distinct days active in past 7 days
    login_days = len(set(a.date for a in recent_activities))

    # Learning consistency: % of past 14 days with any activity
    all_14_day = LearningActivity.query.filter(
        LearningActivity.user_id == user_id,
        LearningActivity.date >= two_weeks_ago
    ).all()
    active_days  = len(set(a.date for a in all_14_day))
    consistency  = round((active_days / 14) * 100, 1)

    return {
        'quiz_score':           round(avg_quiz_score, 1),
        'assignment_score':     round(avg_quiz_score * 0.92, 1),
        'attendance':           min(round(consistency * 1.1, 1), 100.0),
        'study_hours':          round(avg_study_hours, 1),
        'response_time':        round(avg_response, 1),
        'login_frequency':      login_days,
        'previous_performance': round(prev_score, 1),
        'engagement_level':     round(avg_engagement, 1),
        'learning_consistency': consistency,
    }
