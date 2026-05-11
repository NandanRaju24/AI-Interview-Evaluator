from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------- STOPWORDS ----------------
custom_stopwords = {
    "using", "use", "used", "also", "like"
}

stop_words = set(stopwords.words('english')).union(custom_stopwords)

# ---------------- SCORE FUNCTION ----------------
def give_score(similarity):
    if similarity >= 0.8:
        return "Excellent (9-10)"
    elif similarity >= 0.65:
        return "Good (7-8)"
    elif similarity >= 0.4:
        return "Average (5-6)"
    else:
        return "Poor (0-4)"

# ---------------- CONFIDENCE FUNCTION ----------------
def confidence_level(answer):
    length = len(answer.split())

    if length > 12:
        return "High Confidence"
    elif length > 6:
        return "Medium Confidence"
    else:
        return "Low Confidence"

# ---------------- RECOMMENDATION FUNCTION ----------------
def recommendation(avg):
    if avg >= 0.8:
        return "Highly Recommended"
    elif avg >= 0.6:
        return "Recommended"
    elif avg >= 0.4:
        return "Needs Improvement"
    else:
        return "Not Recommended"

# ---------------- KEYWORD EXTRACTION ----------------
def get_keywords(text):
    words = text.lower().split()
    return set([word for word in words if word not in stop_words])

# ---------------- FEEDBACK FUNCTION ----------------
def give_feedback(ideal, student):
    ideal_words = get_keywords(ideal)
    student_words = get_keywords(student)

    missing = ideal_words - student_words

    return list(missing)[:5]

# ---------------- TOPICS ----------------
topics = [
    "Machine Learning",
    "Deep Learning",
    "Overfitting"
]

# ---------------- QUESTIONS ----------------
questions = [
    "What is machine learning?",
    "What is deep learning?",
    "What is overfitting?"
]

# ---------------- IDEAL ANSWERS ----------------
ideal_answers = [
    "Machine learning is a method where systems learn from data",
    "Deep learning is a subset of machine learning using neural networks",
    "Overfitting occurs when a model learns training data too well and fails on new data"
]

# ---------------- STUDENT ANSWERS ----------------
student_answers = [
    "Machine learning allows systems to learn from data automatically",
    "Deep learning uses neural networks",
    "I like cricket"
]

# ---------------- TRACKERS ----------------
total_score = 0
weak_answers = 0

strong_topics = []
weak_topics = []

# ---------------- EVALUATION ----------------
for i in range(len(questions)):

    # BERT similarity
    emb = model.encode([ideal_answers[i], student_answers[i]])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]

    # Weak answer detection
    if sim < 0.4:
        weak_answers += 1

    # Topic analysis
    if sim >= 0.7:
        strong_topics.append(topics[i])

    elif sim < 0.4:
        weak_topics.append(topics[i])

    # Result category
    result = give_score(sim)

    # Confidence level
    conf = confidence_level(student_answers[i])

    # Feedback
    feedback = give_feedback(ideal_answers[i], student_answers[i])

    # Output per question
    print(f"\nQ{i+1}: {questions[i]}")
    print("Similarity:", round(sim, 2))
    print("Result:", result)
    print("Confidence:", conf)
    print("Missing Important Keywords:", feedback)

    total_score += sim

# ---------------- FINAL SCORE ----------------
average = total_score / len(questions)

print("\nBefore Penalty:", round(average, 2))

# Penalty for weak answers
average -= (0.2 * weak_answers)

average = max(0, average)

print("Weak Answers Detected:", weak_answers)

# ---------------- FINAL REPORT ----------------
print("\n===== INTERVIEW REPORT =====")

print("Final Average Score:", round(average, 2))
print("Final Result:", give_score(average))

print("\nStrong Topics:", strong_topics)
print("Weak Topics:", weak_topics)

print("\nRecommendation:", recommendation(average)) 