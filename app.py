import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from datetime import datetime
import nltk
import re

# -------------------------------------------------
# DOWNLOAD NLTK STOPWORDS
# -------------------------------------------------
nltk.download('stopwords')

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Interview Evaluator",
    page_icon="🚀",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------------------------------
# STOPWORDS
# -------------------------------------------------
custom_stopwords = {
    "using", "use", "used", "also", "like"
}

stop_words = set(stopwords.words('english')).union(custom_stopwords)

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------

def give_score(similarity):

    if similarity >= 0.8:
        return "Excellent Performance"

    elif similarity >= 0.65:
        return "Good Performance"

    elif similarity >= 0.4:
        return "Average Performance"

    else:
        return "Poor Performance"


def confidence_level(answer, similarity):

    length = len(answer.split())

    if similarity >= 0.75 and length > 10:
        return "High Confidence"

    elif similarity >= 0.5 and length > 5:
        return "Medium Confidence"

    else:
        return "Low Confidence"


def recommendation(avg):

    if avg >= 0.8:
        return "🌟 Highly Recommended"

    elif avg >= 0.6:
        return "✅ Recommended"

    elif avg >= 0.4:
        return "⚠️ Needs Improvement"

    else:
        return "❌ Not Recommended"


# -------------------------------------------------
# CLEAN TEXT
# -------------------------------------------------

def get_keywords(text):

    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    cleaned_words = [
        word for word in words
        if word not in stop_words
    ]

    return set(cleaned_words)


# -------------------------------------------------
# FEEDBACK
# -------------------------------------------------

def give_feedback(ideal, student):

    ideal_words = get_keywords(ideal)

    student_words = get_keywords(student)

    missing = ideal_words - student_words

    return list(missing)[:5]


# -------------------------------------------------
# PERFORMANCE MESSAGE
# -------------------------------------------------

def performance_message(avg):

    if avg >= 0.8:
        return (
            "Candidate demonstrated strong technical understanding."
        )

    elif avg >= 0.6:
        return (
            "Candidate showed good understanding with minor gaps."
        )

    elif avg >= 0.4:
        return (
            "Candidate has partial understanding and needs improvement."
        )

    else:
        return (
            "Candidate lacks sufficient technical understanding."
        )


# -------------------------------------------------
# DATA
# -------------------------------------------------

topics = [
    "Machine Learning",
    "Deep Learning",
    "Overfitting"
]

questions = [
    "What is machine learning?",

    "What is deep learning?",

    "What is overfitting?"
]

ideal_answers = [
    "Machine learning is a method where systems learn from data automatically.",

    "Deep learning is a subset of machine learning that uses neural networks and multiple layers.",

    "Overfitting occurs when a model memorizes training data and performs poorly on new unseen data."
]

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📌 About")

st.sidebar.info(
    """
AI Interview Evaluator using:

• NLP  
• BERT  
• Semantic Similarity  
• AI Feedback System  
• Topic Analysis  
• Recommendation Engine
"""
)

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------

st.title("🚀 AI Interview Evaluator")

st.write(
    "AI-powered interview analysis using NLP and BERT"
)

# -------------------------------------------------
# CANDIDATE DETAILS
# -------------------------------------------------

candidate_name = st.text_input(
    "👤 Enter Candidate Name"
)

current_time = datetime.now().strftime(
    "%d-%m-%Y %H:%M"
)

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

student_answers = []

for i, question in enumerate(questions):

    answer = st.text_area(
        f"Q{i+1}: {question}",
        height=130,
        placeholder="Type your answer here..."
    )

    student_answers.append(answer)

# -------------------------------------------------
# RESET BUTTON
# -------------------------------------------------

if st.button("🔄 Reset Answers"):
    st.rerun()

# -------------------------------------------------
# EVALUATE BUTTON
# -------------------------------------------------

if st.button("🚀 Evaluate Interview"):

    # -------------------------------------------------
    # EMPTY INPUT CHECK
    # -------------------------------------------------

    if candidate_name.strip() == "":

        st.warning(
            "Please enter candidate name."
        )

        st.stop()

    if any(ans.strip() == "" for ans in student_answers):

        st.warning(
            "Please answer all questions."
        )

        st.stop()

    # -------------------------------------------------
    # LOADING SPINNER
    # -------------------------------------------------

    with st.spinner("Evaluating Interview..."):

        total_score = 0
        weak_answers = 0

        strong_topics = []
        weak_topics = []

        st.header("📋 Interview Analysis")

        # -------------------------------------------------
        # QUESTION LOOP
        # -------------------------------------------------

        for i in range(len(questions)):

            embeddings = model.encode([
                ideal_answers[i],
                student_answers[i]
            ])

            similarity = cosine_similarity(
                [embeddings[0]],
                [embeddings[1]]
            )[0][0]

            total_score += similarity

            # -------------------------------------------------
            # WEAK ANSWER DETECTION
            # -------------------------------------------------

            if similarity < 0.4:
                weak_answers += 1

            # -------------------------------------------------
            # TOPIC ANALYSIS
            # -------------------------------------------------

            if similarity >= 0.7:

                strong_topics.append(topics[i])

            elif similarity < 0.4:

                weak_topics.append(topics[i])

            # -------------------------------------------------
            # FEEDBACK
            # -------------------------------------------------

            feedback = give_feedback(
                ideal_answers[i],
                student_answers[i]
            )

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            confidence = confidence_level(
                student_answers[i],
                similarity
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            result = give_score(similarity)

            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            st.subheader(f"Q{i+1} Result")

            st.write(f"**Topic:** {topics[i]}")

            st.write(
                f"**Similarity Score:** {similarity:.2f}"
            )

            # -------------------------------------------------
            # RESULT COLORS
            # -------------------------------------------------

            if similarity >= 0.8:

                st.success(f"Result: {result}")

            elif similarity >= 0.4:

                st.warning(f"Result: {result}")

            else:

                st.error(f"Result: {result}")

            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            st.write(
                f"**Confidence:** {confidence}"
            )

            # -------------------------------------------------
            # PROGRESS BAR
            # -------------------------------------------------

            progress_value = max(
                0,
                min(int(similarity * 100), 100)
            )

            st.progress(progress_value)

            # -------------------------------------------------
            # FEEDBACK DISPLAY
            # -------------------------------------------------

            if feedback:

                st.write(
                    "**Missing Important Keywords:** " +
                    ", ".join(feedback)
                )

            else:

                st.write(
                    "**Missing Important Keywords:** None"
                )

            st.divider()

        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        average = total_score / len(questions)

        # -------------------------------------------------
        # PENALTY
        # -------------------------------------------------

        average -= (0.2 * weak_answers)

        average = max(0, average)

        # -------------------------------------------------
        # FINAL REPORT
        # -------------------------------------------------

        st.header("📊 Final Interview Report")

        # FINAL PROGRESS BAR
        st.progress(
            max(
                0,
                min(int(average * 100), 100)
            )
        )

        # PERFORMANCE %
        score_percent = int(average * 100)

        st.write(
            f"### Candidate Name: {candidate_name}"
        )

        st.write(
            f"### Evaluation Time: {current_time}"
        )

        st.write(
            f"### Overall Performance: {score_percent}%"
        )

        st.write(
            f"### Final Average Score: {average:.2f}"
        )

        final_result = give_score(average)

        st.write(
            f"### Final Result: {final_result}"
        )

        # -------------------------------------------------
        # PERFORMANCE MESSAGE
        # -------------------------------------------------

        st.info(
            performance_message(average)
        )

        # -------------------------------------------------
        # FINAL STATUS
        # -------------------------------------------------

        if average >= 0.8:

            st.success("🌟 Excellent Candidate")

        elif average >= 0.6:

            st.info("👍 Good Candidate")

        elif average >= 0.4:

            st.warning("⚠️ Average Candidate")

        else:

            st.error("❌ Weak Candidate")

        # -------------------------------------------------
        # WEAK ANSWERS
        # -------------------------------------------------

        st.write(
            f"### Weak Answers Detected: {weak_answers}"
        )

        # -------------------------------------------------
        # STRONG TOPICS
        # -------------------------------------------------

        if strong_topics:

            st.write(
                "### Strong Topics: " +
                ", ".join(strong_topics)
            )

        else:

            st.write(
                "### Strong Topics: None"
            )

        # -------------------------------------------------
        # WEAK TOPICS
        # -------------------------------------------------

        if weak_topics:

            st.write(
                "### Weak Topics: " +
                ", ".join(weak_topics)
            )

        else:

            st.write(
                "### Weak Topics: None"
            )

        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        final_recommendation = recommendation(
            average
        )

        if "Highly" in final_recommendation:

            st.success(
                f"Recommendation: {final_recommendation}"
            )

        elif "Recommended" in final_recommendation:

            st.info(
                f"Recommendation: {final_recommendation}"
            )

        elif "Improvement" in final_recommendation:

            st.warning(
                f"Recommendation: {final_recommendation}"
            )

        else:

            st.error(
                f"Recommendation: {final_recommendation}"
            )

        # -------------------------------------------------
        # AI SUMMARY
        # -------------------------------------------------

        st.markdown("---")

        st.subheader("🧠 AI Evaluation Summary")

        st.info(
            f"""
Candidate Name:
{candidate_name}

Strong Areas:
{', '.join(strong_topics) if strong_topics else 'None'}

Areas Needing Improvement:
{', '.join(weak_topics) if weak_topics else 'None'}

Final Hiring Decision:
{final_recommendation}
"""
        )

        # -------------------------------------------------
        # DOWNLOAD REPORT
        # -------------------------------------------------

        report = f"""
AI INTERVIEW REPORT
=============================

Candidate Name:
{candidate_name}

Evaluation Time:
{current_time}

Overall Performance:
{score_percent}%

Final Average Score:
{average:.2f}

Final Result:
{final_result}

Weak Answers Detected:
{weak_answers}

Strong Topics:
{', '.join(strong_topics) if strong_topics else 'None'}

Weak Topics:
{', '.join(weak_topics) if weak_topics else 'None'}

Recommendation:
{final_recommendation}

Performance Summary:
{performance_message(average)}
"""

        st.download_button(
            label="📥 Download Interview Report",
            data=report,
            file_name="AI_Interview_Report.txt",
            mime="text/plain"
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit + NLP + BERT"
)
#streamlit run app.py
#& ".\.venv\Scripts\Activate.ps1"