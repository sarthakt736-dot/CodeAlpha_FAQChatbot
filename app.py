import nltk
import string
import gradio as gr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

faq_data = {
    "How do I apply for an internship?": "You can apply for internships through your college placement cell, LinkedIn, Internshala, or directly on company websites by submitting your resume and cover letter.",
    "What documents are required for an internship?": "Generally you need a resume, cover letter, college ID card, offer letter (if applying for credits), and sometimes a No Objection Certificate (NOC) from your college.",
    "How many internships can I do during my degree?": "Most colleges allow students to do multiple internships, but it's best to check with your department regarding credit requirements and time limits.",
    "Is the internship paid or unpaid?": "It depends on the company. Many internships offer a stipend, while some startups or NGOs may offer unpaid internships for experience and certification.",
    "Do I get a certificate after completing the internship?": "Yes, most companies provide a completion certificate after you successfully finish all required tasks and submissions.",
    "What is the duration of a typical internship?": "Internship durations vary from 4 weeks to 6 months, depending on the company and the type of internship (summer, part-time, or full-time).",
    "Can I do an internship online/remotely?": "Yes, many companies now offer remote or virtual internships, especially in fields like software development, data analysis, and digital marketing.",
    "How do I write a good resume for internships?": "Keep it to one page, highlight relevant skills and projects, include your education, certifications, and any prior experience. Use clear formatting and avoid spelling errors.",
    "What skills are important for an AI/ML internship?": "Key skills include Python programming, knowledge of libraries like NumPy, Pandas, Scikit-learn, understanding of machine learning algorithms, and basic statistics.",
    "How can I improve my chances of getting selected?": "Build strong projects, maintain a good GitHub profile, prepare for technical interviews, network on LinkedIn, and tailor your resume to each job description.",
    "What is the difference between an internship and a job?": "An internship is usually temporary and focused on learning and gaining experience, often with mentorship, while a job is a long-term position with full responsibilities.",
    "Do internships count towards my degree credits?": "Many colleges offer academic credit for internships if they meet certain duration and documentation requirements. Check with your department coordinator.",
    "What should I do if I face issues during my internship?": "Communicate with your mentor or supervisor first. If issues persist, reach out to your college placement cell or internship coordinator for support.",
    "How do I get a Letter of Recommendation (LOR)?": "LORs are usually given based on your performance during the internship. Maintain good communication, complete tasks on time, and request it formally near the end of your internship.",
    "What happens if I don't complete all internship tasks?": "Most programs require completion of a minimum number of tasks (e.g., 2 or 3 out of 4) to be eligible for a certificate. Incomplete submissions may result in no certificate being issued."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def get_response(user_input, history=None):
    if not user_input.strip():
        return "Please type a question! 😊"
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, question_vectors)
    best_match_idx = similarities.argmax()
    best_score = similarities[0][best_match_idx]
    if best_score < 0.2:
        return "🤔 Sorry, I don't have an answer for that. Try asking about internships, resumes, certificates, or applications!"
    return answers[best_match_idx]

with gr.Blocks(title="🎓 College/Internship FAQ Chatbot — CodeAlpha") as app:
    gr.Markdown("# 🎓 College & Internship FAQ Chatbot")
    gr.Markdown("### CodeAlpha AI Internship | Sarthak Tiwari")
    gr.Markdown("Ask me anything about internships, resumes, certificates, applications, and more!")
    chatbot = gr.ChatInterface(
        fn=get_response,
        examples=[
            "How do I apply for an internship?",
            "Will I get a certificate?",
            "What skills are needed for AI internship?",
            "Can I do internship online?",
            "What is the difference between internship and job?"
        ],
        title="",
    )

app.launch()