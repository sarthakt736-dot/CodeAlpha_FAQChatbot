# 🎓 College & Internship FAQ Chatbot

An intelligent chatbot that answers common college and internship-related questions using NLP techniques and cosine similarity matching.

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/sxrthxk10/CodeAlpha-FAQChatbot)

## 📌 Project Overview
This project was built as **Task 2** of the **CodeAlpha Artificial Intelligence Internship**. The chatbot collects a set of FAQs related to internships and colleges, preprocesses user queries using NLP, and matches them to the most relevant FAQ using TF-IDF and cosine similarity.

## ✨ Features
- Natural language question understanding
- Text preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization for question matching
- Cosine similarity-based best-match retrieval
- Clean and interactive chat interface using Gradio
- Covers topics: internship applications, certificates, resumes, LORs, durations, and more

## 🛠️ Tech Stack
- **Python**
- **NLTK** – text preprocessing & lemmatization
- **Scikit-learn** – TF-IDF Vectorizer & Cosine Similarity
- **Gradio** – chat interface

## ⚙️ How It Works
1. User types a question in natural language
2. The input is preprocessed (lowercased, punctuation removed, tokenized, lemmatized, stopwords removed)
3. The processed input is converted into a TF-IDF vector
4. Cosine similarity is calculated against all FAQ questions
5. The answer corresponding to the highest similarity score is returned
6. If similarity is too low, a fallback message is shown

## 📂 Files
- `app.py` – Main application code (FAQ data, preprocessing, matching logic, Gradio UI)
- `requirements.txt` – Python dependencies

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python app.py
```

## 👤 Author
**Sarthak Tiwari**
CodeAlpha AI Internship — Task 2

## 🙏 Acknowledgment
Thanks to **CodeAlpha** for this internship opportunity.

#CodeAlpha #AIInternship #NLP #Python #Chatbot #MachineLearning
