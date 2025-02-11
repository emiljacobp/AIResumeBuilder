import os

import dotenv
import spacy
import openai
from langchain.chat_models import ChatOpenAI

# Load NLP model
nlp = spacy.load("en_core_web_sm")

dotenv.load_dotenv()

# OpenAI API key (Replace with your API key)
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_skills_from_jd(jd_text):
    """Extracts skills and experience from a job description."""
    doc = nlp(jd_text)
    skills = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:  # Extract potential skills
            skills.add(token.text)
    return list(skills)

def format_resume(resume_text, jd_text):
    """Formats the resume based on JD requirements using AI."""
    chat_model = ChatOpenAI()
    prompt = f"""
    Given the following job description:
    {jd_text}
    
    And the following resume:
    {resume_text}
    
    Please rewrite the resume, emphasizing relevant skills and experience from the job description while keeping the professional tone intact.
    """
    response = chat_model.predict(prompt)
    return response

# Example Usage
jd_text = """
Looking for a Full Stack Developer with experience in Python, AI, and cloud technologies.
Must have strong knowledge of NLP, data processing, and API development.
"""

resume_text = """
Experienced software engineer with expertise in web development and backend systems.
Proficient in various programming languages and frameworks.
"""

# Extract skills from JD
extracted_skills = extract_skills_from_jd(jd_text)
print("Extracted Skills:", extracted_skills)

# Format resume
formatted_resume = format_resume(resume_text, jd_text)
print("\nFormatted Resume:\n", formatted_resume)
