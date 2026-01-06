import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
import pandas as pd
import datetime
import speech_recognition as sr
import re 
import os
import logging
import io

import sys
import traceback

from dotenv import load_dotenv
import traceback
import json

from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from babel.numbers import format_currency

load_dotenv(dotenv_path='.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = 'chroma'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

PROMPT_TEMPLATE = """
Answer the question so that it is easily understandable. The context is provided so that you can take reference from this. Please take inspiration from the context. You can also add things that you think are helpful for girls out there. Do not mention about the context provided. Answer as you usually answer.

{context}

---

{question}
"""

INTRODUCTION_PROMPTS = [
    r'introduce yourself',
    r'tell me about yourself',
    r'who are you',
    r'what can you do',
    r'how can you help me',
    r'what are your capabilities',
    r'what kind of tasks can you assist with',
    r'what are you capable of',
    r'what can i ask you',
    r'what are you good at?',
    r'what are your specialties?',
    r'what is your purpose?',
    r'what are you designed for?',
    r'how do i get started with you?',
    r'how should i interact with you?',
    r'who created you'
    r'hello',
    r'hey',
    r'namaste'
]

# Validate that .env file exists and its structure is correct
def check_env_file():
    logger.info("=== Checking .env File ===")
    if not os.path.exists('.env'):
        logger.warning(".env file does not exist. Assuming secrets are set via environment variables.")
        return

    with open('.env', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            if '=' not in line:
                logger.error(f"Invalid line in .env file: {line}")
                raise ValueError(f"Invalid line in .env file: {line}")
        logger.info("Valid .env file structure.")

def main():
    st.set_page_config(page_title="Financial Advisor for Your Dreams", page_icon=":moneybag:", layout="wide")

    # Diagnostic logging
    logger.info("=== Starting Application Diagnostics ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Google Generative AI package version: {genai.__version__}")
    logger.info(f"Running on Streamlit version: {st.__version__}")
    
    # Log environment info
    logger.info("=== Environment Check ===")
    is_cloud = os.getenv('STREAMLIT_DEPLOYMENT_RUNTIME') == 'cloud'
    logger.info(f"Running on Streamlit Cloud: {is_cloud}")
    
    # Check for proxy-related environment variables
    logger.info("=== Proxy Configuration Check ===")
    proxy_vars = {k: '***' for k, v in os.environ.items() 
                 if 'PROXY' in k.upper() or 'HTTP' in k.upper()}
    logger.info(f"Proxy-related environment variables: {proxy_vars}")

    # Check .env file
    check_env_file()

    # Create a navigation bar
    menu = ["Start with Voice", "Build your Wealth", "Savings and Budgeting"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Create the Gemini client
    try:
        logger.info("Attempting to configure Gemini...")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        client = genai.GenerativeModel('gemini-2.0-flash-001')
        
        logger.info("Successfully created Gemini client")
        st.sidebar.success("AI assistant (Gemini) initialized successfully!")
        
    except Exception as e:
        logger.error("=== Gemini Client Error ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error args: {e.args}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        st.sidebar.error(f"Failed to initialize the AI assistant. Error: {str(e)}")
        return

    # Display the selected page
    if choice == "Start with Voice":
        home_page(client)
    elif choice == "Build your Wealth":
        templates_page(client)
    elif choice == "Savings and Budgeting":
        expense_tracker_page()

def home_page(client):
    st.title("DhanKanya: Financial Empowerment for Girls in India")

    # Center the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=200)

    st.write("""
    ### Welcome to our AI-powered financial literacy application!

    Our mission is to empower girls in India with the knowledge and tools they need to achieve financial independence and success.

    With our user-friendly app, you'll have access to:
    """)

    st.write("- **Interactive Budgeting Tools** to help you track your income and expenses.")
    st.write("- **Educational Resources** on essential financial literacy concepts like saving and investing.")
    st.write("- **Goal Setting Functionality** to plan and save for specific educational milestones.")

    st.markdown("---")

    if "claude_model" not in st.session_state:
        st.session_state["claude_model"] = "claude-3-haiku-20240307"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add voice input functionality
    st.write("You can ask questions in Hindi using your voice or type them in English.")
    
    # Get inputs from both sources
    voice_prompt = get_voice_input()
    chat_prompt = st.chat_input("Ask a question in English")
    
    # Prioritize voice input if available
    if voice_prompt:
        prompt = voice_prompt
    else:
        prompt = chat_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = get_response(prompt, client)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def load_templates():
    try:
        with open('state_templates.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("state_templates.json file not found")
        return {}
    except json.JSONDecodeError:
        logger.error("Error decoding state_templates.json")
        return {}

def templates_page(client):
    st.title("Sample Prompts depending on the state that you reside in.")
    
    # Templates for different states
    templates  = load_templates()
    
    states = list(templates.keys())
    if not states:
        st.error("No templates available. Please check the configuration file.")
        return

    selected_state = st.selectbox("Select your state in India", states)

    if selected_state:
        st.write(f"Selected State: {selected_state}")
        st.write(f"Choose from the following prompt templates related to {selected_state}:")
        
        state_templates = templates.get(selected_state, {})
        for category, prompts in state_templates.items():
            st.subheader(category)
            for prompt in prompts:
                if st.button(prompt):
                    response = get_response(prompt, client)
                    st.markdown(response)

    logger.info(f"User selected state: {selected_state}")

def initialize_session():
    """Initialize session state variables."""
    if 'savings_goal' not in st.session_state:
        st.session_state.savings_goal = 0
    if 'monthly_target' not in st.session_state:
        st.session_state.monthly_target = 0
    if 'earnings' not in st.session_state:
        st.session_state.earnings = []
    if 'expenses' not in st.session_state:
        st.session_state.expenses = []
    if 'saved_amount' not in st.session_state:
        st.session_state.saved_amount = 0  # Track updated savings instantly

def update_savings():
    """Update savings progress immediately."""
    total_earnings = sum(e["amount"] for e in st.session_state.earnings)
    total_expenses = sum(e["amount"] for e in st.session_state.expenses)
    st.session_state.saved_amount = total_earnings - total_expenses

def format_inr(amount):
    return format_currency(amount, 'INR', locale='en_IN')

def expense_tracker_page():
    initialize_session()
    
    # Page title
    st.title("💰 Goal-Oriented Expense Tracker")

    # Savings Goal Section
    st.subheader("Your Savings Goal")
    st.session_state.savings_goal = st.number_input("Total Savings Goal (₹)", min_value=0, step=1000)
    st.session_state.monthly_target = st.number_input("Monthly Savings Target (₹)", min_value=0, step=500)

    if st.session_state.savings_goal > 0 and st.session_state.monthly_target > 0:
        months_required = st.session_state.savings_goal / st.session_state.monthly_target
        months_required = int(months_required) + 1 if months_required % 1 > 0 else int(months_required)
        st.markdown(f"*Estimated time to achieve goal: **{months_required} months***")

    # Progress Calculation
    progress = min(st.session_state.saved_amount / st.session_state.savings_goal, 1.0) if st.session_state.savings_goal > 0 else 0

    st.subheader("Savings Progress")
    st.progress(progress)
    pm_col1, pm_col2, pm_col3 = st.columns(3)
    pm_col1.metric("Saved Amount", format_inr(st.session_state.saved_amount))
    pm_col2.metric("More to Save", format_inr(max(st.session_state.savings_goal - st.session_state.saved_amount, 0)))
    pm_col3.metric("Total Target", format_inr(st.session_state.savings_goal))

    # Earnings and Expenses Input Side by Side
    expns_col1, expns_col2 = st.columns(2)
    with expns_col1:
        st.subheader("Add Expenses")
        with st.form("expenses_form"):
            exp_amount = st.number_input("Expense Amount (₹)", min_value=0, step=100)
            exp_desc = st.text_input("Expense Description")
            exp_category = st.selectbox("Category", ["Rent", "Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"])
            exp_avoidable = st.checkbox("Is this expense avoidable?")
            exp_submit = st.form_submit_button("Add Expense")
            if exp_submit and exp_desc and exp_amount > 0:
                st.session_state.expenses.append({
                    "date": datetime.date.today(),
                    "desc": exp_desc,
                    "amount": exp_amount,
                    "category": exp_category,
                    "avoidable": exp_avoidable
                })
                update_savings()  # Instantly update savings
                st.rerun()  # Force UI update

    with expns_col2:
        st.subheader("Add Earnings")
        with st.form("earnings_form"):
            e_amount = st.number_input("Earning Amount (₹)", min_value=0, step=100)
            e_desc = st.text_input("Earning Description")
            e_category = st.selectbox("Category", ["Scholarship", "Bonus", "Gift", "Investment Return", "Other"])
            e_submit = st.form_submit_button("Add Earning")
            if e_submit and e_desc and e_amount > 0:
                st.session_state.earnings.append({
                    "date": datetime.date.today(),
                    "desc": e_desc,
                    "amount": e_amount,
                    "category": e_category
                })
                update_savings()  # Instantly update savings
                st.rerun()  # Force UI update

    # Summary Section
    st.subheader("Financial Summary")
    total_earnings = sum(e["amount"] for e in st.session_state.earnings)
    total_expenses = sum(e["amount"] for e in st.session_state.expenses)
    total_avoidable_expenses = sum(e["amount"] for e in st.session_state.expenses if e["avoidable"])

    fs_col1, fs_col2, fs_col3 = st.columns(3)
    fs_col1.metric("Total Earnings", format_inr(total_earnings))
    fs_col2.metric("Total Expenses", format_inr(total_expenses))
    fs_col3.metric("Avoidable Expenses", format_inr(total_avoidable_expenses))

    # Ledger Table
    st.subheader("📜 Earnings & Expenses Ledger")
    ledger_data = [
        {**e, "type": "Earning", "avoidable": "-"} for e in st.session_state.earnings
    ] + [
        {**e, "type": "Expense"} for e in st.session_state.expenses
    ]

    ledger_df = pd.DataFrame(ledger_data)
    if not ledger_df.empty:
        ledger_df["avoidable"] = ledger_df.get("avoidable", "-")
        ledger_df = ledger_df.sort_values(by="date", ascending=False)
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
    else:
        st.info("No earnings or expenses recorded yet.")

def get_voice_input():
    st.write("Click the microphone to start recording:")
    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="🛑 Stop Recording",
        just_once=True,
        use_container_width=False,
        key='voice_input'
    )

    if audio:
        status = st.status("Processing voice input...", expanded=True)
        try:
            audio_bytes = audio['bytes']
            # Convert audio bytes to an AudioFile compatible format
            audio_io = io.BytesIO(audio_bytes)
            
            r = sr.Recognizer()
            with sr.AudioFile(audio_io) as source:
                audio_data = r.record(source)

            status.write("Transcribing audio...")
            
            # Try English first, then Hindi
            recognized_text = None
            try:
                recognized_text = r.recognize_google(audio_data, language='en-IN')
            except sr.UnknownValueError:
                try:
                    recognized_text = r.recognize_google(audio_data, language='hi-IN')
                except sr.UnknownValueError:
                    pass

            if recognized_text:
                status.update(label="✅ Voice recognized!", state="complete")
                return recognized_text
            else:
                status.update(label="❌ Could not understand audio.", state="error")
                return None

        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")
            return None
    
    return None


def query(query_text, client):
    try:
        # Prepare the DB with HuggingFaceBgeEmbeddings.
        embedding_function = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            logger.info("Direct query to Claude")
            answer = get_response(query_text, client)
            return answer

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        logger.info("Query with context")
        response = get_response(prompt, client)
        return response
    except Exception as e:
        logger.error(f"Error in query function: {e}")
        logger.error(traceback.format_exc())
        return "I'm sorry, but I encountered an error while processing your query. Please try again later."

def get_response(prompt, client):
    """
    Retrieves response from the Anthropic model based on the prompt.
    """
    try:
        prompt_lower = prompt.lower()
        for pattern in INTRODUCTION_PROMPTS:
            if re.search(pattern, prompt_lower):
                return "Namaste! I'm your financial assistant, developed by the Finance team at 100GIGA and powered by Google's Gemini AI model. My purpose is to provide you with expert financial guidance, enhancing your financial literacy and addressing your needs. Feel free to ask me anything related to finance, and I'll be here to assist you every step of the way."
        
        # Call Gemini API
        response = client.generate_content(prompt)
        return response.text

    except Exception as e:
        logger.error(f"Error getting response from Claude: {e}")
        logger.error(traceback.format_exc())
        return f"I apologize, but I'm having trouble generating a response at the moment. \nError details: {str(e)}"

if __name__ == "__main__":
    main()