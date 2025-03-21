import os
import streamlit as st
import logging
import tempfile
import re
import json
import threading
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from datetime import datetime, timedelta

# Lock for concurrency
LOCK = threading.Lock()

# Load and save user credentials
def load_users():
    return st.session_state.get("users", {})

def save_users(users):
    st.session_state["users"] = users

# Load and save usage data (session-only)
def load_usage_data():
    if "usage_data" not in st.session_state:
        st.session_state["usage_data"] = {
            "last_reset": datetime.now(),
            "request_count": 0,
            "user_requests": {}
        }
    data = st.session_state["usage_data"]
    if isinstance(data["last_reset"], str):
        data["last_reset"] = datetime.fromisoformat(data["last_reset"])
    if datetime.now() - data["last_reset"] > timedelta(days=1):
        data["last_reset"] = datetime.now()
        data["request_count"] = 0
        data["user_requests"] = {}
    return data

def save_usage_data(data):
    st.session_state["usage_data"] = data

# Load API keys from Streamlit secrets
API_KEYS = st.secrets["grok_api_keys"]["keys"]
if "current_api_key_idx" not in st.session_state:
    st.session_state["current_api_key_idx"] = 0

# Session state initialization
if "initialized" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.user_api_key = None
    st.session_state.initialized = True

# Set page config
st.set_page_config(page_title="NIPGR Research Article Chatbot", layout="wide", page_icon="📖", initial_sidebar_state="expanded")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting NIPGR Research Article Chatbot")

# Embeddings and model setup
@st.cache_resource
def get_embeddings():
    logger.info("Initializing embeddings")
    emb = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})
    logger.info("Embeddings initialized successfully")
    return emb

@st.cache_resource
def get_qa_model(api_key_idx=0):
    logger.info(f"Initializing ChatGroq model with API key index {api_key_idx}")
    try:
        api_key = API_KEYS[api_key_idx]
        model = ChatGroq(api_key=api_key, model="llama-3.2-90b-vision-preview", temperature=0)
        logger.info("ChatGroq model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"API key {api_key_idx} failed: {str(e)}")
        if api_key_idx + 1 < len(API_KEYS):
            st.session_state["current_api_key_idx"] += 1
            return get_qa_model(st.session_state["current_api_key_idx"])
        else:
            raise Exception("All API keys exhausted")

embeddings = get_embeddings()
qa_model = get_qa_model(st.session_state["current_api_key_idx"])

# Prompt template
template = """[SYSTEM]
You are a research paper QA assistant for NIPGR. Follow these rules STRICTLY:
1. Answer ONLY using the provided context
2. Provide comprehensive, detailed answers without size restrictions
3. For numerical questions, verify exact numbers from context and cite the page number
4. For methodology questions, quote relevant text directly and include page numbers
5. If the answer is not in context, say "Not found in document" unless it's an author question
6. Use Markdown formatting for better readability
7. For author-related questions (e.g., "Who are the authors of the paper?" or "Who wrote this?" or "Who wrote the review paper?"):
   - Prioritize context from the first page or metadata
   - If not found, provide a helpful response suggesting how to find the authors using citation info (e.g., DOI, journal)
8. Give answers like a real scientist describing the answer with correct data from the provided context in a detailed manner.
[RESPONSE FORMAT]
**Based on the research paper:**
- Provide detailed explanation with relevant context
- Use bullet points for multiple findings
- Include direct quotes with page numbers when appropriate
- Highlight important terms in **bold**

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
"""
prompt = PromptTemplate.from_template(template)

# Define functions
def extract_metadata(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        pdf_path = tmp_file.name
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        first_page = pages[0].page_content if pages else ""
        metadata = pages[0].metadata

        title_from_metadata = metadata.get('/Title', '').strip()
        if title_from_metadata and len(title_from_metadata) > 10 and not re.search(r'(untitled|document)', title_from_metadata, re.I):
            title = title_from_metadata
        else:
            title = None

        lines = first_page.split('\n')
        title_candidates = []
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if not line or re.search(r'^(abstract|introduction|keywords|\d+\s*$)', line, re.I):
                break
            if len(line) > 20 and re.match(r'^[A-Z][A-Za-z0-9 :,\-–—]{10,}$', line):
                title_candidates.append(line)
                if i + 1 < len(lines) and len(lines[i + 1].strip()) > 10 and not re.search(r'(by|and|,|\d)', lines[i + 1], re.I):
                    title_candidates.append(lines[i + 1].strip())
                break

        if not title_candidates:
            title_match = re.search(r'^(.+?)\n(?=\s*(By|Abstract|Introduction))', first_page, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else "Untitled"
        else:
            title = " ".join(title_candidates)

        authors = metadata.get('/Author', '').strip()
        if not authors or authors.lower() in ['unknown', '']:
            authors_match = re.search(r'(?i)(?:By|Authors?:)\s+(.+?)(?:\n|$|\s*(?:Abstract|Introduction|\d))', first_page, re.DOTALL)
            authors = authors_match.group(1).strip() if authors_match else "Unknown Authors"
            authors = re.sub(r'\n\s*', ', ', authors).strip()

        doi_match = re.search(r'(?i)DOI:\s*(10\.\d{4,}/[^\s]+)', first_page)
        doi = doi_match.group(1) if doi_match else None
        journal_match = re.search(r'(?i)(WIREs\s+\w+|Journal\s+of\s+[\w\s]+|[\w\s]+Journal)', first_page)
        journal = journal_match.group(0) if journal_match else None

        return title, authors, doi, journal
    except Exception as e:
        logger.error(f"Metadata extraction error: {str(e)}")
        return "Untitled", "Unknown Authors", None, None
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        pdf_path = tmp_file.name
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
        )
        chunks = text_splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata['page'] += 1
        vectorstore = DocArrayInMemorySearch.from_documents(chunks, embedding=embeddings)
        return pages, vectorstore
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# Interactive CSS
st.markdown("""
    <style>
    .header { font-size: 40px !important; color: #1A73E8 !important; padding: 20px 0; border-bottom: 4px solid #1A73E8; margin-bottom: 30px; text-align: center; animation: fadeIn 1s; }
    .sidebar .sidebar-content { background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 10px; border-radius: 10px; }
    .stButton>button { background: linear-gradient(90deg, #1A73E8, #4285F4); color: white !important; border-radius: 8px; padding: 10px 20px; font-weight: bold; transition: transform 0.2s; }
    .stButton>button:hover { transform: scale(1.05); }
    .stTextInput>div>input { border: 2px solid #1A73E8; border-radius: 8px; padding: 8px; transition: border-color 0.3s; }
    .stTextInput>div>input:focus { border-color: #4285F4; box-shadow: 0 0 5px rgba(66, 133, 244, 0.5); }
    .contact-section { margin-top: 50px; padding: 20px; background: #E8F0FE; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

# App title and info
logger.info("Rendering UI")
st.title("📖 NIPGR Research Article Chatbot")
st.markdown("**Welcome to NIPGR Research Article Chatbot!** Upload research papers and ask detailed questions powered by LLaMA-3.2-90B-Vision-Preview, an advanced Large Language Model from xAI.")
st.markdown("*Developed by SKLAB at NIPGR under the leadership of Dr. Shailesh Kumar, Head of SKLAB.*")

# Login or signup
if st.session_state.user_id is None and st.session_state.user_api_key is None:
    st.subheader("Login or Signup")
    users = load_users()
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    with tab1:
        with st.form(key='login_form'):
            login_id = st.text_input("NIPGR Login ID (e.g., username@nipgr.ac.in)", key="login_id")
            password = st.text_input("Password", type="password", key="password")
            submit = st.form_submit_button(label="Login")
            if submit:
                if not login_id.endswith("@nipgr.ac.in"):
                    st.error("Login ID must end with @nipgr.ac.in")
                elif login_id in users:
                    if users[login_id] == password:
                        st.session_state.user_id = login_id
                        usage_data = load_usage_data()
                        usage_data["user_requests"].setdefault(login_id, 0)
                        save_usage_data(usage_data)
                        st.session_state[f"papers_{login_id}"] = []
                        st.session_state[f"chats_{login_id}"] = {}
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid password")
                else:
                    st.error("User not found. Please sign up.")

    with tab2:
        with st.form(key='signup_form'):
            signup_id = st.text_input("NIPGR Signup ID (e.g., username@nipgr.ac.in)", key="signup_id")
            signup_password = st.text_input("Create Password", type="password", key="signup_password")
            signup_submit = st.form_submit_button(label="Signup")
            if signup_submit:
                if not signup_id.endswith("@nipgr.ac.in"):
                    st.error("Signup ID must end with @nipgr.ac.in")
                elif signup_id in users:
                    st.error("User already exists. Please log in.")
                else:
                    users[signup_id] = signup_password
                    save_users(users)
                    st.session_state.user_id = signup_id
                    usage_data = load_usage_data()
                    usage_data["user_requests"].setdefault(signup_id, 0)
                    save_usage_data(usage_data)
                    st.session_state[f"papers_{signup_id}"] = []
                    st.session_state[f"chats_{signup_id}"] = {}
                    st.success(f"User '{signup_id}' registered successfully!")
                    st.rerun()
else:
    user_id = st.session_state.user_id
    use_api_key = bool(st.session_state.user_api_key)

    if f"papers_{user_id}" not in st.session_state:
        st.session_state[f"papers_{user_id}"] = []
    if f"chats_{user_id}" not in st.session_state:
        st.session_state[f"chats_{user_id}"] = {}

    with st.sidebar:
        st.image("https://nipgr.ac.in/images/nipgr.png", width=200)
        st.markdown("<div class='header'>NIPGR Research Article Chatbot</div>", unsafe_allow_html=True)
        st.write(f"Logged in as: **{user_id}**")
        if st.button("Logout"):
            if user_id:
                del st.session_state[f"papers_{user_id}"]
                del st.session_state[f"chats_{user_id}"]
                if "usage_data" in st.session_state:
                    del st.session_state["usage_data"]
            st.session_state.user_id = None
            st.session_state.user_api_key = None
            st.session_state.initialized = False
            st.rerun()
        if use_api_key:
            st.write("Using personal API key")
        else:
            st.text_input("Enter your Grok API key (optional):", type="password", key="api_key_sidebar", on_change=lambda: st.session_state.update(user_api_key=st.session_state.api_key_sidebar) or qa_model.__setattr__('api_key', st.session_state.api_key_sidebar) or st.rerun())
        st.markdown("""
        <div class='contact-section'>
            <h3>Contact SKLAB</h3>
            <p>National Institute of Plant Genome Research<br>
            Aruna Asaf Ali Marg, New Delhi 110067<br>
            <a href="https://nipgr.ac.in/research/dr_shailesh.php">Visit our Lab Website</a></p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Welcome to NIPGR Research Article Chatbot!**  
        Upload research papers and ask detailed questions powered by LLaMA-3.2-90B-Vision-Preview, an advanced Large Language Model from xAI.  
        Features:  
        - Multi-document analysis  
        - Citation tracking  
        - Enhanced usage limits with multiple API keys  
        *Developed by SKLAB, headed by Dr. Shailesh Kumar, NIPGR.*
        """)
    with col2:
        usage_data = load_usage_data()
        remaining_shared = max(0, 1000 - usage_data["request_count"]) if not use_api_key else "Unlimited (API Key)"
        remaining_user = max(0, 50 - usage_data["user_requests"].get(user_id, 0)) if user_id else "N/A"
        st.metric("Shared Queries Left", remaining_shared)
        st.metric("Your Queries Left", remaining_user)
        st.write(f"Debug: Shared Count = {usage_data['request_count']}, User Count = {usage_data['user_requests'].get(user_id, 0)}")

    if user_id and remaining_user == 0:
        st.warning("Your personal limit (50/day) reached. Use your own API key or wait for reset.")
    elif not use_api_key and remaining_shared == 0:
        st.warning("Shared limit (1000/day) reached. Please use your own API key.")
    else:
        uploaded_files = st.file_uploader("Upload PDFs (Max 3)", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            if len(uploaded_files) > 3:
                st.error("Maximum 3 files allowed.")
            else:
                for file in uploaded_files:
                    file_data = file.getvalue()
                    user_papers = st.session_state[f"papers_{user_id}"]
                    if file.name not in [p['filename'] for p in user_papers]:
                        try:
                            pages, vectorstore = process_pdf(file_data)
                            title, authors, doi, journal = extract_metadata(file_data)
                            user_papers.append({
                                "title": title,
                                "authors": authors,
                                "filename": file.name,
                                "vectorstore": vectorstore,
                                "pages": pages,
                                "doi": doi,
                                "journal": journal
                            })
                            st.session_state[f"papers_{user_id}"] = user_papers
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")

        user_papers = st.session_state[f"papers_{user_id}"]
        if user_papers:
            st.success(f"Loaded {len(user_papers)} papers successfully!")
            selected_paper = st.selectbox("Select a paper:", options=[p['title'] for p in user_papers], format_func=lambda x: f"{x[:40]}...")
            paper_idx = next(idx for idx, p in enumerate(user_papers) if p['title'] == selected_paper)
            paper = user_papers[paper_idx]

            user_chats = st.session_state[f"chats_{user_id}"]
            if paper_idx not in user_chats:
                user_chats[paper_idx] = []
                st.session_state[f"chats_{user_id}"] = user_chats

            with st.expander("Conversation History", expanded=True):
                for chat in user_chats[paper_idx]:
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.markdown("---")

            st.markdown("**Need help with prompts?** Try these sample questions for accurate answers:")
            sample_questions = [
                "What is the summary of the article?",
                "What is the methodology mentioned in the article?",
                "What are the key findings of the study?",
                "Who are the authors of the research article?",
                "What data sources were used in the research article?"
            ]
            for q in sample_questions:
                st.markdown(f"- *{q}*")
            st.markdown("*Tip:* Specific, clear questions yield the best results!")

            question = st.text_input("Ask a question about the paper:", key=f"q_{user_id}_{paper_idx}")
            if st.button("Ask"):
                with st.spinner("Processing..."):
                    try:
                        k = 10 if "author" in question.lower() else 5
                        docs = paper['vectorstore'].similarity_search(question, k=k)
                        if "author" in question.lower():
                            docs = sorted(docs, key=lambda x: x.metadata['page'])[:k]
                        context = "\n\n".join(f"Page {doc.metadata['page']}: {doc.page_content}" for doc in docs if doc.page_content.strip())
                        raw_response = qa_model.invoke(prompt.format(context=context, question=question))
                        clean_response = raw_response.content.strip()

                        if "author" in question.lower() and "Not found in document" in clean_response:
                            fallback_response = "**Based on the research paper:**\n\nUnfortunately, the authors of the research article are not explicitly mentioned in the provided context."
                            if paper['doi'] or paper['journal']:
                                fallback_response += "\n\nHowever, the citation information provided on Page 1 mentions "
                                if paper['doi']:
                                    fallback_response += f"the DOI (**{paper['doi']}**)"
                                    if paper['journal']:
                                        fallback_response += f" and the journal (**{paper['journal']}**)"
                                elif paper['journal']:
                                    fallback_response += f"the journal (**{paper['journal']}**)"
                                fallback_response += ", which can be used to look up the article and find the authors."
                            clean_response = fallback_response

                        usage_data = load_usage_data()
                        if not use_api_key:
                            usage_data["request_count"] += 1
                            logger.info(f"Updated shared count: {usage_data['request_count']}")
                        if user_id:
                            usage_data["user_requests"][user_id] = usage_data["user_requests"].get(user_id, 0) + 1
                            logger.info(f"Updated user count for {user_id}: {usage_data['user_requests'][user_id]}")
                        save_usage_data(usage_data)

                        user_chats[paper_idx].append({"question": question, "answer": clean_response})
                        st.session_state[f"chats_{user_id}"] = user_chats
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                        if "rate limit" in str(e).lower() and st.session_state["current_api_key_idx"] + 1 < len(API_KEYS):
                            st.session_state["current_api_key_idx"] += 1
                            qa_model = get_qa_model(st.session_state["current_api_key_idx"])
                            st.warning("API rate limit hit. Switching to next key.")
                            st.rerun()
                        elif "rate limit" in str(e).lower():
                            st.warning("All API keys exhausted. Try again later.")
        else:
            st.info("Please upload PDFs to start chatting.")
