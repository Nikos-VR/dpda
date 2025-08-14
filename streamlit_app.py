import streamlit as st
import io
import os
import asyncio
import pysqlite3
import sys

# Ρύθμιση του pysqlite3 για συμβατότητα με το Chroma
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
# Ρύθμιση του asyncio event loop για να αποφευχθεί το λάθος "There is no current event loop"
try:
    _ = asyncio.get_running_loop()
except RuntimeError as ex:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Δημιουργία του Gemini Pro μοντέλου
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=st.secrets["GOOGLE_API_KEY"])

def get_text_from_url(url):
    """Διαβάζει το κείμενο από μια ιστοσελίδα."""
    try:
        url_content = Browse(query="the entire text content of the webpage", url=url)
        return url_content
    except Exception as e:
        st.error(f"Σφάλμα κατά την ανάγνωση του URL {url}: {e}")
        return ""

@st.cache_resource
def process_documents(pdf_directory):
    """Επεξεργάζεται τα PDF, HTML και URLs που βρίσκονται στον φάκελο δεδομένων."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_text = ""

    # Βρίσκει όλα τα αρχεία PDF, HTML και το urls.txt στον φάκελο
    files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory)]

    for file_path in files:
        try:
            if file_path.endswith('.pdf'):
                # Διαβάζει PDF
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    all_text += page.extract_text()
            elif file_path.endswith(('.html', '.htm')):
                # Διαβάζει HTML
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    all_text += soup.get_text()
            elif file_path.endswith('.txt') and os.path.basename(file_path) == 'urls.txt':
                # Διαβάζει URLs από το αρχείο urls.txt
                with open(file_path, 'r', encoding='utf-8') as file:
                    urls = file.read().splitlines()
                    for url in urls:
                        if url.strip():
                            all_text += get_text_from_url(url.strip())

        except Exception as e:
            st.error(f"Σφάλμα κατά την ανάγνωση του αρχείου {file_path}: {e}")
            return None
    
    if not all_text.strip():
        st.error("Δεν βρέθηκε κείμενο για επεξεργασία. Ελέγξτε τα αρχεία ή το URL.")
        return None

    text_chunks = text_splitter.split_text(all_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma.from_texts(text_chunks, embeddings)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Ρύθμιση του Streamlit UI
st.set_page_config(page_title="ΤΠΨΤ chatbot", layout="wide")
st.header("Είμαι ο βοηθός σας, καλωσήλθατε!")

# Αποθήκευση ιστορικού συνομιλίας
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Δημιουργία της διαδρομής προς τον φάκελο με τα έγγραφα
pdf_dir = "data"

# Εμφάνιση μηνύματος φόρτωσης στην αρχή
if st.session_state.qa_chain is None:
    with st.spinner("Μισό λεπτό παρακαλώ, διαβάζω τα απαραίτητα έγγραφα..."):
        st.session_state.qa_chain = process_documents(pdf_dir)
        if st.session_state.qa_chain:
            st.success("Η ενημέρωσή μου ολοκληρώθηκε με επιτυχία!")
        else:
            st.error("Αδυναμία επεξεργασίας των εγγράφων...")

# Εμφάνιση του ιστορικού συνομιλίας
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Είσοδος χρήστη
if prompt := st.chat_input("Ρωτήστε με κάτι για το τμήμα Παραστατικών και Ψηφιακών Τεχνών..."):
    # Εμφάνιση ερώτησης χρήστη
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Παραγωγή απάντησης από το chatbot
    with st.chat_message("assistant"):
        if st.session_state.qa_chain:
            # Μετατροπή ιστορικού σε συμβατή μορφή
            chat_history_formatted = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_history_formatted.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history_formatted.append(AIMessage(content=msg["content"]))

            response = st.session_state.qa_chain({"question": prompt, "chat_history": chat_history_formatted})
            answer = response["answer"]
            st.markdown(answer)
        else:
            answer = "Παρακαλώ επανεκκινήστε την εφαρμογή για να επεξεργαστώ τα έγγραφα."
            st.markdown(answer)

    # Αποθήκευση απάντησης στο ιστορικό
    st.session_state.messages.append({"role": "assistant", "content": answer})
