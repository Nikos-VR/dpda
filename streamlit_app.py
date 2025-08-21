# Τοποθέτησε αυτές τις γραμμές ΑΜΕΣΑ στην κορυφή του αρχείου
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Απαραίτητες εισαγωγές
import streamlit as st
import os
import asyncio
import mysql.connector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Ρύθμιση του asyncio event loop
try:
    _ = asyncio.get_running_loop()
except RuntimeError as ex:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Δημιουργία του Gemini Pro μοντέλου
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=st.secrets["keys"]["GOOGLE_API_KEY"])

# --- Ασφαλής Σύνδεση στη Βάση Δεδομένων MySQL ---
@st.cache_resource
def get_database_connection():
    """Δημιουργεί και επιστρέφει μια σύνδεση στη βάση δεδομένων MySQL."""
    try:
        conn = mysql.connector.connect(
            host=st.secrets["database"]["host"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            database=st.secrets["database"]["db"]
        )
        return conn
    except Exception as e:
        st.error(f"Σφάλμα σύνδεσης στη βάση δεδομένων: {e}")
        return None

# --- Επεξεργασία Δεδομένων και Δημιουργία RAG ---
@st.cache_resource
def process_data_from_db(data_from_db):
    """Δημιουργεί το RAG σύστημα από δεδομένα που προέρχονται από τη βάση δεδομένων."""
    st.info("Επεξεργασία δεδομένων...")
    
    documents = []
    if not data_from_db:
        st.warning("Δεν βρέθηκαν δεδομένα για επεξεργασία.")
        return None

    for idx, row in enumerate(data_from_db):
        content = " ".join(str(item) for item in row)
        documents.append(Document(page_content=content, metadata={"source": f"Database Row {idx+1}"}))
        
    st.write(f"Βρέθηκαν {len(documents)} γραμμές για επεξεργασία.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma.from_documents(text_chunks, embeddings)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# --- Κεντρική Λογική της Εφαρμογής ---
st.set_page_config(page_title="MySQL & Gemini Chatbot", layout="wide")
st.header("💬 Chatbot με Δεδομένα από MySQL")

# Αποθήκευση ιστορικού συνομιλίας και QA chain
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Αυτόματη φόρτωση δεδομένων κατά την εκκίνηση
if st.session_state.qa_chain is None:
    st.info("Φόρτωση δεδομένων από τον πίνακα 'table1'...")
    conn = get_database_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM table1;")
                results = cursor.fetchall()
            if results:
                st.session_state.qa_chain = process_data_from_db(results)
                st.success("Τα δεδομένα φορτώθηκαν με επιτυχία! Μπορείτε να ξεκινήσετε τη συνομιλία.")
            else:
                st.warning("Ο πίνακας 'table1' είναι άδειος ή το query δεν επέστρεψε αποτελέσματα.")
                st.session_state.qa_chain = None
        except Exception as e:
            st.error(f"Σφάλμα κατά την εκτέλεση του query: {e}")
            st.session_state.qa_chain = None
    else:
        st.error("Δεν ήταν δυνατή η σύνδεση στη βάση δεδομένων.")

# Εμφάνιση του chatbot μόνο αν έχει φορτωθεί το QA chain
if st.session_state.qa_chain:
    st.write("---")
    
    # Εμφάνιση του ιστορικού συνομιλίας
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Είσοδος χρήστη
    if prompt := st.chat_input("Ρώτησέ με κάτι για τα δεδομένα..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Παραγωγή απάντησης από το chatbot
        with st.chat_message("assistant"):
            last_two_messages = st.session_state.messages[-2:]
            chat_history_formatted = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in last_two_messages
            ]

            response = st.session_state.qa_chain({"question": prompt, "chat_history": chat_history_formatted})
            answer = response["answer"]
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
