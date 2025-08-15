# Διόρθωση για το chromadb
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Απαραίτητες εισαγωγές
import streamlit as st
import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage

# Ρύθμιση του asyncio event loop
try:
    _ = asyncio.get_running_loop()
except RuntimeError as ex:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Δημιουργία του Gemini Pro μοντέλου
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def process_single_txt_file(file_path):
    """Επεξεργάζεται ένα μόνο αρχείο κειμένου."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    if not os.path.exists(file_path):
        st.error(f"Το αρχείο {file_path} δεν βρέθηκε.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
    except Exception as e:
        st.error(f"Σφάλμα κατά την ανάγνωση του αρχείου: {e}")
        return None

    if not document_text.strip():
        st.error("Το αρχείο κειμένου είναι κενό.")
        return None

    text_chunks = text_splitter.split_text(document_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma.from_texts(text_chunks, embeddings)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Ρύθμιση του Streamlit UI
st.set_page_config(page_title="Απλό Chatbot", layout="wide")
st.header("💬 Απλό Chatbot με Gemini")

# Αποθήκευση ιστορικού συνομιλίας
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Επεξεργασία του αρχείου μόνο μία φορά
file_path = os.path.join("data", "data.txt")
if st.session_state.qa_chain is None:
    with st.spinner("Επεξεργάζομαι το αρχείο κειμένου..."):
        st.session_state.qa_chain = process_single_txt_file(file_path)
        if st.session_state.qa_chain:
            st.success("Το αρχείο επεξεργάστηκε με επιτυχία!")
        else:
            st.error("Αδυναμία επεξεργασίας του αρχείου. Παρακαλώ ελέγξτε τον φάκελο 'data'.")

# Εμφάνιση του ιστορικού συνομιλίας
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Είσοδος χρήστη
if prompt := st.chat_input("Ρώτησέ με κάτι για το κείμενο..."):
    # Εμφάνιση ερώτησης χρήστη
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Παραγωγή απάντησης από το chatbot
    with st.chat_message("assistant"):
        if st.session_state.qa_chain:
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
            answer = "Παρακαλώ επανεκκινήστε την εφαρμογή για να επεξεργαστούν τα έγγραφα."
            st.markdown(answer)

    # Αποθήκευση απάντησης στο ιστορικό
    st.session_state.messages.append({"role": "assistant", "content": answer})
