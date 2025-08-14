__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import io
import asyncio

# Ρύθμιση του event loop
try:
    _ = asyncio.get_running_loop()
except RuntimeError as ex:  # No running event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader

# Δημιουργία του Gemini Pro μοντέλου
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5, google_api_key=st.secrets["GOOGLE_API_KEY"])

def get_document_chunks(pdf_docs):
    """Μετατροπή PDF σε κομμάτια κειμένου από το UploadedFile αντικείμενο."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_text = ""
    for pdf_file in pdf_docs:
        # Διάβασμα του αρχείου στη μνήμη
        pdf_reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
        for page in pdf_reader.pages:
            all_text += page.extract_text()
    
    return text_splitter.split_text(all_text)

def create_vector_store(text_chunks):
    """Δημιουργία βάσης δεδομένων από κομμάτια κειμένου."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = Chroma.from_texts(text_chunks, embeddings)
    return vector_store

def create_qa_chain(vector_store):
    """Δημιουργία της αλυσίδας (chain) για ερωτήσεις-απαντήσεις."""
    retriever = vector_store.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Ρύθμιση του Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("💬 PDF Chatbot με Gemini Pro")

# Αποθήκευση ιστορικού συνομιλίας
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Φόρμα για ανέβασμα αρχείων
with st.sidebar:
    st.header("Φόρτωσε τα έγγραφά σου")
    uploaded_files = st.file_uploader(
        "Ανέβασε PDF αρχεία", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Επεξεργασία"):
            with st.spinner("Επεξεργάζομαι τα έγγραφα..."):
                text_chunks = get_document_chunks(uploaded_files)
                vector_store = create_vector_store(text_chunks)
                st.session_state.qa_chain = create_qa_chain(vector_store)
                st.success("Τα έγγραφα έχουν επεξεργαστεί με επιτυχία!")

# Εμφάνιση του ιστορικού συνομιλίας
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Είσοδος χρήστη
if prompt := st.chat_input("Ρώτησέ με κάτι για τα έγγραφα που ανέβασες..."):
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
            answer = "Παρακαλώ φόρτωσε και επεξεργάσου κάποια PDF αρχεία για να ξεκινήσεις."
            st.markdown(answer)
    
    # Αποθήκευση απάντησης στο ιστορικό
    st.session_state.messages.append({"role": "assistant", "content": answer})
