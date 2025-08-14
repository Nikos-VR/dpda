import streamlit as st
import io
import os
import asyncio
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from PyPDF2 import PdfReader

# Ρύθμιση του asyncio event loop για να αποφευχθεί το λάθος "There is no current event loop"
try:
    _ = asyncio.get_running_loop()
except RuntimeError as ex:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Δημιουργία του Gemini Pro μοντέλου
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def process_preloaded_documents(pdf_directory):
    """Επεξεργάζεται τα PDF που βρίσκονται σε έναν συγκεκριμένο φάκελο."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    all_text = ""
    # Βρίσκει όλα τα PDF στον φάκελο
    pdf_files = [
        os.path.join(pdf_directory, f)
        for f in os.listdir(pdf_directory)
        if f.endswith('.pdf')
    ]

    for pdf_path in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                all_text += page.extract_text()
        except Exception as e:
            st.error(f"Σφάλμα κατά την ανάγνωση του αρχείου {pdf_path}: {e}")
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
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("💬 PDF Chatbot με Gemini")

# Δημιουργία της διαδρομής προς τον φάκελο με τα έγγραφα
pdf_dir = "data"

# Αποθήκευση ιστορικού συνομιλίας
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Εμφάνιση μηνύματος φόρτωσης στην αρχή
if st.session_state.qa_chain is None:
    with st.spinner("Επεξεργάζομαι τα έγγραφα..."):
        st.session_state.qa_chain = process_preloaded_documents(pdf_dir)
        if st.session_state.qa_chain:
            st.success("Τα έγγραφα έχουν επεξεργαστεί με επιτυχία!")
        else:
            st.error("Αδυναμία επεξεργασίας των εγγράφων. Παρακαλώ ελέγξτε τα αρχεία σας.")

# Εμφάνιση του ιστορικού συνομιλίας
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Είσοδος χρήστη
if prompt := st.chat_input("Ρώτησέ με κάτι για τα έγγραφα..."):
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
            answer = "Παρακαλώ επανεκκινήστε την εφαρμογή για να επεξεργαστούν τα έγγραφα."
            st.markdown(answer)

    # Αποθήκευση απάντησης στο ιστορικό
    st.session_state.messages.append({"role": "assistant", "content": answer})
