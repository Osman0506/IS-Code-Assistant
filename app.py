import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import time
from datetime import datetime
import json

load_dotenv()

# Initialize session state
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files with error handling"""
    text = ""
    file_details = []
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            pdf_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text
            
            if pdf_text.strip():
                text += pdf_text
                file_details.append({
                    'name': pdf.name,
                    'pages': len(pdf_reader.pages),
                    'chars': len(pdf_text)
                })
            else:
                st.warning(f"⚠️ No text extracted from {pdf.name}. It might be scanned/image-based.")
        
        except Exception as e:
            st.error(f"❌ Error reading {pdf.name}: {str(e)}")
    
    return text, file_details

def get_text_chunks(text):
    """Split text into chunks with metadata"""
    if not text or len(text.strip()) < 100:
        raise ValueError("Insufficient text content to process")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Create and save vector store with session state caching"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        
        # Cache in session state
        st.session_state.embeddings = embeddings
        st.session_state.vector_store = vector_store
        st.session_state.vector_store_ready = True
        
        return True
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return False

def get_answer_from_docs(docs, question, api_key):
    """Generate answer from documents using direct LLM call (no chain)"""
    
    # Combine document contents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""You are an expert assistant for Indian Standard (IS) codes and civil engineering specifications.

Answer the question as detailed as possible using ONLY the provided context from the IS codes.

Guidelines:
- Provide specific clause numbers, section references, or table numbers when available
- If the answer requires numerical values, tables, or formulas, include them precisely
- If the context doesn't contain the answer, clearly state: "This information is not available in the uploaded IS codes."
- Do not make assumptions or provide information from outside the given context
- Structure your answer clearly with relevant headings if needed

Context:
{context}

Question:
{question}

Answer:
"""
    
    try:
        # Initialize the model
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=api_key
        )
        
        # Get response directly
        response = model.invoke(prompt)
        
        # Extract text from response
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        st.error(f"❌ Error generating answer: {str(e)}")
        return None

def user_input(user_question, api_key):
    """Process user question and generate response with source citations"""
    try:
        # Use cached vector store if available
        if st.session_state.vector_store is not None:
            new_db = st.session_state.vector_store
        else:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=api_key
            )
            new_db = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.session_state.vector_store = new_db
        
        # Retrieve relevant documents
        docs = new_db.similarity_search(user_question, k=4)
        
        if not docs:
            st.warning("⚠️ No relevant information found in the uploaded documents.")
            return None, None
        
        # Generate answer using direct LLM call
        with st.spinner("🔍 Searching through IS codes..."):
            answer = get_answer_from_docs(docs, user_question, api_key)
        
        if not answer:
            return None, None
        
        # Add to chat history
        st.session_state.chat_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'question': user_question,
            'answer': answer,
            'sources': [doc.page_content[:200] + "..." for doc in docs[:2]]
        })
        
        return answer, docs
    
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")
        return None, None

def display_chat_history():
    """Display chat history in reverse chronological order"""
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"🕐 {chat['timestamp']} - {chat['question'][:50]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                
                if 'sources' in chat and chat['sources']:
                    st.markdown("**📄 Source Excerpts:**")
                    for idx, source in enumerate(chat['sources'], 1):
                        st.text(f"{idx}. {source}")

def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.chat_history:
        chat_data = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            label="📥 Download Chat History (JSON)",
            data=chat_data,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    st.set_page_config(
        page_title="CivilCode AI", 
        page_icon="🏗️",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🏗️ CivilCode AI: Chat with IS Codes")
    st.markdown("*Ask questions about Indian Standard codes - Get precise, cited answers*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key:", 
            type="password",
            help="Get your free API key from https://makersuite.google.com/app/apikey"
        )
        
        st.markdown("---")
        st.subheader("📁 Upload Documents")
        
        pdf_docs = st.file_uploader(
            "Upload IS Codes (PDF)", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        # Process button
        if st.button("🚀 Process PDFs", use_container_width=True):
            if not api_key:
                st.error("❌ Please enter your Google API Key first")
            elif not pdf_docs:
                st.error("❌ Please upload at least one PDF file")
            else:
                with st.spinner("⏳ Processing PDFs..."):
                    # Extract text
                    raw_text, file_details = get_pdf_text(pdf_docs)
                    
                    if not raw_text or len(raw_text.strip()) < 100:
                        st.error("❌ No sufficient text extracted from PDFs. Please check your files.")
                    else:
                        # Display file info
                        st.success(f"✅ Extracted text from {len(file_details)} file(s)")
                        for detail in file_details:
                            st.info(f"📄 {detail['name']}: {detail['pages']} pages, {detail['chars']:,} characters")
                        
                        # Create chunks
                        try:
                            text_chunks = get_text_chunks(raw_text)
                            st.info(f"📊 Created {len(text_chunks)} text chunks")
                            
                            # Create vector store
                            if get_vector_store(text_chunks, api_key):
                                st.session_state.processed_files = [f.name for f in pdf_docs]
                                st.success("✅ Processing complete! You can now ask questions.")
                                st.balloons()
                        
                        except ValueError as e:
                            st.error(f"❌ {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Error during processing: {str(e)}")
        
        # Show processed files
        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("✅ Processed Files")
            for file in st.session_state.processed_files:
                st.text(f"• {file}")
        
        # Clear history button
        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            
            export_chat_history()
        
        # Info section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        This app uses RAG (Retrieval-Augmented Generation) to answer questions about IS codes.
        
        **How it works:**
        1. Upload PDF IS codes
        2. Click "Process PDFs"
        3. Ask questions naturally
        4. Get cited answers
        
        **Features:**
        - Source citations
        - Chat history
        - Export conversations
        - Multi-document support
        
        **Note:** This version uses direct LLM calls (no chains) to avoid import issues.
        """)
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input
        st.subheader("❓ Ask a Question")
        user_question = st.text_input(
            "Type your question here:",
            placeholder="e.g., What is the minimum grade of concrete for RCC work?",
            label_visibility="collapsed"
        )
        
        # Process question
        if user_question:
            if not api_key:
                st.error("❌ Please enter your Google API Key in the sidebar")
            elif not st.session_state.vector_store_ready and not os.path.exists("faiss_index"):
                st.error("❌ Please upload and process PDF files first")
            else:
                answer, docs = user_input(user_question, api_key)
                
                if answer:
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(answer)
                    
                    # Display sources
                    if docs:
                        with st.expander("📚 View Source Excerpts", expanded=False):
                            for idx, doc in enumerate(docs[:3], 1):
                                st.markdown(f"**Source {idx}:**")
                                st.text(doc.page_content[:500] + "...")
                                st.markdown("---")
    
    with col2:
        # Display chat history
        if st.session_state.chat_history:
            display_chat_history()
        else:
            st.info("💭 Your conversation history will appear here")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit + LangChain + Google Gemini | "
        "⚠️ Always verify critical information from official IS code documents"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
