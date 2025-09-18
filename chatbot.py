import streamlit as st
import pdfplumber
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
HUGGING_FACE_API_KEY = "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"


def initialize_llm():
    """Initialize the language model with error handling"""
    try:
        st.info("Loading language model... This may take a moment.")

        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1

        # Create pipeline with optimized settings
        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=device,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=0
        )

        llm = HuggingFacePipeline(
            pipeline=hf_pipeline,
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )

        st.success("Language model loaded successfully!")
        return llm

    except Exception as e:
        st.warning(f"Could not load advanced language model: {str(e)}")
        st.info("Falling back to simple response generation.")
        return None


def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
            st.info(f"Processing {total_pages} pages...")

            progress_bar = st.progress(0)

            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    text += f"No text found on page {i + 1}.\n"

                # Update progress bar
                progress_bar.progress((i + 1) / total_pages)

            progress_bar.empty()

    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

    return text.strip()


def create_vector_store(text):
    """Create vector store from text chunks"""
    try:
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)

        if not chunks:
            st.error("No valid chunks created from the text.")
            return None

        st.info(f"Created {len(chunks)} text chunks.")

        # Generate embeddings
        st.info("Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create FAISS vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        st.success("Vector store created successfully!")

        return vector_store

    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def generate_simple_response(query, matching_chunks):
    """Generate simple response when LLM is not available"""
    if not matching_chunks:
        return "I couldn't find relevant information in the document for your query."

    context_text = ""
    for i, chunk in enumerate(matching_chunks[:3]):
        content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        context_text += f"\n**Relevant Section {i + 1}:**\n{content}\n"

    response = f"""**Query:** {query}

**Answer:** Based on the uploaded document, here are the most relevant sections:
{context_text}

*Note: This is a basic text search result. For more sophisticated answers, ensure the language model is properly loaded.*"""

    return response


def generate_llm_response(llm, query, matching_chunks):
    """Generate response using the language model"""
    try:
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions based on the provided context.

        Context: {context}

        Question: {input}

        Answer: Provide a clear and concise answer based only on the information given in the context. If the context doesn't contain enough information to answer the question, say so.
        """)

        # Create document chain
        chain = create_stuff_documents_chain(llm, prompt_template)

        # Generate response
        response = chain.invoke({
            "input": query,
            "context": matching_chunks
        })

        return response

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return generate_simple_response(query, matching_chunks)


# Main Streamlit App
def main():
    st.set_page_config(
        page_title="PDF ChatBot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 PDF ChatBot")
    st.markdown("Upload a PDF file and ask questions about its content!")

    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to start chatting with its content"
        )

        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            file_details = {
                "Filename": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size} bytes"
            }
            st.json(file_details)

    # Main content area
    if uploaded_file is not None:
        # Initialize session state
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'llm' not in st.session_state:
            st.session_state.llm = None
        if 'text_extracted' not in st.session_state:
            st.session_state.text_extracted = False

        # Process PDF if not already done
        if not st.session_state.text_extracted:
            with st.spinner("Processing PDF..."):
                # Extract text from PDF
                text = extract_text_from_pdf(uploaded_file)

                if text:
                    st.success(f"Extracted {len(text)} characters from PDF")

                    # Create vector store
                    st.session_state.vector_store = create_vector_store(text)

                    if st.session_state.vector_store:
                        # Initialize LLM
                        st.session_state.llm = initialize_llm()
                        st.session_state.text_extracted = True
                else:
                    st.error("Failed to extract text from PDF")
                    return

        # Chat interface
        if st.session_state.text_extracted and st.session_state.vector_store:
            st.header("💬 Chat with your PDF")

            # Query input
            user_query = st.text_input(
                "Ask a question about your document:",
                placeholder="e.g., What is the main topic of this document?"
            )

            # Search button
            if st.button("🔍 Search", type="primary"):
                if user_query:
                    with st.spinner("Searching for relevant information..."):
                        # Perform similarity search
                        try:
                            matching_chunks = st.session_state.vector_store.similarity_search(
                                user_query, k=3
                            )

                            if matching_chunks:
                                st.subheader("📋 Answer:")

                                # Generate response
                                if st.session_state.llm:
                                    response = generate_llm_response(
                                        st.session_state.llm,
                                        user_query,
                                        matching_chunks
                                    )
                                else:
                                    response = generate_simple_response(
                                        user_query,
                                        matching_chunks
                                    )

                                st.markdown(response)

                                # Show source chunks in expander
                                with st.expander("📚 Source Information"):
                                    for i, chunk in enumerate(matching_chunks):
                                        content = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                                        st.markdown(f"**Chunk {i + 1}:**")
                                        st.text(content[:300] + "..." if len(content) > 300 else content)
                                        st.markdown("---")
                            else:
                                st.warning("No relevant information found in the document.")

                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
                else:
                    st.warning("Please enter a question.")

            # Example queries
            st.subheader("💡 Example Questions:")
            example_queries = [
                "What is this document about?",
                "Who are the main people mentioned?",
                "What are the key findings?",
                "Summarize the main points",
                "What dates are mentioned?"
            ]

            cols = st.columns(len(example_queries))
            for i, query in enumerate(example_queries):
                with cols[i]:
                    if st.button(f"📝 {query}", key=f"example_{i}"):
                        st.session_state.example_query = query
                        st.rerun()

            # Handle example query selection
            if 'example_query' in st.session_state:
                user_query = st.session_state.example_query
                del st.session_state.example_query

                with st.spinner("Searching for relevant information..."):
                    matching_chunks = st.session_state.vector_store.similarity_search(user_query, k=3)

                    if matching_chunks:
                        st.subheader("📋 Answer:")

                        if st.session_state.llm:
                            response = generate_llm_response(
                                st.session_state.llm,
                                user_query,
                                matching_chunks
                            )
                        else:
                            response = generate_simple_response(
                                user_query,
                                matching_chunks
                            )

                        st.markdown(response)

    else:
        # Welcome message
        st.markdown("""
        ### Welcome to PDF ChatBot! 🚀

        **How to use:**
        1. Upload a PDF file using the sidebar
        2. Wait for the document to be processed
        3. Ask questions about the content
        4. Get AI-powered answers based on your document

        **Features:**
        - ✅ Extract text from PDF files
        - ✅ Intelligent text chunking
        - ✅ Semantic search using embeddings
        - ✅ AI-powered question answering
        - ✅ Source information display

        Start by uploading a PDF file! 📄
        """)


if __name__ == "__main__":
    main()