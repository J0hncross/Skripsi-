from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def is_court_decision(text):
    # Memastikan teks mengandung setidaknya 3 kata kunci unik
    keywords = ["putusan", "pengadilan", "putusan hakim", "putusan perkara", 
                "pengadilan negeri", "terdakwa", "putusan pidana", "putusan perdata"]
    found_keywords = {keyword for keyword in keywords if keyword in text.lower()}
    return len(found_keywords) >= 3  # Memeriksa apakah ada minimal 3 kata kunci unik

def main():
    load_dotenv()
    st.set_page_config(page_title="Tanya PDF")
    st.header("Tanya Jawab PDF💬")
    
    pdf = st.file_uploader("Upload PDF Anda", type="pdf")
    # Extract teks
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Cek apakah PDF berisi informasi mengenai putusan pengadilan
        if not is_court_decision(text):
            st.warning("File yang di-upload bukan mengenai putusan pengadilan. Silakan unggah dokumen yang sesuai.")
            return
        
        # Cek apakah teks terlalu pendek untuk diproses
        minimum_length = 1000  # minimal karakter
        if len(text) < minimum_length:
            st.warning("Dokumen terlalu singkat. Silakan unggah dokumen yang lebih panjang.")
            return
        
        # Chunk Of text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings and knowledge base
        if chunks:
            success_message = st.success("Dokumen berhasil diunggah, silakan bertanya.")
            
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            # Question
            user_question = st.text_input("Tanya PDF:")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                    # Display the response with justified text
                    st.markdown(f"<p style='text-align: justify;'>{response}</p>", unsafe_allow_html=True)

# Jalankan aplikasi
if __name__ == '__main__':
    main()
