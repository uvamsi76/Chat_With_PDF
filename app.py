import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
def main():
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
            )
        chunks = text_splitter.split_text(text=text)

        vectorstore = Chroma.from_texts(embedding=OpenAIEmbeddings(),texts=chunks,collection_name='test')
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
        query = st.text_input("Ask questions about your PDF file:")

        result=rag_chain.invoke(query)
        # print(result)
        st.write(result)
 
if __name__ == '__main__':
    main()