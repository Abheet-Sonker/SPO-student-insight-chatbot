import os
import zipfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# # Load environment variables
# load_dotenv()
# groq_api_key = "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"

# Load embedding model locally (avoids Hugging Face API connection and DNS resolution errors)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vectorstore relative to file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index_insights")

vectorstore = FAISS.load_local(
    folder_path=FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define prompt
prompt_template = """
You are a helpful placement assistant for IIT Kanpur students.
Answer ONLY using the below context. otherwise, say you don't know.
only give specific answers to the questions
Only give answers for the selected company strictly
Context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(prompt_template)

# Load LLM
llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.1-8b-instant",
    groq_api_key="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"
)

# Format documents helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Company lists
placement_company_list = [
    'Alan Harshan Jaguar Land Rover India Limited', 'Applied Intelligence',
    'Bajaj Auto Limited  Chetak Technology Limited', 'Bajaj Electricals',
    'Balvantbhai Sap Labs', 'Barclays', 'Battery Smart', 'Caterpillar',
    'Chaudhuri United Airlines', 'Cohesity', 'Das Kivi Capital', 'Databricks',
    'Deutsche India Private Limited', 'Electric', 'Finmechanics', 'Flipkart',
    'Ford Business Solutions', 'Glean', 'Godrej  Boyce', 'Google', 'Greyorange',
    'Group India', 'Gupta Raspa Pharma Private Limited', 'Hilabs', 'Hilti Technology',
    'Hitachi Energy', 'Hpcl', 'Hyper Bots System', 'Icici Bank', 'Idfc First Bank',
    'Jadhav Flipkart', 'Jaguar Land Rover India Limited', 'Javis', 'Karthik Ormae',
    'Kumar Goud Sap Labs', 'Kunsoth Merilytics', 'Larsen And Toubro Limited',
    'Mahindra Susten Pvt Ltd', 'Master Card', 'Medianetdirecti', 'Mekala Western Digital',
    'Menon Publicis Sapient', 'Microsoft India', 'Na', 'Navi', 'Neterwala Group Aon',
    'Niva Bupa Health Insurance', 'Nmtronics India Private Limited', 'Novo Nordisk',
    'Npci', 'Ola', 'Palo Alto Networks', 'Petronet Lng Limited', 'Pharmaace',
    'Pine Labs', 'Qualcomm', 'Quantbox Research', 'Rosy Blue India Pvt Ltd',
    'Singh Nvidia', 'Smarttrak Ai', 'Solutions Pvt Ltd', 'Sprinklr', 'Steel',
    'Stripe', 'Taiwan Semiconductor Manufacturing Company', 'Tata Projects Ltd',
    'Technology', 'Thakur Navi', 'Uniorbit Technologies',
    'Uniorbit Technologies Private Ltd', 'Varghese Mrf Limited', 'World Quant', 'Zomato'
]

intern_company_list = [
    'Amazon', 'American Express', 'Anuj Rubrik', 'Atlassian', 'Axxela',
    'Bain  Company', 'Bain And Company', 'Barclays', 'Bcg', 'Bny Mellon',
    'Bosch', 'Boston Consulting Group', 'Cisco', 'Citi', 'Citi Bank', 'Cohesity',
    'Databricks Sde', 'Deshpande Alphagrep Securities Private Limited', 'Discovery',
    'Dr Reddys Laboratories', 'Dr Reddys Laboratories  Core Engineering',
    'Edelweiss Financial Services Limited', 'Express', 'Finmechanics', 'Flipkart',
    'Geetika Uber', 'Glean', 'Goldman Sachs', 'Google', 'Google India', 'Graviton',
    'Hindustan Unilever Limited', 'Hul', 'India', 'Indxx', 'Itc Limited', 'Itc Ltd',
    'J P Morgan  Chase', 'Jaguar Land Rover India Limited', 'Jindal Medianet',
    'Jiwane Oracle', 'Jlr', 'Jp Morgan Chase', 'Jsw', 'Kane American Express',
    'Kumar Samsung Noida', 'Mckinsey  Company', 'Medianet', 'Microsoft India',
    'Mondelez International', 'Morgan Stanley', 'N Bny Mellon',
    'National Payments Corporation Of India Npci', 'Nestle', 'Nk Security',
    'Nobroker', 'Nobroker Technologies', 'Nvidia', 'Optiver', 'Optiverquant Role',
    'Quadeye', 'Quantbox', 'R Qualcomm', 'Raj Databricks', 'Rubrik', 'Sachan Ibm',
    'Sachs', 'Saluja Mastercard', 'Samplytics Technologies Private Limited',
    'Samsung Rd Banglore', 'Samsung Research Bangalore', 'Samsung South Korea',
    'Sprinklr', 'Standard Chartered', 'Tata Steel', 'Texas Instruments',
    'Tiwari De Shaw', 'Tomar Jlr', 'Tower Research Capital',
    'Trexquant Investment Llp', 'Uber', 'Vedanta Resources Limited', 'Winzo Games'
]

query_types = ["Sample Interview Questions", "Interview Process", "Resources", "Advice"]

# Chatbot function
def ask_bot(company, query_types, question_text):
    if not company:
        return "❗ Please select a company.", ""

    if query_types:
        questions = [f"Give me {qt.lower()} only for {company} , Dont give for any other company." for qt in query_types]
    elif question_text.strip():
        questions = [question_text]
    else:
        return "❗ Please select a query type or enter a question.", ""

    full_response = ""
    all_sources = set()

    for q in questions:
        answer = rag_chain.invoke(q)
        source_docs = retriever.invoke(q)
        
        full_response += f"**Q: {q}**\n{answer}\n\n"
        for doc in source_docs:
            all_sources.add(doc.page_content.strip())

    sources_text = "\n---\n".join(all_sources) if all_sources else "*No source documents found.*"
    return full_response.strip(), sources_text

# --- Streamlit UI ---
st.set_page_config(page_title="Company Interview Chatbot", page_icon="🤖")
st.title("🤖 Company Interview Chatbot (IITK Insights)")

col1, col2 = st.columns(2)
with col1:
    purpose = st.selectbox("🎯 Purpose (Placement or Intern)", ["Placement", "Intern"], index=0)

with col2:
    company_list = placement_company_list if purpose == "Placement" else intern_company_list
    company_name = st.selectbox("🏢 Select Company", company_list)

query_options = st.multiselect(
    "🔎 Select What You Want to Know",
    ["Sample Interview Questions", "Interview Process", "Resources", "Advice"]
)
question = st.text_input("❓ Ask a custom question (optional)")

if st.button("🔍 Ask"):
    final_query = ""

    if query_options:
        final_query += f"{', '.join(query_options)} for {company_name}. "

    if question.strip():
        final_query += question.strip()

    if not final_query:
        st.warning("Please select options or enter a custom question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(final_query)
            source_docs = retriever.invoke(final_query)

            st.markdown("### ✅ Answer")
            st.success(answer)

            st.markdown("### 📚 Source Documents")
            for doc in source_docs:
                st.markdown(f"**Source**: {doc.metadata.get('source', 'N/A')}")
                st.markdown(doc.page_content[:500] + "...")
