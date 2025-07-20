import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pyngrok import ngrok

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "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")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
faiss_index_path = "faiss_index_insights"
vectorstore = FAISS.load_local(
    folder_path=faiss_index_path,
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
    model_name="llama3-8b-8192",
    groq_api_key="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"
)

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Company lists
placement_company_list = ['Alan Harshan Jaguar Land Rover India Limited',
 'Applied Intelligence',
 'Bajaj Auto Limited  Chetak Technology Limited',
 'Bajaj Electricals',
 'Balvantbhai Sap Labs',
 'Barclays',
 'Battery Smart',
 'Caterpillar',
 'Chaudhuri United Airlines',
 'Cohesity',
 'Das Kivi Capital',
 'Databricks',
 'Deutsche India Private Limited',
 'Electric',
 'Finmechanics',
 'Flipkart',
 'Ford Business Solutions',
 'Glean',
 'Godrej  Boyce',
 'Google',
 'Greyorange',
 'Group India',
 'Gupta Raspa Pharma Private Limited',
 'Hilabs',
 'Hilti Technology',
 'Hitachi Energy',
 'Hpcl',
 'Hyper Bots System',
 'Icici Bank',
 'Idfc First Bank',
 'Jadhav Flipkart',
 'Jaguar Land Rover India Limited',
 'Javis',
 'Karthik Ormae',
 'Kumar Goud Sap Labs',
 'Kunsoth Merilytics',
 'Larsen And Toubro Limited',
 'Mahindra Susten Pvt Ltd',
 'Master Card',
 'Medianetdirecti',
 'Mekala Western Digital',
 'Menon Publicis Sapient',
 'Microsoft India',
 'Na',
 'Navi',
 'Neterwala Group Aon',
 'Niva Bupa Health Insurance',
 'Nmtronics India Private Limited',
 'Novo Nordisk',
 'Npci',
 'Ola',
 'Palo Alto Networks',
 'Petronet Lng Limited',
 'Pharmaace',
 'Pine Labs',
 'Qualcomm',
 'Quantbox Research',
 'Rosy Blue India Pvt Ltd',
 'Singh Nvidia',
 'Smarttrak Ai',
 'Solutions Pvt Ltd',
 'Sprinklr',
 'Steel',
 'Stripe',
 'Taiwan Semiconductor Manufacturing Company',
 'Tata Projects Ltd',
 'Technology',
 'Thakur Navi',
 'Uniorbit Technologies',
 'Uniorbit Technologies Private Ltd',
 'Varghese Mrf Limited',
 'World Quant',
 'Zomato']


  # your existing placement company list
intern_company_list = ['Amazon',
 'American Express',
 'Anuj Rubrik',
 'Atlassian',
 'Axxela',
 'Bain  Company',
 'Bain And Company',
 'Barclays',
 'Bcg',
 'Bny Mellon',
 'Bosch',
 'Boston Consulting Group',
 'Cisco',
 'Citi',
 'Citi Bank',
 'Cohesity',
 'Databricks Sde',
 'Deshpande Alphagrep Securities Private Limited',
 'Discovery',
 'Dr Reddys Laboratories',
 'Dr Reddys Laboratories  Core Engineering',
 'Edelweiss Financial Services Limited',
 'Express',
 'Finmechanics',
 'Flipkart',
 'Geetika Uber',
 'Glean',
 'Goldman Sachs',
 'Google',
 'Google India',
 'Graviton',
 'Hindustan Unilever Limited',
 'Hul',
 'India',
 'Indxx',
 'Itc Limited',
 'Itc Ltd',
 'J P Morgan  Chase',
 'Jaguar Land Rover India Limited',
 'Jindal Medianet',
 'Jiwane Oracle',
 'Jlr',
 'Jp Morgan Chase',
 'Jsw',
 'Kane American Express',
 'Kumar Samsung Noida',
 'Mckinsey  Company',
 'Medianet',
 'Microsoft India',
 'Mondelez International',
 'Morgan Stanley',
 'N Bny Mellon',
 'National Payments Corporation Of India Npci',
 'Nestle',
 'Nk Security',
 'Nobroker',
 'Nobroker Technologies',
 'Nvidia',
 'Optiver',
 'Optiverquant Role',
 'Quadeye',
 'Quantbox',
 'R Qualcomm',
 'Raj Databricks',
 'Rubrik',
 'Sachan Ibm',
 'Sachs',
 'Saluja Mastercard',
 'Samplytics Technologies Private Limited',
 'Samsung Rd Banglore',
 'Samsung Research Bangalore',
 'Samsung South Korea',
 'Sprinklr',
 'Standard Chartered',
 'Tata Steel',
 'Texas Instruments',
 'Tiwari De Shaw',
 'Tomar Jlr',
 'Tower Research Capital',
 'Trexquant Investment Llp',
 'Uber',
 'Vedanta Resources Limited',
 'Winzo Games']

    # your existing intern company list

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
        response = qa_chain.invoke({"query": q})
        full_response += f"**Q: {q}**\n{response['result']}\n\n"
        for doc in response.get("source_documents", []):
            all_sources.add(doc.page_content.strip())

    sources_text = "\n---\n".join(all_sources) if all_sources else "*No source documents found.*"
    return full_response.strip()

# Update company dropdown based on selection
def update_company_list(purpose):
    if purpose == "Placement":
        return gr.Dropdown.update(choices=placement_company_list, value=None, interactive=True)
    elif purpose == "Intern":
        return gr.Dropdown.update(choices=intern_company_list, value=None, interactive=True)
    return gr.Dropdown.update(choices=[], value=None, interactive=False)


# --- 🧱 Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("## 🤖 Company Interview Chatbot (IITK Insights)")

    with gr.Row():
        purpose = gr.Dropdown(choices=["Placement", "Intern"], label="🎯 Purpose (Placement or Intern)",value ="Placement")
        company_name = gr.Dropdown(choices=placement_company_list, label="🏢 Select Company", interactive=True)

    # Inline logic using lambda + if-else
    purpose.change(
        fn=lambda p: gr.update(
            choices=placement_company_list if p == "Placement"
            else intern_company_list if p == "Intern"
            else [],
            value=None,
            interactive=True
        ),
        inputs=purpose,
        outputs=company_name
    )

    with gr.Row():
        query_checkboxes = gr.CheckboxGroup(choices=["Sample Interview Questions", "Interview Process", "Resources", "Advice"], label="🔎 Select What You Want to Know")

    question = gr.Textbox(label="💬 Or Ask Custom Question", placeholder="e.g. Give me Sample Interview Questions for Google?", lines=1)
    submit = gr.Button("🔍 Ask")
    answer = gr.Markdown(label="✅ Answer")
    sources = gr.Markdown(label="📚 Source Documents")

    submit.click(
        fn=ask_bot,
        inputs=[company_name, query_checkboxes, question],
        outputs=[answer]
    )


# --- 🚀 Launch ---
if __name__ == "__main__":
    public_url = ngrok.connect(7860)
    print("🔗 Public URL:", public_url)
    iface.launch(share=True)