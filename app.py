import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import os
from ai71 import AI71
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AI71_API_KEY = "YOUR_API"

genai.configure(api_key=GOOGLE_API_KEY)

# Define the function to load the vector store
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Define the function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Define the function to handle user input
def handle_user_query(user_question):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # initial response generator agent
    initial_response = response.get("output_text", "No response generated.")
    # detailed response generator agent
    detailed_response = generate_detailed_response(initial_response, user_question)
    # translator agent
    urdu_response = generate_urdu_response(detailed_response)
    
    return detailed_response, urdu_response

def clean_detailed_answer(response_text):
    # Remove the "Reply:" prefix at the start
    response_text = re.sub(r'^Reply:\s*', '', response_text, flags=re.IGNORECASE)
    
    # Remove the "User:" suffix at the end (if applicable)
    response_text = re.sub(r'\s*User:\s*$', '', response_text, flags=re.IGNORECASE)
    
    return response_text

# Define the function to generate a detailed response using Falcon LLM with streaming
def generate_detailed_response(initial_response, question):
    prompt = f"""
    Provide a detailed and relevant explanation based on the initial response. Avoid any apologies or unnecessary prefaces.

    Initial Response:
    {initial_response}

    Question:
    {question}

    Detailed Answer:
    """
    
    detailed_answer = ""
    for chunk in AI71(AI71_API_KEY).chat.completions.create(
        model="tiiuae/falcon-180b-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            detailed_answer += chunk.choices[0].delta.content
            # Optionally, print each chunk for debugging
            print(chunk.choices[0].delta.content, sep="", end="", flush=True)
    cleaned_answer = clean_detailed_answer(detailed_answer)
    return cleaned_answer

# Define the function to generate a response in Urdu using Falcon LLM
def generate_urdu_response(english_text):
    prompt = f"""
    Translate the following text into Urdu while preserving the meaning and details.

    English Text:
    {english_text}

    Urdu Translation:
    """
    
    urdu_response = ""
    for chunk in AI71(AI71_API_KEY).chat.completions.create(
        model="tiiuae/falcon-180b-chat",
        messages=[
            {"role": "system", "content": "You are a translation assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    ):
        if chunk.choices[0].delta.content:
            urdu_response += chunk.choices[0].delta.content
            # Optionally, print each chunk for debugging
            print(chunk.choices[0].delta.content, sep="", end="", flush=True)
    
    return urdu_response

# Define the main function for Streamlit app
def main():
    st.set_page_config("Chat with PDF")
    st.header("ASK about economic studies")

    # Initialize session state if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Load the vector store initially
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = load_vector_store()
    
    # Text input for user query
    user_question = st.text_input("Ask a Question")

    if st.button("Generate Response"):
        if user_question:
            with st.spinner('Generating response, please wait...'):
                english_response, urdu_response = handle_user_query(user_question)
                st.markdown("**English Response:**")
                st.write(english_response)
                st.markdown("**Urdu Translation:**")
                st.write(urdu_response)
                # Add new query and response at the beginning of the history
                st.session_state.history.insert(0, {
                    'user_question': user_question,
                    'english_response': english_response,
                    'urdu_response': urdu_response
                })

    # Display the history
    if st.session_state.history:
        st.subheader("***----------------------------Response History----------------------------***")
        for entry in st.session_state.history:
            st.markdown("**User's Question:**")
            st.write(entry['user_question'])
            st.markdown("**English Response:**")
            st.write(entry['english_response'])
            st.markdown("**Urdu Translation:**")
            st.write(entry['urdu_response'])

if __name__ == "__main__":
    main()
