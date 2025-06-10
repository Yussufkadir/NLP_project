import streamlit as st
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from peft import PeftModel
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(
    page_title="Mental Health Chatbot", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.sidebar.info(f"Using device: {DEVICE}")

@st.cache_resource
def load_model_and_tokenizer():
    st.sidebar.text("Loading the model and tokenizer")
    model_name = "google/flan-t5-large"

    tokenizer_name = "syurmen/T5-finetuned"
    fine_tuned_model = "syurmen/T5-finetuned"

    try:
        base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

        model = PeftModel.from_pretrained(base_model, fine_tuned_model)
        model = model.merge_and_unload()

        model.eval()
        model.to(DEVICE)

        st.sidebar.text("Models and tokenizers loaded.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model and tokenizer")
        st.stop()

@st.cache_resource
def setup_emergency():
    emergency_resources = {
        "suicide_prevention": {
            "hotline": "112 or 800 70 2222 (psychological crisis line)",
            "website": "www.telefonzaufania.org"
        },
        "response": """I notice you mentioned something concerning. If you're having thoughts of harming yourself, please know that help is available. 

**Immediate resources:**
- National Suicide Prevention Lifeline: 112 or 800 70 2222 (available 24/7)
- Or go to your nearest emergency room

Would you like me to provide more specific resources or someone to talk to right now?

Remember, you're not alone, and trained professionals are ready to help."""
    }
    return emergency_resources
@st.cache_resource
def setup_rag(_direct_model, _tokenizer):
    st.sidebar.text("Setting up rag..")
    books = "./books"
    
    if not os.path.exists(books) or not os.listdir(books):
        st.sidebar.warning(f"book directory is not foudn or it is empty.")
        return None
    
    try:
        loader = PyPDFDirectoryLoader(books)
        documents = loader.load()

        if not documents:
            st.sidebar.warning("there are no documents to set up RAG")
            return None
        st.sidebar.text("loadaed {documents} for RAG")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        doc_chunks = splitter.split_documents(documents)
        if not doc_chunks:
            st.sidebar.warning("Document splitting did not do any chunking")
            return None
        st.sidebar.text(f"Split documents into {len(doc_chunks)} chunks")

        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device":DEVICE}
        )

        store = FAISS.from_documents(doc_chunks, embeddings)
        retriever = store.as_retriever(search_kwargs={"k":3})

        rag_llm_pipe = pipeline(
            "text2text-generation",
            model=_direct_model,
            tokenizer=_tokenizer,
            device=0 if DEVICE.type=="cuda" else -1,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.4,
            truncation=True,
            max_length=1024
        )

        llm_rag = HuggingFacePipeline(pipeline=rag_llm_pipe)

        prompt_template_str = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and helpful.

Context: {context}

Question: {question}

Helpful Answer:"""
        QA_PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_rag,
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        st.sidebar.text("RAG pipeline created successfully.")
        return qa_chain
    except Exception as e:
        st.sidebar.error(f"Error setting up RAG pipeline: {e}")
        return None
    
def get_chatbot_response(user_query, direct_model_instance, tokenizer_instance, qa_chain_rag_instance, rag_keywords=None):
    if rag_keywords is None:
        rag_keywords = [
            "tell me more about", "explain", "what does the document say about", 
            "details on", "resources for", "coping mechanisms for", "information on"
        ]

        critical_keywords = [
            "kill", "death", "suicide", "suicidal", "lethal",
            "damage myself", "I want to end myself", "I want to bring an end to this",
            "I do not want to live anymore", "I dont want to live"
        ]
    
    use_rag = False
    use_emergency = False

    for keyword in critical_keywords:
        if keyword.lower() in user_query.lower():
            use_emergency = True
            break
    
    if use_emergency:
        st.sidebar.warning("Emergency content detected - providing crisis resources")
        emergency_resources = setup_emergency()
        return emergency_resources["response"]

    if qa_chain_rag_instance:
        for keyword in rag_keywords:
            if keyword.lower() in user_query.lower():
                use_rag = True
                break
    
    response_text = ""
    source_info_docs_info = None

    if use_rag and qa_chain_rag_instance:
        st.sidebar.info("Attempting RAG pipeline...")
        try:
            rag_result = qa_chain_rag_instance.invoke({"query":user_query})
            response_text = rag_result.get("result", "No specific information found in documents.")
            if rag_result.get("source_documents"):
                source_info_docs_info = [doc.metadata.get("source", "Unknown source") for doc in rag_result["source_documents"]]
                st.sidebar.write("RAG Sources", source_info_docs_info)

        except Exception as e:
            st.sidebar.error(f"Error during rag query: {e}")
            response_text = "Sorry, I had trouble finding detailed information. Let me try a general answer."
            use_rag=False
    
    if not use_rag or not response_text:
        if not use_rag:
            st.sidebar.info("using direct model")
        
        try:
            inputs = tokenizer_instance(user_query, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            max_len = len_balancer(user_query)

            with torch.no_grad():
                outputs = direct_model_instance.generate(
                    **inputs,
                    max_length=max_len,
                    num_beams=5,
                    early_stopping=False,
                    do_sample=True,
                    temperature=0.6,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    length_penalty=1.5
                )
            direct_answer = tokenizer_instance.decode(outputs[0], skip_special_tokens=True)

            if not direct_answer.strip() and qa_chain_rag_instance and not response_text:
                st.sidebar.info("Direct model repsonse was empty, trying RAG as fallback...")
                try:
                    rag_result = qa_chain_rag_instance.invoke({"query":user_query})
                    response_text = rag_result.get("result", "I'm not sure how to answer that.")
                    if rag_result.get("source_documents"):
                        source_docs_info = [doc.metadata.get("source", "Unknown source") for doc in rag_result["source_documents"]]
                        st.sidebar.write("RAG fallback sources:", source_docs_info)
                except Exception as e:
                    st.sidebar.error(f"Error during RAG fallback: {e}")
                    response_text = "I'm having trouble finding an answer right now."
            elif direct_answer.strip():
                response_text = direct_answer
            elif not direct_answer.strip() and not response_text:
                response_text = "I'm not sure how to respond to that. Could you please rephrase?"
            
        except Exception as e:
            st.sidebar.error(f"Error during direct model query: {e}")
            response_text = "Sorry, I encountered an error processing your request."
    if not response_text.strip():
        response_text = "I'm unable to provide a response at this moment. Please try again later or rephrase your question."
    response_text = remove_repetitions(response_text)
    return response_text

def remove_repetitions(text):
    sentences = text.split('. ')
    unique_sentences = []
    
    for sentence in sentences:
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)

    cleaned_text = '. '.join(unique_sentences)
    if text.endswith('.'):
        cleaned_text += '.'
    return cleaned_text

def len_balancer(query):
    tokens = len(query.split())

    if "explain" in query.lower() or "details" in query.lower():
        return 1000
    elif tokens < 8:
        return 600
    else:
        return 800


def main():
    st.title("Mental Health Support Chatbot")
    st.caption("This chatbot can provide general information and support. It is not a replacement for professional medical advice. Type 'exit' or 'quit' to end the session.")

    with st.expander("Emergency Resources", expanded=False):
        st.markdown("""
        **If you're experiencing a mental health emergency:**
        - National Suicide Prevention Lifeline: 112 or 800 70 2222
        - Call your local emergency services: 112 (EU)
        - Go to your nearest emergency room
        """)
       
    with st.spinner("Initializing chatbot... This might take a moment on first run."):
        direct_model_instance, tokenizer_instance = load_model_and_tokenizer()
        qa_chain_rag_instance = setup_rag(direct_model_instance, tokenizer_instance)

    if qa_chain_rag_instance is None:
        st.sidebar.error("RAG system not available. Chatbot will rely on general knowledge only.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I support you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    if prompt := st.chat_input("Type your message here..."):
        if prompt.lower() in ["exit", "quit"]:
            st.info("Session ended. Thank you for chatting!")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                full_response = get_chatbot_response(
                    prompt,
                    direct_model_instance,
                    tokenizer_instance,
                    qa_chain_rag_instance
                )
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()