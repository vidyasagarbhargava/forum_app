import streamlit as st
import base64
from huggingface_hub import notebook_login
from byaldi import RAGMultiModalModel
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image
from io import BytesIO
import torch
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', 
)

upload_dir = "./doc"

# Set page layout to wide
st.set_page_config(layout="wide")

st.title("Colpali Based Multimodal RAG App")

# Create sidebar for configuration options
with st.sidebar:
    st.header("Configuration Options")
    
    # Dropdown for selecting Colpali model
    colpali_model = st.selectbox(
        "Select Colpali Model",
        options=["vidore/colpali", "vidore/colpali-v1.2"]
    )
    
    # Dropdown for selecting Multi-Model LLM
    multi_model_llm = st.selectbox(
        "Select Multi-Model LLM",
        options=["llama3.2"]
    )
    
    # File upload button
    uploaded_file = st.file_uploader("Choose a Document", type=["pdf"])

# Set device for Mac M1 compatibility
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# st.write(f"Using device: {device}")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Uploaded Document")
        save_path = os.path.join(upload_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved: {uploaded_file.name}")
        
        @st.cache_resource
        def load_models(colpali_model):
            RAG = RAGMultiModalModel.from_pretrained(colpali_model, verbose=10, device="mps")
            return RAG
        
        RAG = load_models(colpali_model)
        
        @st.cache_data
        def create_rag_index(image_path):
            RAG.index(
                input_path=image_path,
                index_name="image_index",
                store_collection_with_index=True,
                overwrite=True,
            )
        
        create_rag_index(save_path)
        
    with col2:
        # Text input for the user query
        text_query = st.text_input("Enter your text query")

        # Search and Extract Text button
        if st.button("Search and Extract Text"):
            if text_query:
                results = RAG.search(text_query, k=1, return_base64_results=True)
                
                image_data = base64.b64decode(results[0].base64)
                image = Image.open(BytesIO(image_data))
                st.image(image, caption="Result Image", use_container_width=True)
                response = client.chat.completions.create(
                model=multi_model_llm,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{results[0].base64}"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
                # print(response)
                output =response.choices[0].message.content
                st.subheader("Query with LLM Model")
                st.markdown(output, unsafe_allow_html=True)
            else:
                st.warning("Please enter a query.")
else:
    st.info("Upload a document to get started.")
