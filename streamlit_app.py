import streamlit as st
import pandas as pd
import hashlib
import time
from txtai.embeddings import Embeddings

@st.cache_resource
def get_embeddings_model():
    return Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})

embeddings_model = get_embeddings_model()

def create_1d_string_list(data, cols):
    data_rows = data[cols].astype(str).values
    return [" ".join(row) for row in data_rows]

def get_data_hash(data):
    data_str = data.to_string()
    return hashlib.md5(data_str.encode()).hexdigest()

@st.cache_data
def index_data(data, search_field=None):
    if search_field:
        data_1d = data[search_field].astype(str).tolist()
    else:
        data_1d = create_1d_string_list(data, data.columns)
    
    total_items = len(data_1d)
    batch_size = 100
    
    progress_bar = st.progress(0)
    percentage_text = st.empty()
    
    embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
    
    for i in range(0, total_items, batch_size):
        batch = data_1d[i:i+batch_size]
        embeddings.index([(uid, text, None) for uid, text in enumerate(batch, start=i)])
        
        progress = (i + len(batch)) / total_items
        progress_bar.progress(progress)
        percentage = int(progress * 100)
        percentage_text.text(f"Indexing Progress: {percentage}%")
        
        time.sleep(0.01)

    percentage_text.text("Indexing Complete: 100%")
    
    return embeddings

@st.cache_data
def search_with_scores(_embeddings, query, limit):
    return _embeddings.search(query, limit=limit)

st.title("CSV File Query App")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file, encoding_errors="ignore")
    st.write("Data Preview:")
    st.write(st.session_state.data.head())

    if len(st.session_state.data) > 10000:
        st.warning("Large dataset detected. Processing may take longer.")

    search_type = st.radio("Search Type", ["All Fields", "Single Field"])

    search_field = None
    if search_type == "Single Field":
        search_field = st.selectbox("Select field to search", st.session_state.data.columns)

    if st.button("Index Data"):
        with st.spinner('Preparing to index data...'):
            st.session_state.embeddings = index_data(st.session_state.data, search_field)
        st.success('Indexing complete!')

if st.session_state.embeddings is not None:
    query = st.text_input("Enter Query", "")

    max_results = min(20, len(st.session_state.data))
    result_limit = st.number_input("Number of results", min_value=1, max_value=max_results, value=min(5, max_results))

    if query:
        try:
            st.write(f"Top {result_limit} results:")
            results_with_scores = search_with_scores(st.session_state.embeddings, query, result_limit)
            
            result_ids = [uid for uid, _ in results_with_scores]
            scores = [score for _, score in results_with_scores]
            
            result_df = st.session_state.data.iloc[result_ids].reset_index(drop=True)
            result_df['Similarity Score'] = scores
            
            result_df['Similarity Score'] = result_df['Similarity Score'].apply(lambda x: round(x, 4))
            
            st.write(result_df)
        except Exception as e:
            st.error(f"An error occurred: {e}")