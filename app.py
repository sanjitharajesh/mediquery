import streamlit as st
import time
from backend.rag.chain import get_rag_chain

# Page config
st.set_page_config(
    page_title="MediQuery - FDA Drug Assistant",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS with better visibility
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    .stTextInput > div > div > input {font-size: 16px;}
    
    .answer-box {
        background-color: #ffffff;
        color: #1e1e1e;
        padding: 25px;
        border-radius: 10px;
        margin-top: 20px;
        line-height: 1.8;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        white-space: pre-wrap;
    }
    
    .disclaimer {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin-top: 20px;
        font-size: 13px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.title("MediQuery")
st.markdown("**FDA Drug Information Assistant**")

# Global disclaimer at top
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Medical Disclaimer:</strong> This information is from FDA documents and is not a substitute for professional medical advice. Always consult your healthcare provider.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Example Queries")
    
    examples = [
        "What are the side effects of Adderall?",
        "What warnings exist for Accutane?",
        "What is Lipitor used for?",
        "Contraindications for Metformin",
        "Ibuprofen dosage information"
    ]
    
    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.query = example
    
    st.markdown("---")
    st.header("About")
    st.info("""
    **MediQuery** searches FDA drug labels using:
    - Hybrid search (BM25 + Vector)
    - Local LLM via Ollama
    - RAG with LangChain
    """)
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Main interface
query = st.text_input(
    "Ask a question about FDA drug information:",
    placeholder="e.g., What are the side effects of Adderall?",
    key="query_input",
    value=st.session_state.get('query', '')
)

search_button = st.button("Search", type="primary", use_container_width=True)

# Handle query
if search_button and query:
    with st.spinner("Searching FDA documents..."):
        try:
            chain = get_rag_chain()
            
            start_time = time.time()
            answer = chain.invoke(query, verbose=False)
            latency = (time.time() - start_time) * 1000
            
            # Add to history
            st.session_state.history.append({
                'question': query,
                'answer': answer,
                'latency': latency,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Clear query
            if 'query' in st.session_state:
                del st.session_state.query
            
            st.success("Query completed!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("**Troubleshooting:**\n- Make sure Ollama is running: `ollama serve`\n- Check if data is ingested")

# Display results
if st.session_state.history:
    st.markdown("---")
    
    latest = st.session_state.history[-1]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Time", f"{latest['latency']:.0f}ms")
    with col2:
        st.metric("Total Queries", len(st.session_state.history))
    with col3:
        avg_time = sum(h['latency'] for h in st.session_state.history) / len(st.session_state.history)
        st.metric("Avg Time", f"{avg_time:.0f}ms")
    
    st.markdown("---")
    st.subheader("Answer")
    
    # Display answer without individual disclaimer
    st.markdown(f"""
    <div class="answer-box">
    {latest['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Show history
    if len(st.session_state.history) > 1:
        st.markdown("---")
        with st.expander(f"Query History ({len(st.session_state.history)-1} previous)"):
            for i, item in enumerate(reversed(st.session_state.history[:-1]), 1):
                with st.container():
                    st.markdown(f"**{item['timestamp']}**")
                    st.markdown(f"**Q:** {item['question']}")
                    preview = item['answer'].split('\n')[0][:150]
                    st.text(preview + "...")
                    st.caption(f"{item['latency']:.0f}ms")
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Built with LangChain")
with col2:
    st.caption("ChromaDB + BM25")
with col3:
    st.caption("Powered by Ollama")