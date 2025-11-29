import streamlit as st
import os
from pathlib import Path
from agent import get_agent
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title='KnowledgeBase Agent', layout='wide')
st.title('KnowledgeBase Agent — OpenRouter')

st.markdown('Upload PDFs or TXT files to `./docs`, click **Ingest**, then ask questions.')

col1, col2 = st.columns([1,2])

with col1:
    uploaded = st.file_uploader('Upload PDF or TXT (multiple)', accept_multiple_files=True)
    if uploaded:
        os.makedirs('docs', exist_ok=True)
        for f in uploaded:
            fp = Path('docs') / f.name
            with open(fp, 'wb') as out:
                out.write(f.getbuffer())
        st.success(f'Saved {len(uploaded)} file(s) to ./docs')

    if st.button('Ingest documents'):
        try:
            st.info('Ingesting... this may take a while depending on docs and embeddings quota.')
            from ingest import ingest
            ingest()
            st.success('Ingestion complete.')
        except Exception as e:
            st.error(f'Ingestion failed: {e}')

with col2:
    query = st.text_input('Ask a question about your documents:')
    if st.button('Get Answer') and query:
        openrouter = os.getenv('OPENROUTER_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openrouter or not openai_key:
            st.error('Set OPENROUTER_API_KEY and OPENAI_API_KEY in Streamlit Secrets (TOML).')
        else:
            with st.spinner('Querying...'):
                agent = get_agent()
                ans = agent.answer(query)
            st.subheader('Answer')
            st.write(ans)
