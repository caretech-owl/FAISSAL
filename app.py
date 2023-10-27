import streamlit as st
from utils import df_from_doc, faiss_search, create_search_index, wrap_print, generate_context, load_llm, run_llm
import pathlib
from tempfile import NamedTemporaryFile

st.title("Entlassbriefanalyse")
st.subheader("mit LLama 2, FAISS und LangChain")

st.divider()

uploaded_file = st.file_uploader("Ein Dokument hochladen:", accept_multiple_files=False, type=['pdf', 'txt'])

if uploaded_file is not None:
    filetype = pathlib.Path(uploaded_file.name).suffix
    with NamedTemporaryFile(dir='.', suffix=filetype) as f:
        f.write(uploaded_file.getbuffer())
        f.flush()
        docs = df_from_doc.df_from_doc(f.name, str(filetype).replace(".", ""))

        with st.expander("Inhalt", expanded=True):
            f.seek(0)
            st.text(''.join(l.decode("utf-8") for l in f.readlines()))

    
    # model_name = st.selectbox("Sentence-Transformers Model for Embeddings:", ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"])
    model_name = "all-MiniLM-L6-v2"

    pkl = create_search_index.create_search_index(docs, model_name)
    # model_type = st.selectbox("Select LLM Type:", ["LLaMA-7B", "LLaMA-13B"])
    model_type = "LLaMA-7B"
    llm = load_llm.load_llm(model_type=model_type, model_path=f"./models/{model_type}.gguf")
    
    with st.form('user_form', clear_on_submit = False):
        question = st.text_input("Stellen Sie Ihre Frage: ", value = "")
        submit_button = st.form_submit_button(label="Frage stellen")
        context = generate_context.generate_context(pkl, question, model_name, num_results = 5)
        # st.write("Geschätzte Antwortlänge:", round(4/3*len(context.split())), "tokens", "\n")
        wrap_print.wrap_print(context)

    # context_dependency = st.selectbox("Select Context Dependence Level (set to low if the model is failing to generate context dependent answers):",
    #                                   ["low", "medium", "high"])

    if submit_button:
        with st.spinner(f"{model_type} generiert Antwort..."):
            output = run_llm.run_llm(llm, question, context)
            print(output)
            # answer = output["choices"][0]["text"].strip().replace("\"", "").split("\n")[0].split(r"([^Dr].")[0]
            answer = output["choices"][0]["text"].replace("\"", "").split("Ich")[0]
            st.success(f"Antwort: {answer}")
