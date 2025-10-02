import streamlit as st


st.set_page_config(layout="wide") 
                    # page_title="Centro Mao Mao",
                    # page_icon="🐦",
                    # initial_sidebar_state="expanded")


st.logo("logo3.png")

pages=[
    st.Page("inicio.py", title="Inicio", icon="🏠", default=True),
    st.Page("identificador_v02.py", title="Identificador de Aves", icon="🔎"),
    st.Page("visualizador.py", title="Visualizador de Aves", icon="🌎"),
]

pg = st.navigation(pages, position="sidebar", expanded=True)
# Ejecuta la página seleccionada
pg.run()

# Estilo CSS personalizado para una interfaz moderna de IA
st.markdown("""
<style>
    .main {
        background-color: #EEAECA;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
    .bird-classification {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)
# Título y descripción


#Sidebar para la barra de opciones
# if "menu" not in st.session_state:
#     st.session_state.menu = "identificador"

# with st.sidebar:
#     st.session_state.menu =  st.selectbox("Menu", ["Identificador", "Visualizador"])

# if st.session_state.menu == "Identificador":
#     iden()
# elif st.session_state.menu == "Visualizador":
#     vis()

# Pie de página
