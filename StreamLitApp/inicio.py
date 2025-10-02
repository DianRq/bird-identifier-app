import streamlit as st

st.markdown(
    """
    <style>
    .title-container {
        background: linear-gradient(90deg, #dda0dd 0%, #00ced1 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    </style>
    <div class="title-container">
    <h1>🐦Gestor de Aves del Sitio Ramsar Marismas Nacionales, Nayarit</h1>
    <p>Centro de Investigación Mao Mao 🌎</p>
    </div>
    
    """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align: center;">
    <p style="font-size: 18px;">
        Esta aplicación fue desarrollada para <strong>identificar y visualizar</strong> registros de aves en Sitio Ramsar Marismas Nacionales🌎<br><br>
        Usa un modelo de Inteligencia artificial llamado Mobilenet_v2, entrenado con aproximadamente <strong> 5 mil imágenes </strong> obtenidas de 
        <em>iNaturalist México y Google</em>.<br><br>
    </p>
</div>
          
            
---
            
### Intrucciones de uso: 
 
#### Con el identificador de aves:
- 🐤 Agrega una imagen de un ave que hayas visto en tu visita al Sitio Ramsar Marismas Nacionales y que se encuentre dentro del listado de entrenamiento📸
- ✔️ Cuando el modelo la reconozca se desplegará información de la especie y avistamientos en un mapa.
- ✖️ Si el modelo no logra compilar intenta nuevamente, tal vez tu link o imagen no se hayan cargado correctamente . 
#### Con el visualizador de aves: 
- 🌍 Explora en el mapa interactivo</strong> las observaciones de aves registradas en Marismas Nacionales.
- 📊 Maneja la información y diviértete
            
---
            """, unsafe_allow_html=True
)

especies = [
    'Anas crecca', 'Ardea alba', 'Ardea herodias', 'Buteogallus anthracinus',
    'Chloroceryle americana', 'Cochlearius cochlearius', 'Egretta tricolor',
    'Fulica americana', 'Nyctanassa violacea', 'Oxyura jamaicensis'
]
html_code = f"""
<div style="
    border: 2px solid #FF4B4B; 
    border-radius: 8px; 
    padding: 15px; 
    background-color: #0f3529;
    max-width: 600px;
    margin-bottom: 20px;
">
    <p font-weight: bold; font-size: 18px; margin-bottom: 15px;">
        Nota:
    </p>
    <p>Actualmente la aplicación se encuentra en desarrollo, por lo que es ilustrativa y es un primer esfuerzo para el desarrollo de una aplicación con datos del Centro Mao Mao.<br>
    Las especies usadas para el entrenamiento del modelo fueron:</p>
    <ul style="margin-top: 0; padding-left: 20px;">
        {''.join(f'<li>{especie}</li>' for especie in especies)}
    </ul>
    Estas especies se eligieron por contar con un mayor número de registros en iNaturalista México dentro del Sitio RAMSAR Marismas Nacionales<br>

</div>
"""

st.markdown(html_code, unsafe_allow_html=True)

logo="logo3.png" 
info_adicional="""
Link oficial iNaturalista:
<https://mexico.inaturalist.org/>
---
Link oficial eBird:
<https://ebird.org/home>
"""
st.sidebar.title("Información adicional:")
st.sidebar.info(info_adicional)
st.sidebar.image(logo)    

st.markdown( """
<div class="footer">
    <p>Desarrollado con TensorFlow y Streamlit con información de iNaturalista México</p>
    <p>Proyecto final Python para Futuro Digital </p>
    <p>Desarrollado por Centro de Investigación Mao Mao, Autora: Diana Rios </p>
</div>
""", unsafe_allow_html=True)

## para instalar las librerias: pip install -r requisitos.txt