import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image 
import requests
from io import BytesIO
import leafmap.foliumap as leafmap
import geopandas as gpd
import matplotlib.pyplot as plt

# Importar tensorflow_hub si está disponible
try:
    import tensorflow_hub as hub
    TF_HUB_AVAILABLE = True
except ImportError:
    st.warning("TensorFlow Hub no está disponible. Algunas funciones pueden estar limitadas.")
    TF_HUB_AVAILABLE = False

st.title("Identificador de Aves en el sitio Ramsar Marismas Nacionales: Fase 1")
st.markdown("---")
st.markdown("""
<style>
    .title-container {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
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

# Carga de shapefiles 
ruta_Marismas = 'datosEspaciales/delimitacion_MarismasNacionales.shp'
ruta_aves = 'datosEspaciales/registros_avesMarismas.shp'

try:
    delimitacion_Marismas = gpd.read_file(ruta_Marismas)
    registros_Aves = gpd.read_file(ruta_aves)
except Exception as e:
    st.error(f"Error al cargar los archivos espaciales: {e}")
    delimitacion_Marismas = None
    registros_Aves = None

registros_Aves.drop(columns=['id', 'species_gu', 'iconic_tax', 'url', 'image_url', 'latitude', 'longitude'], inplace=True)
registros_Aves['observed_o'] = registros_Aves['observed_o'].astype(str)
cambio_nombres= {'observed_o':'Fecha de observación',
                 'scientific':'Nombre científico',
                 'common_nam':"Nombre común",
                 'tipo':'Tipo',
                 'Municipio':'Ubicación'}
tabla_actualizada=registros_Aves.rename(columns=cambio_nombres)
# Código para declarar el mapa en nuestra ventana
model_map = leafmap.Map(minimap_control=True)

# listado de clases - DEBE coincidir exactamente con el entrenamiento
clases = [
    'Anas crecca',
    'Ardea alba', 
    'Ardea herodias',
    'Buteogallus anthracinus',
    'Chloroceryle americana',
    'Cochlearius cochlearius',
    'Egretta tricolor',
    'Fulica americana',
    'Nyctanassa violacea',
    'Oxyura jamaicensis'
]

# función para filtrar los registros con base en la especie identificada
def filtrar_registros_por_especie(registros, especie_predicha):
    if registros is not None:
        return registros[registros['Nombre científico'].str.lower() == especie_predicha.lower()]
    return None

# función para mostrar a la especie predicha en el mapa que ya tenemos
def mapa_filtrado(especie_predicha):
    if tabla_actualizada is not None:
        filtrado = filtrar_registros_por_especie(tabla_actualizada, especie_predicha)
        if filtrado is not None and not filtrado.empty:
            model_map.add_gdf(filtrado, layer_name=f'Avistamientos de {especie_predicha}')
        else:
            st.warning(f"No se encontraron registros para {especie_predicha}")

# Código para cargar el modelo en la aplicación
@st.cache_resource
def load_model():
    try:
        # Método 1: Cargar con configuraciones personalizadas para TensorFlow Hub
        model = tf.keras.models.load_model(
            "modelos/aves_classifyer.h5",
            custom_objects={'KerasLayer': tf.keras.utils.get_custom_objects().get('KerasLayer', None)},
            compile=False
        )
        
        # Recompilar manualmente
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
        st.success("✅ Modelo cargado exitosamente con TensorFlow Hub")
        return model
        
    except Exception as e:
        st.warning(f"Método 1 falló: {e}")
        
        # Método 2: Recrear la arquitectura y cargar pesos
        try:
            st.info("Intentando recrear la arquitectura del modelo...")
            model = recreate_model_architecture()
            
            # Cargar solo los pesos
            model.load_weights("modelos/aves_classifyer.h5")
            st.success("✅ Modelo recreado y pesos cargados exitosamente")
            return model
            
        except Exception as e2:
            st.warning(f"Método 2 falló: {e2}")
            
            # Método 3: Usando tf.keras.utils.get_file si es necesario
            try:
                # Intentar cargar sin custom_objects
                model = tf.keras.models.load_model("modelos/aves_classifyer.h5", compile=False)
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                st.success("✅ Modelo cargado con método básico")
                return model
                
            except Exception as e3:
                st.error(f"❌ Todos los métodos fallaron:")
                st.error(f"Error 1: {e}")
                st.error(f"Error 2: {e2}")
                st.error(f"Error 3: {e3}")
                
                # Mostrar información adicional para debugging
                show_model_debug_info()
                return None

def recreate_model_architecture():
    """
    Recrea la arquitectura exacta de tu modelo entrenado.
    Basado en tu código de entrenamiento.
    """
    if not TF_HUB_AVAILABLE:
        st.error("TensorFlow Hub no está disponible. No se puede recrear el modelo.")
        return None
        
    try:
        import tensorflow_hub as hub
        
        # Suprimir warnings temporalmente
        import warnings
        warnings.filterwarnings('ignore')
        
        # URL del modelo MobileNetV2 que usaste
        url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        
        st.info("Descargando MobileNetV2 desde TensorFlow Hub...")
        
        # Crear la capa de TensorFlow Hub
        mobilnetv2 = hub.KerasLayer(url, input_shape=(224, 224, 3))
        mobilnetv2.trainable = False  # Como en tu entrenamiento
        
        # Recrear el modelo exacto que entrenaste
        modelo = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Lambda(lambda x: mobilnetv2(x)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax")  # 10 clases como en tu entrenamiento
        ])
        
        # Compilar con los mismos parámetros
        modelo.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        st.success("✅ Arquitectura del modelo recreada exitosamente")
        return modelo
        
    except Exception as e:
        st.error(f"❌ Error al recrear arquitectura: {e}")
        st.info("💡 Esto puede deberse a:")
        st.markdown("""
        - Problemas de conectividad a internet
        - Incompatibilidades de versiones de TensorFlow
        - Falta TensorFlow Hub
        """)
        return None

def show_model_debug_info():
    """Muestra información de debugging del modelo"""
    import os
    
    st.subheader("🔍 Información de debugging:")
    
    # Verificar archivo
    model_path = "modelos/aves_modelo_clean.h5"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        st.info(f"✅ Archivo encontrado: {size:,} bytes")
    else:
        st.error(f"❌ Archivo no encontrado: {model_path}")
        st.error("Verifica que el archivo esté en la ruta correcta")
    
    # Información de TensorFlow
    st.info(f"Versión de TensorFlow: {tf.__version__}")
    
    # Sugerencias
    st.markdown("""
    **Posibles soluciones:**
    1. Asegúrate de tener conexión a internet (para TensorFlow Hub)
    2. Verifica que el archivo del modelo esté en `modelos/aves_modelo_clean.h5`
    3. Intenta entrenar y guardar el modelo nuevamente
    4. Considera usar el formato SavedModel en lugar de .h5
    """)

# Función alternativa para crear modelo simple si TensorFlow Hub falla
def create_simple_model():
    """
    Crea un modelo CNN simple como respaldo si TensorFlow Hub no está disponible.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 clases
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Código para procesar las imágenes que el usuario cargue
def preprocess_image(image):
    """
    Preprocesa la imagen para que coincida con el formato usado en el entrenamiento.
    Durante el entrenamiento usaste: rescale=1./255
    """
    # Convertir a array numpy y normalizar (igual que en entrenamiento)
    img = np.array(image).astype(float) / 255.0
    
    # Redimensionar a 224x224 (igual que en entrenamiento)
    img = cv2.resize(img, (224, 224)) 
    
    # Asegurar que tenga 3 canales (RGB)
    if len(img.shape) == 2:  # Si es escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Si tiene canal alpha (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Reshape para el modelo (añadir dimensión de batch)
    return img.reshape(-1, 224, 224, 3)

# código para clasificar las imágenes
def clasificador_imagen(model, image):
    imagen_procesada = preprocess_image(image)
    prediccion = model.predict(imagen_procesada)
    resultado = np.argmax(prediccion[0], axis=-1)
    confidence = float(prediccion[0][resultado])
    return resultado, confidence

st.markdown("""
<div class="title-container">
    <h1>🐣 Birdy Identifier</h1>
    <p><strong>Iniciativa de IA para detectar especies de aves observadas en el Sitio RAMSAR Marismas Nacionales</strong></p>
</div>
""", unsafe_allow_html=True)

modelo = load_model()

# Si no se puede cargar el modelo, usar un modelo mock para testing
if modelo is None:
    st.error("⚠️ No se pudo cargar el modelo entrenado")
    st.info("💡 **Soluciones sugeridas:**")
    st.markdown("""
    1. **Regenerar el modelo**: En Colab, guarda el modelo así:
       ```python
       modelo.save("aves_modelo_clean", save_format='tf')  # Formato SavedModel
       ```
    
    2. **Instalar TensorFlow Hub**:
       ```bash
       pip install tensorflow-hub
       ```
    
    3. **Verificar archivos**: Asegúrate de que `aves_modelo_clean.h5` esté en `modelos/`
    """)
    
    # Crear un modelo mock para testing de la interfaz
    if st.checkbox("🧪 Usar modelo de prueba (predicciones aleatorias)"):
        class MockModel:
            def predict(self, x):
                # Retorna predicciones aleatorias para testing
                return np.random.rand(1, 10)
        
        modelo = MockModel()
        st.warning("🧪 Usando modelo de prueba - las predicciones son aleatorias")

if modelo:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Opciones de entrada de la información
    opcion = st.radio("Selecciona como quieres analizar tu imagen: ",
                      ("Subiendo una imagen", "Agregando un link"))
    imagen = None

    # Código para subir una imagen
    if opcion == "Subiendo una imagen":
        carga_imagen = st.file_uploader("Sube la imagen del ave que deseas identificar 📷",
                                        type=["jpg", "jpeg", "png"])
        if carga_imagen is not None:
            imagen = Image.open(carga_imagen).convert("RGB")  # Para que no tengamos problema
            st.image(imagen, caption="Imagen subida correctamente", use_column_width=True)
    else:
        url = st.text_input("Ingresa la url de la especie de ave que deseas conocer")
        if url:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Lanza excepción si hay error HTTP
                imagen = Image.open(BytesIO(response.content)).convert("RGB")
                st.image(imagen, caption="Imagen cargada correctamente desde la url", use_column_width=True)
            except Exception as e:
                st.error(f"Error al cargar la imagen desde la URL proporcionada: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Clasificador
    if imagen is not None:
        if st.button("Analizar imagen", key="analyze"):
            with st.spinner("Analizando tu imagen 🤖"):
                # Código para realizar la predicción
                resultado, confidence = clasificador_imagen(modelo, imagen)
                # el resultado es un entero el cual representa el índice de la clase con mayor probabilidad
                # que devuelve el modelo
                prediccion = clases[resultado]

                st.markdown(f"""
                <div class="result-card bird-classification">
                    <h2>🦜 La especie predicha fue: <b>{prediccion}</b></h2>
                    <p>La confianza de la predicción fue de: {confidence:.2%}</p>
                </div>          
                """, unsafe_allow_html=True)
                
                mapa_filtrado(prediccion)
else:
    st.error("No se pudo cargar el modelo. Asegúrate de que el archivo del modelo esté disponible")

# Separadores 
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("### Visualización espacial 🌐")

if delimitacion_Marismas is not None:
    model_map.add_basemap("NASAGIBS.BlueMarble")
    model_map.add_gdf(delimitacion_Marismas,
                      layer_name="Sitio Ramsar Marismas Nacionales")
    model_map.to_streamlit(height=500) 
else:
    st.error("No se pudieron cargar los datos espaciales")

# Explicación del modelo
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("¿Cómo funciona?")
st.write("""
Este sistema utiliza una red neuronal convolucional (CNN) entrenada para clasificar
imágenes de ciertas especies de aves. El modelo ha sido entrenado con cerca de 13 mil imágenes de 28 especies diferentes de aves
y basa su estimación en características visuales
como color, textura y forma.
""")

logo = "logo3.png" 
info_adicional = """Link oficial iNaturalista:
<https://mexico.inaturalist.org/>
---
Link oficial eBird:
<https://ebird.org/home>
"""
st.sidebar.title("Información adicional:")
st.sidebar.info(info_adicional)

# Verificar si el logo existe antes de intentar cargarlo
try:
    st.sidebar.image(logo)
except Exception as e:
    st.sidebar.warning("No se pudo cargar el logo")

try:
    divider01 = Image.open("divider_v02.jpg")
    st.image(divider01, use_container_width=True)
except Exception as e:
    st.warning("No se pudo cargar la imagen divisoria")

st.markdown('</div>', unsafe_allow_html=True)