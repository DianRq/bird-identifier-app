import streamlit as st
import pandas as pd
import geopandas as gdp
import leafmap.foliumap as leafmap
st.title("Visualizador de Aves en el Sitio Ramsar Marismas Nacionales")
logo="logo3.png" 
info_adicional="""Link oficial iNaturalista:
<https://mexico.inaturalist.org/>
---
Link oficial eBird:
<https://ebird.org/home>
"""
st.sidebar.title("Información adicional:")
st.sidebar.info(info_adicional)
st.sidebar.image(logo)
#Cargado de datos
ruta_Marismas='datosEspaciales/delimitacion_MarismasNacionales.shp'
ruta_aves_marismas='datosEspaciales/base_aves_marismas.shp'
registros_aves_Marismas=gdp.read_file(ruta_aves_marismas)
delimitacion_Marismas=gdp.read_file(ruta_Marismas)

#eliminación de columnas que no necesitamos de la base de datos
registros_aves_Marismas.drop(columns=['url', 'image_url', 'latitude', 'longitude'], inplace=True)

#Selección de la especie que deseo visualizar tanto en la tabla como en el mapa
seleccion_especie=st.selectbox("Selecciona la especie que deseas visualizar",
                               options=registros_aves_Marismas['scientific'].values)

#Variable de filtrado de especies
filtrado= registros_aves_Marismas[registros_aves_Marismas['scientific']==seleccion_especie]

st.subheader(f"Registros de la especie: {seleccion_especie} en Marismas Nacionales")
st.write(f"Resultados encontrados: {len(filtrado)}")



filtrado['observed_o'] = filtrado['observed_o'].astype(str)

cambio_nombres= {'observed_o':'Fecha de observación',
                 'scientific':'Nombre científico',
                 'common_nam':"Nombre común",
                 'tipo':'Tipo',
                 'Municipio':'Ubicación'}
tabla_actualizada=filtrado.rename(columns=cambio_nombres)

#Código para declarar el mapa en nuestra ventana
model_map = leafmap.Map(minimap_control=True)
model_map.add_gdf(tabla_actualizada, layer_name=f"Avistamientos de {seleccion_especie}")

#Codigo del mapa base
drop_columnas=delimitacion_Marismas.drop(columns=['OBJECTID', 'Shape_Leng', 'Shape_Area'])
model_map.add_basemap("NASAGIBS.BlueMarble")
model_map.add_gdf(drop_columnas,
                   layer_name="Sitio Ramsar Marismas Nacionales")
model_map.to_streamlit(height=400)