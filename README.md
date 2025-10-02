# 🐦 Bird Identification & Visualization System for Marismas Nacionales Ramsar Site 🐦

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A comprehensive AI-powered web application for identifying and visualizing bird species in the Marismas Nacionales Ramsar Site, Nayarit, Mexico.


## 🌟 Features

- **🦜 AI Bird Identification**: Upload images or provide URLs for automatic species classification using MobileNetV2
- **🗺️ Interactive Mapping**: Visualize bird sightings on interactive maps with geospatial data
- **📊 Multi-Species Support**: Currently supports 10 bird species with training on 5,000+ images
- **🔍 Data Integration**: Combines data from iNaturalist Mexico and eBird platforms
- **🎯 Real-time Predictions**: Get instant species identification with confidence scores


## Installation

1. **Clone the repository**
```bash
    git clone https://github.com/DianRq/bird-identifier-app.git
    cd bird-identification-marismas
```
2. **Install dependencies** 
```python
    pip install -r requirements.txt
```
3. **Run the application** 
```python
    streamlit run app_aves.py
```
## 🦆 Supported Bird Species
The model currently identifies these 10 species:
| Scientific Name | Description          |
| :-------- | :------------------------- |
| *Anas crecca* | Green-winged Teal|
| *Ardea alba* | Great Egret|
| *Ardea herodias* | Great Blue Heron|
| *Buteogallus anthracinus* | Common Black Hawk|
| *Chloroceryle americana* | Green Kingfisher|
| *Cochlearius cochlearius* | Boat-billed Heron|
| *Egretta tricolor* | Tricolored Heron|
| *Fulica americana* | American Coot|
| *Nyctanassa violacea* | Yellow-crowned Night Heron |
| *Oxyura jamaicensis* | Ruddy Duck |

## 🎯 How to Use
**Bird Identification**🔎
1. Navigate to "Identificador de Aves" in the sidebar

2. Choose input method: Upload an image or provide a URL

3. Click "Analizar imagen" to process the image

4. View results: Species prediction with confidence score and map visualization
**Bird Visualization** 🗺️
1. Navigate to "Visualizador de Aves" in the sidebar

2. Select a species from the dropdown menu

3. Explore sightings on the interactive map

4. View observation details in the data table
## 🛠️Technical Details
**AI Model Architecture** 🤖
 - **Base Model:** MobileNetV2 (Transfer Learning)
 - **Input Size:** 224×224 pixels
 - **Training Data:** 5,000 images (500 per species)
 - **Data Augmentation:** Rotation, shifting, shearing, zoom, brightness variation
 - **Framework:** TensorFlow 2.19.0 with Keras

 **Data Sources** 🗃️
 - **Bird Images:** iNaturalist Mexico & Google
 - **Spatial Data:** Marismas Nacionales Ramsar Site boundaries
 - **Observation Records:** Field data and citizen science platforms

 **Key Technologies**📍

 - **Web Framework:** Streamlit 1.45.1
 - **Geospatial:** GeoPandas, Leafmap, Folium
 - **Image Processing:** OpenCV, Pillow
 - **Machine Learning:** TensorFlow, NumPy, Scikit-learn
## 📊 Model Training

The model was trained using:
```python
# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.15,
    shear_range=0.2,
    zoom_range=[0.7,1.3],
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    validation_split=0.2
)

# Model Architecture
modelo = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Lambda(lambda x: mobilnetv2(x)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])
```
##  🌍Environmental Context
This project focuses on the Marismas Nacionales Ramsar Site in Nayarit, Mexico - a crucial wetland ecosystem that provides habitat for numerous bird species and supports local biodiversity.
## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for:
- Adding new bird species
- Improving model accuracy
- Enhancing the user interface
- Adding new features


## License

This project was developed as part of the Python course of the Full Stack Diploma at Futuro Digital, Ingeniería Condor by [Diana Rios ](www.linkedin.com/in/dianariosq)


## 🙏 Acknowledgments
**AI Model Architecture**
 - **iNaturalist Mexico** for bird observation data
 - **eBird** for additional bird sighting records
 - **TensorFlow Hub** for the MobileNetV2 model
 - **Ramsar Convention** for wetland conservation framework



## Contact

- Support: email diana.riosq@gmail.com
- For questions or collaborations, please open an issue in this repository. 

🐦 Protect Birds • 🌿 Conserve Habitats • 💻 Innovate for Nature
