```markdown
# 🧠 TextInsight 2.0 – Intelligent NLP Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57.svg)](https://sqlite.org/)

> **The NLP classifier that learns, explains, and improves with you.**  
> **El clasificador NLP que aprende, explica y mejora contigo.**

---

## 📖 Description (English)

**TextInsight 2.0** is a complete Natural Language Processing (NLP) tool designed for junior devs who want to understand and apply machine learning in real‑world scenarios. It goes beyond a simple script:

- **🧹 Smart preprocessing** – removes noise (URLs, mentions, numbers, punctuation) automatically.
- **🤖 AutoML Lite** – compares Naive Bayes vs. Logistic Regression and picks the best model for your data.
- **🔍 Explainability** – shows the top 3 keywords that influenced each prediction.
- **📦 Batch processing** – classifies hundreds of texts from a CSV file in seconds.
- **🧠 Active learning** – allows you to correct wrong predictions and saves the new examples for retraining.
- **📊 Advanced dashboard** – displays class distribution and average confidence per label.

It stores everything in a lightweight **SQLite** database, persists trained models with **pickle**, and provides an interactive CLI menu.

---

## 📖 Descripción (Español)

**TextInsight 2.0** es una herramienta completa de Procesamiento de Lenguaje Natural (NLP) diseñada para **desarrolladores junior** que quieren entender y aplicar machine learning en casos reales. Va más allá de un simple script:

- **🧹 Preprocesamiento inteligente** – elimina ruido (URLs, menciones, números, puntuación) automáticamente.
- **🤖 AutoML Lite** – compara Naive Bayes vs. Regresión Logística y elige el mejor modelo para tus datos.
- **🔍 Explicabilidad** – muestra las 3 palabras clave que influyeron en cada predicción.
- **📦 Procesamiento por lotes** – clasifica cientos de textos desde un archivo CSV en segundos.
- **🧠 Aprendizaje activo** – permite corregir predicciones erróneas y guarda los nuevos ejemplos para reentrenar.
- **📊 Dashboard avanzado** – muestra la distribución de clases y la confianza promedio por etiqueta.

Todo se almacena en una ligera base de datos **SQLite**, los modelos entrenados se persisten con **pickle** y se ofrece un menú interactivo por consola.

---

## 🚀 Key Features / Características principales

| Feature (EN) | Característica (ES) |
|--------------|----------------------|
| 🧹 Smart text cleaning | 🧹 Limpieza inteligente de texto |
| 🤖 Automatic model selection (Naive Bayes vs LogReg) | 🤖 Selección automática del mejor modelo (Naive Bayes vs LogReg) |
| 🔍 Explainability with top‑3 keywords | 🔍 Explicabilidad con las 3 palabras clave |
| 📦 Batch classification from CSV | 📦 Clasificación por lotes desde CSV |
| 🧠 Active learning (on‑the‑fly correction) | 🧠 Aprendizaje activo (corrección en caliente) |
| 📊 Dashboard: class balance & confidence | 📊 Dashboard: balance de clases y confianza |
| 💾 Persistent model storage (pickle) | 💾 Almacenamiento persistente del modelo (pickle) |
| 🗄️ SQLite database for training and predictions | 🗄️ Base de datos SQLite para entrenamiento y predicciones |

---
```
## 📂 Project Structure / Estructura del proyecto

```
textinsight/
├── textinsight_v2.py          # Main application (clase TextInsight)
├── textinsight.db             # SQLite database (creada automáticamente)
├── models/                    # Carpeta con los modelos guardados
│   ├── vectorizer.pkl
│   ├── classifier.pkl
│   └── labels.pkl
├── data/                      # (Opcional) tus archivos CSV
│   ├── train.csv
│   └── unlabeled.csv
└── README.md                  # Este archivo
```

---

## ⚙️ Installation / Instalación

### Prerequisites / Requisitos previos
- Python 3.8 or higher / Python 3.8 o superior
- pip (package manager)

### Steps / Pasos

1. **Clone the repository** / **Clona el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/textinsight.git
   cd textinsight
   ```

2. **Create a virtual environment (recommended)** / **Crea un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   # o en Windows: venv\Scripts\activate
   ```

3. **Install dependencies** / **Instala las dependencias**
   ```bash
   pip install pandas scikit-learn
   ```

4. **Run the application** / **Ejecuta la aplicación**
   ```bash
   python textinsight_v2.py
   ```

---

## 🧪 Quick Start / Inicio rápido

### 1. Prepare your training CSV / Prepara tu CSV de entrenamiento

Create a CSV file with at least two columns (e.g., `texto` and `etiqueta`):
```csv
texto,etiqueta
"El producto llegó en perfectas condiciones",positivo
"El servicio fue pésimo",negativo
"La entrega fue rápida pero el producto venía dañado",neutro
```

### 2. Load the data / Carga los datos
From the main menu, choose **option 6** and enter the file path and column names.

### 3. Train the model / Entrena el modelo
Choose **option 1** – the system will automatically test two algorithms and pick the best one.

### 4. Classify a text / Clasifica un texto
Choose **option 2**, type any text, and see the predicted category with confidence.

### 5. Correct if needed / Corrige si es necesario
If the prediction is wrong, answer **'n'** and provide the correct label – it will be saved for future retraining.

### 6. Explain a prediction / Explica una predicción
Choose **option 7** to see the top‑3 keywords that influenced the decision.

### 7. Process a batch / Procesa un lote
Choose **option 8** and pass a CSV with unlabeled texts – you’ll get a new CSV with predictions.

---

## 💡 Example Usage / Ejemplo de uso

**Training output / Salida de entrenamiento:**
```
🤖 EJECUTANDO AUTOML... Comparando algoritmos:
----------------------------------------
  Naive Bayes: Accuracy = 87.50%
  Regresión Logística: Accuracy = 91.67%
----------------------------------------
🏆 MODELO GANADOR: Regresión Logística con 91.67%
```

**Prediction with explanation / Predicción con explicación:**
```
Introduce el texto a analizar: El servicio fue excelente y muy rápido
🏷️  Categoría: positivo (confianza: 94.23%)
🔑 Palabras clave: excelente, rápido, servicio
```

---

## 🛠️ Technologies Used / Tecnologías utilizadas

- **Python 3.8+** – core language / lenguaje principal
- **scikit-learn** – TF‑IDF vectorization, classifiers (Naive Bayes, Logistic Regression) / vectorización TF‑IDF, clasificadores
- **pandas** – CSV handling and data manipulation / manejo de CSV y manipulación de datos
- **SQLite3** – lightweight database for training and predictions / base de datos ligera para entrenamiento y predicciones
- **pickle** – model serialization / serialización de modelos
- **re (regex)** – text cleaning / limpieza de texto

---

## 🔮 Future Improvements / Mejoras futuras

*I’m a junior developer and I’d love to grow this project. Here are some ideas:*  
*Soy un desarrollador junior y me encantaría hacer crecer este proyecto. Aquí algunas ideas:*

- 🌐 **Multilingual support** – detect language and use appropriate stopwords.
- 🧠 **Deep learning** – integrate with Hugging Face Transformers for even better accuracy.
- 🖥️ **Web interface** – build a simple Flask or Streamlit app.
- 📈 **Incremental learning** – update the model without retraining from scratch.
- 🧪 **Cross‑validation** – add k‑fold CV for more robust evaluation.
- 📦 **Dockerization** – create a Docker image for easy deployment.
- 📊 **More visualizations** – generate confusion matrix plots and word clouds.

---

## 🤝 Contributing / Contribuciones

Contributions are welcome! Feel free to open an issue or submit a pull request.  
¡Las contribuciones son bienvenidas! No dudes en abrir un issue o enviar un pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License / Licencia

Distributed under the MIT License. See `LICENSE` for more information.  
Distribuido bajo la licencia MIT. Consulta el archivo `LICENSE` para más información.

---

## 👨‍💻 Author / Autor

[@visintinitech](https://github.com/visintinitech) – [tu-email@example.com](mailto:visintinitech@gmail.com)

*Made with ❤️ as a junior developer project.*  
*Hecho con ❤️ como proyecto de desarrollador junior.*

---

⭐ **If you like this project, give it a star!**  
⭐ **Si te gusta este proyecto, ¡dale una estrella!**
```

