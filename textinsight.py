"""
TextInsight 2.0 - Clasificador NLP Inteligente con AutoML y Aprendizaje Activo
Autor: Dev Junior - Versión 2.0
"""

import sqlite3
import pickle
import os
import re
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class TextInsight:
    """
    Clasificador de texto con:
    - Preprocesamiento inteligente
    - Selección automática del mejor algoritmo (Naive Bayes vs Regresión Logística)
    - Explicabilidad de predicciones
    - Procesamiento por lotes
    - Aprendizaje activo (corrección en caliente)
    """

    def __init__(self, db_path='textinsight.db', model_dir='models'):
        self.db_path = db_path
        self.model_dir = model_dir
        self.vectorizer = None
        self.classifier = None
        self.labels = None
        self._init_db()
        os.makedirs(model_dir, exist_ok=True)

    # ======================== 1. PREPROCESAMIENTO Y BD ========================
    def _init_db(self):
        """Crea las tablas necesarias si no existen."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS entrenamiento (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT NOT NULL,
                etiqueta TEXT NOT NULL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT NOT NULL,
                etiqueta_predicha TEXT NOT NULL,
                confianza REAL,
                fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _limpiar_texto(self, texto):
        """
        Limpia el texto eliminando ruido digital:
        - URLs, menciones, hashtags, números y puntuación.
        """
        if not isinstance(texto, str):
            return ""
        texto = texto.lower()
        texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
        texto = re.sub(r'@\w+|#\w+', '', texto)
        texto = re.sub(r'\d+', '', texto)
        texto = re.sub(r'[^\w\s]', '', texto)
        return texto.strip()

    def cargar_datos_desde_csv(self, archivo_csv, col_texto, col_etiqueta):
        """
        Carga datos desde un CSV, limpia los textos y los guarda en la BD.
        """
        try:
            df = pd.read_csv(archivo_csv)
        except Exception as e:
            print(f"❌ Error al leer el CSV: {e}")
            return

        if col_texto not in df.columns or col_etiqueta not in df.columns:
            print(f"❌ Columnas '{col_texto}' o '{col_etiqueta}' no encontradas.")
            return

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        contador = 0
        for _, row in df.iterrows():
            texto_limpio = self._limpiar_texto(str(row[col_texto]))
            if texto_limpio:  # Evitamos textos vacíos
                c.execute("INSERT INTO entrenamiento (texto, etiqueta) VALUES (?, ?)",
                          (texto_limpio, str(row[col_etiqueta])))
                contador += 1
        conn.commit()
        conn.close()
        print(f"✅ Datos cargados desde {archivo_csv} ({contador} registros válidos)")

    def obtener_datos_entrenamiento(self):
        """Recupera todos los textos y etiquetas de la BD."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT texto, etiqueta FROM entrenamiento")
        datos = c.fetchall()
        conn.close()
        if not datos:
            return [], []
        textos, etiquetas = zip(*datos)
        return list(textos), list(etiquetas)

    # ======================== 2. ENTRENAMIENTO CON AUTOML ========================
    def entrenar(self, test_size=0.2):
        """
        Entrena y compara Naive Bayes vs Regresión Logística.
        Se queda con el modelo que mejor accuracy obtenga.
        """
        textos, etiquetas = self.obtener_datos_entrenamiento()
        if len(textos) < 3:
            print("❌ Se necesitan al menos 3 ejemplos para entrenar.")
            return

        # Dividimos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            textos, etiquetas, test_size=test_size, random_state=42, stratify=etiquetas
        )

        # Vectorizamos (usamos español como idioma base)
        self.vectorizer = TfidfVectorizer(stop_words='spanish', max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Diccionario de modelos a probar
        modelos = {
            'Naive Bayes': MultinomialNB(),
            'Regresión Logística': LogisticRegression(max_iter=1000, random_state=42)
        }

        mejor_acc = -1.0
        mejor_modelo = None
        mejor_nombre = ""

        print("\n🤖 EJECUTANDO AUTOML... Comparando algoritmos:")
        print("-" * 40)
        for nombre, modelo in modelos.items():
            modelo.fit(X_train_vec, y_train)
            acc = modelo.score(X_test_vec, y_test)
            print(f"  {nombre}: Accuracy = {acc:.2%}")
            if acc > mejor_acc:
                mejor_acc = acc
                mejor_modelo = modelo
                mejor_nombre = nombre

        # Asignamos el ganador
        self.classifier = mejor_modelo
        self.labels = list(set(etiquetas))
        print("-" * 40)
        print(f"🏆 MODELO GANADOR: {mejor_nombre} con {mejor_acc:.2%}")

        # Reporte detallado del ganador
        y_pred = self.classifier.predict(X_test_vec)
        print("\n📋 REPORTE DEL MODELO GANADOR:")
        print(classification_report(y_test, y_pred))

        # Guardamos el modelo
        self.guardar_modelo()
        print("💾 Modelo guardado correctamente.")

    # ======================== 3. PERSISTENCIA ========================
    def guardar_modelo(self):
        """Guarda vectorizador, clasificador y etiquetas en archivos pickle."""
        with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.model_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(os.path.join(self.model_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(self.labels, f)

    def cargar_modelo(self):
        """Carga el modelo desde los archivos pickle."""
        try:
            with open(os.path.join(self.model_dir, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(self.model_dir, 'classifier.pkl'), 'rb') as f:
                self.classifier = pickle.load(f)
            with open(os.path.join(self.model_dir, 'labels.pkl'), 'rb') as f:
                self.labels = pickle.load(f)
            print("✅ Modelo cargado exitosamente.")
            return True
        except FileNotFoundError:
            print("❌ No se encontró el modelo. Entrena primero.")
            return False

    # ======================== 4. PREDICCIÓN + EXPLICABILIDAD ========================
    def predecir(self, texto):
        """
        Clasifica un texto, guarda la predicción en BD y retorna (etiqueta, confianza).
        """
        if not self.classifier or not self.vectorizer:
            if not self.cargar_modelo():
                return None, None

        texto_limpio = self._limpiar_texto(texto)
        if not texto_limpio:
            return "VACIO", 0.0

        vec = self.vectorizer.transform([texto_limpio])
        probas = self.classifier.predict_proba(vec)[0]
        idx = probas.argmax()
        etiqueta = self.classifier.classes_[idx]
        confianza = probas[idx]

        # Guardar en BD
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO predicciones (texto, etiqueta_predicha, confianza) VALUES (?, ?, ?)",
            (texto_limpio, etiqueta, confianza)
        )
        conn.commit()
        conn.close()

        return etiqueta, confianza

    def explicar_prediccion(self, texto):
        """
        Devuelve (etiqueta, confianza, palabras_clave) que influyeron en la decisión.
        """
        etiqueta, confianza = self.predecir(texto)
        if not etiqueta:
            return None, None, []

        texto_limpio = self._limpiar_texto(texto)
        vec = self.vectorizer.transform([texto_limpio])
        # Obtenemos los coeficientes del modelo (si es Lineal) o log-probs (si es NB)
        # Mapeamos la clase predicha a su índice
        clases = list(self.classifier.classes_)
        idx_clase = clases.index(etiqueta)

        if hasattr(self.classifier, 'coef_'):
            # Para Regresión Logística (lineal)
            coefs = self.classifier.coef_[idx_clase]
        else:
            # Para Naive Bayes
            coefs = self.classifier.feature_log_prob_[idx_clase]

        # Multiplicamos el vector TF-IDF por los coeficientes
        importancia = vec.toarray()[0] * coefs
        # Obtenemos los índices de las 3 palabras más importantes (orden descendente)
        top_indices = importancia.argsort()[-3:][::-1]
        # Filtramos solo las que tienen peso distinto de 0
        nombres = self.vectorizer.get_feature_names_out()
        palabras_clave = [nombres[i] for i in top_indices if importancia[i] != 0]

        return etiqueta, confianza, palabras_clave

    # ======================== 5. PROCESAMIENTO POR LOTES ========================
    def predecir_lote(self, archivo_entrada, archivo_salida, col_texto='texto'):
        """Clasifica un CSV entero y guarda los resultados con las nuevas columnas."""
        if not self.classifier or not self.vectorizer:
            if not self.cargar_modelo():
                return

        try:
            df = pd.read_csv(archivo_entrada)
        except Exception as e:
            print(f"❌ Error al leer {archivo_entrada}: {e}")
            return

        if col_texto not in df.columns:
            print(f"❌ Columna '{col_texto}' no encontrada.")
            return

        predicciones = []
        confianzas = []
        for texto in df[col_texto]:
            etiq, conf = self.predecir(str(texto))
            predicciones.append(etiq if etiq else 'Error')
            confianzas.append(conf if conf else 0.0)

        df['etiqueta_predicha'] = predicciones
        df['confianza'] = confianzas
        df.to_csv(archivo_salida, index=False)
        print(f"✅ Lote procesado. {len(df)} registros guardados en {archivo_salida}")

    # ======================== 6. APRENDIZAJE ACTIVO ========================
    def corregir_y_guardar(self, texto_original, etiqueta_correcta):
        """
        Guarda la corrección en la tabla de entrenamiento para futuros reentrenos.
        El texto se limpia automáticamente.
        """
        texto_limpio = self._limpiar_texto(texto_original)
        if not texto_limpio:
            print("❌ Texto vacío después de limpiar, no se guardó.")
            return

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO entrenamiento (texto, etiqueta) VALUES (?, ?)",
                  (texto_limpio, etiqueta_correcta))
        conn.commit()
        conn.close()
        print("✅ Corrección guardada en la base de datos. ¡Reentrena para mejorar el modelo!")

    # ======================== 7. ESTADÍSTICAS AVANZADAS ========================
    def mostrar_estadisticas(self):
        """Muestra distribución de clases y confianza promedio por etiqueta."""
        conn = sqlite3.connect(self.db_path)

        df_ent = pd.read_sql_query("SELECT etiqueta FROM entrenamiento", conn)
        df_pred = pd.read_sql_query("SELECT etiqueta_predicha, confianza FROM predicciones", conn)
        conn.close()

        print("\n" + "="*50)
        print("📊 DASHBOARD DE SALUD DEL MODELO")
        print("="*50)

        print(f"\n📌 Total registros entrenamiento: {len(df_ent)}")
        if not df_ent.empty:
            print("\n📈 DISTRIBUCIÓN DE CLASES (Entrenamiento):")
            print(df_ent['etiqueta'].value_counts())

        print(f"\n📌 Total predicciones realizadas: {len(df_pred)}")
        if not df_pred.empty:
            print("\n📈 CONFIANZA PROMEDIO POR ETIQUETA (Predicciones):")
            print(df_pred.groupby('etiqueta_predicha')['confianza'].mean().round(4))

        # Alertas de desbalanceo
        if not df_ent.empty:
            distrib = df_ent['etiqueta'].value_counts()
            if distrib.min() / distrib.max() < 0.3:
                print("\n⚠️  ALERTA: Las clases están muy desbalanceadas. "
                      "Considera agregar más ejemplos de las clases minoritarias.")

    def exportar_predicciones(self, archivo_csv='predicciones_export.csv'):
        """Exporta todas las predicciones a CSV."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM predicciones", conn)
        conn.close()
        df.to_csv(archivo_csv, index=False)
        print(f"✅ Predicciones exportadas a {archivo_csv}")

    # ======================== 8. MENÚ INTERACTIVO ========================
    def menu(self):
        """Menú principal con todas las opciones."""
        while True:
            print("\n" + "="*55)
            print("   🧠 TextInsight 2.0 - Clasificador NLP Inteligente")
            print("="*55)
            print("1.  Entrenar modelo (AutoML: NB vs RegLog)")
            print("2.  Clasificar un texto + Corrección (Aprendizaje Activo)")
            print("3.  Cargar modelo guardado")
            print("4.  Ver estadísticas avanzadas (Dashboard)")
            print("5.  Exportar predicciones a CSV")
            print("6.  Cargar datos desde CSV (entrenamiento)")
            print("7.  Explicar predicción (palabras clave)")
            print("8.  Clasificar lote desde CSV")
            print("9.  Salir")
            opcion = input("\nElige una opción: ")

            if opcion == '1':
                self.entrenar()

            elif opcion == '2':
                texto = input("Introduce el texto a clasificar: ")
                etiqueta, conf = self.predecir(texto)
                if etiqueta:
                    print(f"🏷️  Categoría: {etiqueta} (confianza: {conf:.2%})")
                    # Aprendizaje activo: preguntamos si es correcto
                    corregir = input("¿Es correcta la clasificación? (s/n): ")
                    if corregir.lower() == 'n':
                        etiqueta_correcta = input("Introduce la etiqueta correcta: ")
                        self.corregir_y_guardar(texto, etiqueta_correcta)
                else:
                    print("❌ No se pudo realizar la predicción.")

            elif opcion == '3':
                self.cargar_modelo()

            elif opcion == '4':
                self.mostrar_estadisticas()

            elif opcion == '5':
                self.exportar_predicciones()

            elif opcion == '6':
                archivo = input("Ruta del archivo CSV: ")
                col_texto = input("Nombre de la columna de texto: ")
                col_etiqueta = input("Nombre de la columna de etiqueta: ")
                self.cargar_datos_desde_csv(archivo, col_texto, col_etiqueta)

            elif opcion == '7':
                texto = input("Introduce el texto a analizar: ")
                etiqueta, conf, palabras = self.explicar_prediccion(texto)
                if etiqueta:
                    print(f"🏷️  Categoría: {etiqueta} (confianza: {conf:.2%})")
                    print(f"🔑 Palabras clave: {', '.join(palabras) if palabras else 'Ninguna destacada'}")
                else:
                    print("❌ No se pudo realizar la explicación.")

            elif opcion == '8':
                archivo_entrada = input("Ruta del CSV con textos a clasificar: ")
                archivo_salida = input("Ruta del CSV de salida (resultados): ")
                col_texto = input("Nombre de la columna que contiene el texto: ")
                self.predecir_lote(archivo_entrada, archivo_salida, col_texto)

            elif opcion == '9':
                print("👋 ¡Hasta luego! Sigue entrenando a TextInsight.")
                break

            else:
                print("❌ Opción no válida. Intenta de nuevo.")


# ======================== PUNTO DE ENTRADA ========================
if __name__ == "__main__":
    app = TextInsight()
    app.menu()
