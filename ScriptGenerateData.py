import csv
import random
import fitz  # PyMuPDF
import spacy
import pandas as pd
import re

# Cargar el modelo de lenguaje de spaCy para detectar entidades y conceptos clave
nlp = spacy.load("en_core_web_sm")

# Definir temas y tipos de preguntas
temas = ["Aritmética", "Álgebra", "Geometría", "Estadística", "Cálculo"]

# Tipos de preguntas y respuestas asociadas
tipos_preguntas_respuestas = [
    ("¿Cómo se resuelve este problema?", "Para resolver este problema, sigue los pasos específicos del concepto de {}."),
    ("¿Cuál es la respuesta a esta operación?", "La respuesta depende de la operación exacta. ¿Puedes proporcionar más detalles sobre {}?"),
    ("¿Puedes explicar el concepto de {}?", "{} es un concepto que involucra ciertos principios básicos. Te recomiendo revisar los fundamentos."),
    ("¿Cómo encuentro el valor de {}?", "Para encontrar el valor de {}, usa la fórmula y los pasos apropiados para ese tipo de problema."),
    ("¿Cuál es la fórmula para {}?", "La fórmula de {} es generalmente ... [incluye la fórmula común aquí]."),
    ("¿Qué significa {} en matemáticas?", "{} se refiere a un concepto importante que abarca..."),
    ("¿Puedes darme ejemplos de {}?", "Claro, un ejemplo común de {} es..."),
    ("¿Qué pasos necesito seguir para resolver {}?", "Los pasos para resolver {} incluyen... [explicar los pasos básicos]."),
    ("¿Cuándo se usa {}?", "Se usa {} en situaciones donde..."),
    ("¿Cómo aplicar {} en un problema?", "Para aplicar {}, identifica el tipo de problema y sigue los pasos básicos del concepto.")
]

# Definir conceptos comunes para cada tema
conceptos = {
    "Aritmética": ["suma", "resta", "multiplicación", "división", "fracciones", "porcentajes"],
    "Álgebra": ["ecuación lineal", "factorización", "variable", "expresión algebraica", "sistema de ecuaciones"],
    "Geometría": ["área", "perímetro", "triángulo", "ángulo", "volumen", "círculo"],
    "Estadística": ["media", "mediana", "moda", "desviación estándar", "probabilidad"],
    "Cálculo": ["derivada", "integral", "límite", "función continua", "tasa de cambio"]
}

# Función para limpiar el texto extraído
def clean_text(text):
    # Eliminar saltos de línea y espacios extra
    text = re.sub(r'\s+', ' ', text)
    # Eliminar caracteres no deseados
    text = re.sub(r'[^\w\s.,;:!?()\-]', '', text)
    return text

# Función para extraer texto de un PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")
    return clean_text(text)

# Función para generar preguntas y respuestas basadas en conceptos clave
def generate_qa_pairs(text, num_pairs_per_concept=10):
    doc = nlp(text)
    qa_pairs = []
    for ent in doc.ents:
        concept = ent.text
        tema = ent.label_  # Usa el tipo de entidad como tema (puedes refinarlo más)
        
        # Generar múltiples preguntas y respuestas para cada concepto
        for _ in range(num_pairs_per_concept):
            pregunta_tipo, respuesta_tipo = random.choice(tipos_preguntas_respuestas)
            pregunta = pregunta_tipo.format(concept)
            respuesta = respuesta_tipo.format(concept)
            qa_pairs.append((pregunta, tema, respuesta))
    return qa_pairs

# Función para generar ejercicios y soluciones
def generate_exercises_and_solutions(num_exercises=100):
    exercises = []
    for _ in range(num_exercises):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        operation = random.choice(["+", "-", "*", "/"])
        if operation == "+":
            question = f"¿Cuál es el resultado de {a} + {b}?"
            answer = str(a + b)
        elif operation == "-":
            question = f"¿Cuál es el resultado de {a} - {b}?"
            answer = str(a - b)
        elif operation == "*":
            question = f"¿Cuál es el resultado de {a} * {b}?"
            answer = str(a * b)
        elif operation == "/":
            question = f"¿Cuál es el resultado de {a} / {b}?"
            answer = str(a / b)
        exercises.append((question, "Ejercicio", answer))
    return exercises

# Función para generar evaluaciones de respuestas
def generate_evaluations(num_evaluations=100):
    evaluations = []
    for _ in range(num_evaluations):
        question = "¿Es correcta la respuesta a esta pregunta?"
        correct_answer = random.choice(["Sí", "No"])
        user_answer = random.choice(["Sí", "No"])
        evaluation = "Correcto" if correct_answer == user_answer else "Incorrecto"
        evaluations.append((question, "Evaluación", evaluation))
    return evaluations

# Lista de rutas de archivos PDF
pdf_paths = [
    "pdf/5to-primaria-1.pdf",
    "pdf/5to-primaria-2.pdf",
    "pdf/6to-primaria-1.pdf",
    "pdf/6to-primaria-2.pdf",
    "pdf/6to-primaria-3.pdf",
    "pdf/6to-primaria-4.pdf"
]

# Ruta del archivo CSV de salida
ruta_archivo_csv = r"C:/Users/estef/OneDrive/Documentos/Ciclo 8/Inteligencia Artificial/CHATBOT IA/preguntas_respuestas_libros.csv"

# Extraer texto de cada PDF y generar preguntas y respuestas
all_qa_pairs = []
for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    qa_pairs = generate_qa_pairs(text, num_pairs_per_concept=50)  # Aumentar el número de pares por concepto
    all_qa_pairs.extend(qa_pairs)

# Generar ejercicios y soluciones
exercises_and_solutions = generate_exercises_and_solutions(num_exercises=10000)
all_qa_pairs.extend(exercises_and_solutions)

# Generar evaluaciones de respuestas
evaluations = generate_evaluations(num_evaluations=10000)
all_qa_pairs.extend(evaluations)

# Asegurarse de tener al menos 300,000 datos
while len(all_qa_pairs) < 300000:
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        qa_pairs = generate_qa_pairs(text, num_pairs_per_concept=50)
        all_qa_pairs.extend(qa_pairs)
        if len(all_qa_pairs) >= 300000:
            break

# Guardar todas las preguntas y respuestas en un archivo CSV
df = pd.DataFrame(all_qa_pairs, columns=["text", "label", "response"])
df.to_csv(ruta_archivo_csv, index=False, encoding="utf-8")

print(f"Archivo '{ruta_archivo_csv}' generado con éxito con más de 300,000 datos.")