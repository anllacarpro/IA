import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers.integrations import TensorBoardCallback
import datetime
import os

# Cargar los datos desde el archivo CSV
ruta_archivo_csv = "data/libro.csv"
df = pd.read_csv(ruta_archivo_csv)

# Asegúrate de que hay una columna 'text' para el texto y 'label' para las etiquetas
assert 'text' in df.columns and 'label' in df.columns, "El CSV debe tener columnas 'text' y 'label'"

# Dividir los datos en conjuntos de entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.1)

# Cargar el tokenizer y el modelo preentrenado de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['label'].unique()))

# Tokenizar los datos
def tokenize_data(data):
    return tokenizer(data['text'].tolist(), padding=True, truncation=True, return_tensors='pt')

train_encodings = tokenize_data(train_df)
val_encodings = tokenize_data(val_df)

# Crear datasets de PyTorch
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_df['label'].tolist())
val_dataset = TextDataset(val_encodings, val_df['label'].tolist())

# Configurar la carpeta de logs con marca de tiempo para identificar diferentes ejecuciones
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", "fit", current_time)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=log_dir,  # Directorio de logs para TensorBoard
    logging_steps=10,
    evaluation_strategy="epoch",
    report_to="tensorboard"  # Asegúrate de que los logs se envíen a TensorBoard
)

# Inicializar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[TensorBoardCallback()]
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
eval_results = trainer.evaluate()
print(f"Resultados de la evaluación: {eval_results}")

# Guardar el modelo entrenado
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Función para predecir la clase de un texto
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class

# Ejemplo de uso
text = "¿Cómo se resuelve este problema?"
predicted_class = predict(text)
print(f"Clase predicha: {predicted_class}")