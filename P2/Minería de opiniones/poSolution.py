import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

# Paso 1: carga los datos y selecciona 10000 comentarios aleatorios
data = pd.read_csv("amazon_baby.csv")
data = data.sample(n=10000, random_state=42)

# Paso 2: divide los comentarios en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data["review"], data["rating"], test_size=0.2, stratify=data["rating"], random_state=42)

# Paso 3: utiliza tf-idf para vectorizar los comentarios y construir un clasificador de regresión logística
tfidf = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
clf_tfidf = LogisticRegression()
clf_tfidf.fit(X_train_tfidf, y_train)
acc_tfidf = clf_tfidf.score(X_test_tfidf, y_test)

# Paso 4: utiliza DistilBERT para vectorizar los comentarios y construir otro clasificador de regresión logística
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
X_train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
X_test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train_encodings), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test_encodings), y_test))

# model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
# model.fit(train_dataset.shuffle(1000).batch(16), epochs=2, batch_size=16, validation_data=test_dataset.shuffle(1000).batch(16))
# acc_transformer = model.evaluate(test_dataset.batch(16))[1]

# Paso 5: utiliza Fine Tuning para construir otro clasificador con DistilBERT
model_ft = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
optimizer_ft = tf.keras.optimizers.Adam(learning_rate=5e-5)
model_ft.compile(optimizer=optimizer_ft, loss=model_ft.compute_loss, metrics=['accuracy'])
model_ft.fit(train_dataset.shuffle(1000).batch(16), epochs=2, batch_size=16, validation_data=test_dataset.shuffle(1000).batch(16))
acc_ft = model_ft.evaluate(test_dataset.batch(16))[1]

# Paso 6: evalúa la exactitud de los tres clasificadores y compáralos
print("Exactitud utilizando tf-idf:", acc_tfidf)
# print("Exactitud utilizando DistilBERT:", acc_transformer)
print("Exactitud utilizando Fine Tuning de DistilBERT:", acc_ft)
