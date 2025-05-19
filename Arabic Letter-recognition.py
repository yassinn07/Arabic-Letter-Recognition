#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import rotate
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, f1_score ,classification_report
from keras.layers import Dense, Activation,Flatten
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix


# In[2]:


train_images = pd.read_csv('Arabic-Characters-Recognition\\csvTrainImages 13440x1024.csv', delimiter=',', header=None)
train_labels = pd.read_csv('Arabic-Characters-Recognition\\csvTrainLabel 13440x1.csv', delimiter=',', header=None)
test_images = pd.read_csv('Arabic-Characters-Recognition\\csvTestImages 3360x1024.csv', delimiter=',', header=None)
test_labels = pd.read_csv('Arabic-Characters-Recognition\\csvTestLabel 3360x1.csv', delimiter=',', header=None)


# In[3]:


train_images_reshaped = np.array(train_images).reshape((train_images.shape[0], 32, 32))
test_images_reshaped = np.array(test_images).reshape((test_images.shape[0], 32, 32))


# In[4]:


unique_classes = train_labels[0].unique()
num_classes = len(unique_classes)
class_distribution = train_labels[0].value_counts()

print(f"Number of unique classes: {num_classes}")
print("Distribution of samples in each class:")
print(class_distribution)
train_labels.columns = ['label']


# In[5]:


print("Train Images Data Types:", train_images.dtypes)
print("Test Images Data Types:", test_images.dtypes)
print("NaN values in test_images:")
print(test_images.isna().sum())

print("Train Images Columns:", train_images.columns)
print("Test Images Columns:", test_images.columns)
test_images.columns = train_images.columns


# In[6]:


train_images_normalized = train_images_reshaped/255.0
test_images_normalized = test_images_reshaped/255.0


# In[7]:


X_train, X_val, y_train, y_val = train_test_split(train_images/255.0, train_labels['label'], test_size=0.2, random_state=40)


# In[8]:


def reconstruct_images(image_values, display=False):
    image_array = np.asarray(image_values)
    image_array = image_array.reshape(32, 32).astype('uint8')
    image_array = np.flip(image_array, 0)
    image_array = rotate(image_array, -90)
    if display:
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()
    return image_array

for i in range(6):
    reconstruct_images(train_images.loc[i*8],True)


# In[9]:


svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_val)

f1_val = f1_score(y_val, y_pred, average='weighted')

print(f'Average F1 Score (Validation): {f1_val}')


# In[10]:


test_predictions = svm_model.predict(test_images/255.0)


conf_matrix_test = confusion_matrix(test_labels, test_predictions)
average_f1_test_svm = f1_score(test_labels, test_predictions, average='weighted')

print(f'Confusion Matrix (Testing):\n{conf_matrix_test}')
print(f'Average F1 Score (Testing): {average_f1_test_svm}')


# In[11]:


def EvaluateKNN(X_train, y_train, X_val, y_val, k):
    KNNModel = KNeighborsClassifier(n_neighbors=k)
    KNNModel.fit(X_train, y_train)
    y_pred = KNNModel.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='weighted')
    return f1


# In[12]:


k_values = [1,3,5,7,9]
f1_scores = []

for k in k_values:
    f1 = EvaluateKNN(X_train, y_train, X_val, y_val, k)
    f1_scores.append(f1)
    print(f'K={k}, F1 Score: {f1}')

plt.plot(k_values, f1_scores, marker='o')
plt.title('F1 Scores with Different K Values')
plt.xlabel('K Values')
plt.ylabel('F1 Score')
plt.show()


# In[13]:


BestK = k_values[np.argmax(f1_scores)]
print(f'Best K value: {BestK}')

BestModel = KNeighborsClassifier(n_neighbors=BestK)
BestModel.fit(train_images, train_labels['label'])

test_predictions = BestModel.predict(test_images)

conf_matrix = confusion_matrix(test_labels, test_predictions)
average_f1_test_KNN = f1_score(test_labels, test_predictions, average='weighted')

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Average F1 Score: {average_f1_test_KNN}')


# In[14]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32,32)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')
])


# In[15]:


model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


# In[16]:


history1=model.fit(np.array(X_train).reshape((X_train.shape[0], 32, 32)), y_train, epochs=10,validation_split=0.2)


# In[17]:


test_loss, test_acc = model.evaluate(np.array(X_val).reshape((X_val.shape[0], 32, 32)), y_val, verbose=2)

print('\nTest accuracy:', test_acc)


# In[18]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()


# In[19]:


model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')
])


# In[20]:


model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


# In[21]:


history2=model2.fit(np.array(X_train).reshape((X_train.shape[0], 32, 32)), y_train, epochs=10,validation_split=0.2)


# In[22]:


test_loss2, test_acc2 = model2.evaluate(np.array(X_val).reshape((X_val.shape[0], 32, 32)), y_val, verbose=2)
print('\nTest accuracy:', test_acc2)


# In[23]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()


# In[24]:


if test_acc>test_acc2:
    predictions = model.predict(test_images_normalized)
    predicted_labels = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    avg_f1_NN = f1_score(test_labels, predicted_labels, average='weighted', zero_division=1)
    print('\nConfusion Matrix for Best model :')
    print(conf_matrix)
    print('\n AVG F1 score:' , avg_f1_NN)
else:
    predictions = model2.predict(test_images_normalized)
    predicted_labels = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    avg_f1_NN = f1_score(test_labels, predicted_labels, average='weighted', zero_division=1)
    print('\nConfusion Matrix for Best model :')
    print(conf_matrix)
    print('\n AVG F1 score:' , avg_f1_NN)
    
    


# In[25]:


if average_f1_test_svm > average_f1_test_KNN and average_f1_test_svm > avg_f1_NN:
    print("The best model to use in this problem is SVM")
elif average_f1_test_KNN > average_f1_test_svm and average_f1_test_KNN > avg_f1_NN:
    print("The best model to use in this problem is KNN")
elif avg_f1_NN > average_f1_test_KNN and avg_f1_NN > average_f1_test_svm:
    print("The best model to use in this problem is NN")
else: 
    print("They are all the same")

