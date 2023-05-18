from keras.datasets import mnist 
from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense


def load_dataset():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test

def digit_recognition_cnn():

    cnn = Sequential()
    cnn.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = (28, 28, 1)))
    cnn.add(MaxPool2D(pool_size = 2, strides = 2))
    cnn.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
    cnn.add(MaxPool2D(pool_size = 2, strides = 2))
    cnn.add(Flatten())
    cnn.add(Dense(units = 128, activation = 'relu'))
    cnn.add(Dense(units = 10, activation = 'softmax'))
    
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return cnn

cnn = digit_recognition_cnn()

X_train, y_train, X_test, y_test = load_dataset()
cnn.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 15, batch_size = 175)


loss, accuracy = cnn.evaluate(X_test, y_test)
print('Test loss:', loss, "     Test accuracy: ", accuracy)

cnn.save('digitRecognizer.h5')


# Step 8: load required keras libraries
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
 
def load_new_image(path):
    newImage = load_img(path, grayscale=True, target_size=(28, 28))
    newImage = img_to_array(newImage)
    newImage = newImage.reshape((1, 28, 28, 1)).astype('float32')
    newImage = newImage / 255
    return newImage

def test_model_performance():
    img = load_new_image('sample_images/digit2.png')
    cnn = load_model('digitRecognizer.h5')
    imageClass = cnn.predict(img)
    print(imageClass[0])
    print("Predicted label:", imageClass.argmax())
 
test_model_performance()