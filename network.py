import os
from PIL import Image
import tensorflow as tf
import numpy as np
# from tensorflow import keras #tensorflow2
import keras #tensorflow1
from keras.optimizers import RMSprop, SGD
#
k=10
from keras.models import Model # resnet
from keras.regularizers import l2 #resnet
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation,\
    Reshape, ZeroPadding2D, BatchNormalization, AveragePooling2D
from keras.optimizers import RMSprop, Adam
from keras.datasets import cifar10, cifar100

class FCNetwork():

    def __init__(self):
        self.number_of_train_data = 50000
        self.number_of_test_data = 10000
        self.resize_width = 28
        self.resize_height = 28

        # Cifar10 vgg16
        self.input_shape = (32, 32, 3)
        self.num_classes = 10

    def load_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()
        train_labels = train_labels[:self.number_of_train_data]
        test_labels = test_labels[:self.number_of_test_data]
        train_datas = train_datas[:self.number_of_train_data].reshape(-1, self.resize_width, self.resize_height, 1) / 255.0
        test_datas = test_datas[:self.number_of_test_data].reshape(-1, self.resize_width, self.resize_height, 1) / 255.0
        
        # Standardize training samples  
        mean_px = train_datas.mean().astype(np.float32)
        std_px = train_datas.std().astype(np.float32)
        train_datas = (train_datas - mean_px) / std_px

        # Standardize test samples  
        mean_px = test_datas.mean().astype(np.float32)
        std_px = test_datas.std().astype(np.float32)
        test_datas = (test_datas - mean_px) / std_px

        # tensorflow1
        # One-hot encoding the labels
        train_labels = keras.utils.np_utils.to_categorical(train_labels)
        test_labels = keras.utils.np_utils.to_categorical(test_labels)


        return (train_datas, train_labels), (test_datas, test_labels)

    def load_cifar10(self):
        (train_datas, train_labels), (test_datas, test_labels) = cifar10.load_data()

        # 将数据转换为浮点型并归一化到 [0,1] 范围
        train_datas = train_datas.astype('float32') / 255
        test_datas = test_datas.astype('float32') / 255

        # 将标签转换为 one-hot 编码
        train_labels = keras.utils.to_categorical(train_labels, self.num_classes)
        test_labels = keras.utils.to_categorical(test_labels, self.num_classes)

        return (train_datas, train_labels), (test_datas, test_labels)


    def delete_data(self, model, data_path):
        i = 0
        for filename1 in os.listdir(data_path):
            self.class_num += 1
            filename2 = data_path + "/" + filename1
            file = os.listdir(filename2)
            self.test_pic_num.append(len(file))
            self.test_pic_label.append(filename1)
            single_dele_num = 0
            os.chdir(filename2)
            for filename_img in os.listdir(filename2):
                img = Image.open(filename_img).convert('L')
                img = np.reshape(img, (28, 28, 1)) / 255.
                x = np.array([img])
                y = model.predict(x)
                if (np.argmax(y) != int(filename1)):
                    single_dele_num += 1
                    os.remove(filename_img)
            self.dele_pic[i] = single_dele_num
            i += 1


    def load_model(self, file_name=None):
        # file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)


    def train_model(self, model, train_datas, train_labels, name_of_file=None, epochs=50, batch_size=128, with_checkpoint=False):
        #base batchsize=none
        if with_checkpoint:
            prefix = ''
            filepath = prefix + name_of_file + '-{epoch:02d}-{loss:.4f}.h5'
            checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=5, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=1)
        else:
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=None, verbose=1)
        return model
    
    def evaluate_model(self, model, test_datas, test_labels, mode='normal'):
        loss, acc = model.evaluate(test_datas, test_labels, batch_size=128)
        if mode == 'normal':
            print('Normal model accurancy: {:5.2f}%'.format(100*acc))
            print('')
        else:
            print(mode, 'mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')
        return acc

    def evaluate_mutscore(self,model,data_path):
        error_rate = []
        for filename1 in os.listdir(data_path):
            filename2 = data_path + "/" + filename1
            # print(filename2)
            file = os.listdir(filename2)
            num_img = len(file)
            true1 = 0
            os.chdir(filename2)
            for filename_img in file:
                img = Image.open(filename_img).convert('L')
                img = np.reshape(img, (28, 28, 1)) / 255.
                x = np.array([img])
                y = model.predict(x)
                if (np.argmax(y) == int(filename1)):
                    true1 += 1
            error_rate.append((num_img - true1) / num_img)
        print("错误率：",error_rate)

        return  error_rate

    def save_model(self, model, name_of_file, mode='normal'):
        prefix = ''
        file_name = prefix + name_of_file + '.h5'
        model.save(file_name)
        if mode == 'normal':
            print('Normal model is successfully trained and saved at', file_name)
        else:
            print('Mutated model by ' + mode + ' is successfully saved at', file_name)
        print('')

    def Lenet1(self):
        nb_classes = 10
        # convolution kernel size
        kernel_size = (5, 5)
        input_tensor = Input(shape=[28, 28, 1])
        # block1
        x = Conv2D(4, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
        # block2
        x = Conv2D(12, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(nb_classes, name='before_softmax')(x)
        x = Activation('softmax', name='predictions')(x)

        model = keras.Model(input_tensor, x)
        return model

    def Lenet5(self, input_tensor=None, train=False):
        nb_classes = 10
        # convolution kernel size
        kernel_size = (5, 5)

        input_tensor = Input(shape=[28, 28, 1])

        # block1
        x = Conv2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

        # block2
        x = Conv2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(120, activation='relu', name='fc1')(x)
        x = Dense(84, activation='relu', name='fc2')(x)
        x = Dense(nb_classes, name='before_softmax')(x)
        x = Activation('softmax', name='predictions')(x)

        model = keras.Model(input_tensor, x)

        return model

    def Cifarmodel(self):
        input_tensor = Input(shape=[32, 32, 3])
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        x = Flatten(name='flatten')(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dense(10, name='before_softmax')(x)
        x = Activation('softmax', name='predictions')(x)
        model = keras.Model(input_tensor, x, name='cifarmodel')
        return model



    def compile_model(self, model):#lenet1,lenet5
        # lenet
        # model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

        # cifarmodel
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])
        return model


