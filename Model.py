 

from google .colab import drive 
drive .mount ('/content/drive')



import os 
import tensorflow as tf 
from tensorflow .keras .applications import MobileNetV2 
from tensorflow .keras .layers import Dense ,GlobalAveragePooling2D 
from tensorflow .keras .models import Model 
from tensorflow .keras .preprocessing .image import ImageDataGenerator 
import shutil 



DRIVE_TRAIN_DIR ='/content/drive/MyDrive/bt_split/train'
DRIVE_TEST_DIR ='/content/drive/MyDrive/bt_split/test'




LOCAL_TRAIN_DIR ='/content/bt_split/train'
LOCAL_TEST_DIR ='/content/bt_split/test'

print ("[INFO] Sao chép dữ liệu TRAIN từ Drive vào Local Colab...")
shutil .copytree (DRIVE_TRAIN_DIR ,LOCAL_TRAIN_DIR ,dirs_exist_ok =True )
print (f"[INFO] Đã sao chép {len (os .listdir (LOCAL_TRAIN_DIR ))} thư mục lớp từ Drive vào {LOCAL_TRAIN_DIR }")

print ("[INFO] Sao chép dữ liệu TEST từ Drive vào Local Colab...")
shutil .copytree (DRIVE_TEST_DIR ,LOCAL_TEST_DIR ,dirs_exist_ok =True )
print (f"[INFO] Đã sao chép {len (os .listdir (LOCAL_TEST_DIR ))} thư mục lớp từ Drive vào {LOCAL_TEST_DIR }")


TRAIN_DIR =LOCAL_TRAIN_DIR 
TEST_DIR =LOCAL_TEST_DIR 

train_datagen =ImageDataGenerator (
rescale =1. /255 ,
rotation_range =20 ,
width_shift_range =0.1 ,
height_shift_range =0.1 ,
horizontal_flip =True 
)


test_datagen =ImageDataGenerator (rescale =1. /255 )


train_generator =train_datagen .flow_from_directory (
TRAIN_DIR ,
target_size =(224 ,224 ),
batch_size =16 ,
class_mode ='categorical'
)

test_generator =test_datagen .flow_from_directory (
TEST_DIR ,
target_size =(224 ,224 ),
batch_size =16 ,
class_mode ='categorical'
)


NUM_CLASSES =train_generator .num_classes 

base_model =MobileNetV2 (weights ='imagenet',include_top =False ,input_shape =(224 ,224 ,3 ))


base_model .trainable =False 


x =base_model .output 
x =GlobalAveragePooling2D ()(x )
x =Dense (128 ,activation ='relu')(x )


predictions =Dense (NUM_CLASSES ,activation ='softmax')(x )


model =Model (inputs =base_model .input ,outputs =predictions )



model .compile (
optimizer ='adam',
loss ='categorical_crossentropy',
metrics =['accuracy']
)


print ("\n[INFO] Bắt đầu quá trình huấn luyện...")
H =model .fit (
train_generator ,
validation_data =test_generator ,
epochs =5 
)



MODEL_NAME ='identity_recognition_model.h5'
model .save (MODEL_NAME )

