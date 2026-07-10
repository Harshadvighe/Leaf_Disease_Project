from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

img_size=(224,224)
gen=ImageDataGenerator(rescale=1./255,validation_split=0.2)

train=gen.flow_from_directory("dataset",target_size=img_size,batch_size=32,
class_mode="categorical",subset="training")
val=gen.flow_from_directory("dataset",target_size=img_size,batch_size=32,
class_mode="categorical",subset="validation")

base=MobileNetV2(weights="imagenet",include_top=False,input_shape=(224,224,3))
base.trainable=False

x=GlobalAveragePooling2D()(base.output)
out=Dense(train.num_classes,activation="softmax")(x)
model=Model(base.input,out)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(train,validation_data=val,epochs=5)
model.save("model.keras")
print(train.class_indices)
