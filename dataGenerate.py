from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img = load_img('data/train/cats/cat.4.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

test_generator = datagen.flow_from_directory(
    "/home/ubtaiki/Picture/data",
    target_size=(150, 150),
    batch_size=2,
    class_mode=None,
    shuffle=False,
    save_to_dir="preview")

i = 0
for batch in datagen.flow(x, batch_size=1,save_to_dir='preview',save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 40:
        break



""""
train_generator = test_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    save_to_dir = 'preview')

history = model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=10,
    validation_data=validation_generator,
    nb_val_samples=800)
"""