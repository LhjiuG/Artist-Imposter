import time
import IPython.display as display

from model import *
from utils import *


class CONFIG:
    NOISE_RATIO = 0.6
    EPOCHS = 10
    STEPS_PER_EPOCH = 100
    STYLE_IMAGE = 'images/pixel_style.png'  # Style image to use.
    CONTENT_IMAGE = 'images/content1.jpg'  # Content image to use.
    OUTPUT_DIR = 'output/'
    CONTENT_LAYER = ['block5_conv2']
    STYLE_LAYERS = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    TOTAL_VARIATION_WEIGHT = 30


style_image = load_img(CONFIG.STYLE_IMAGE)
content_image = load_img(CONFIG.CONTENT_IMAGE)
image = tf.Variable(content_image)

model = StyleContentModel(CONFIG.STYLE_LAYERS, CONFIG.CONTENT_LAYER)
loss = StyleContentLoss(CONFIG.STYLE_LAYERS, CONFIG.CONTENT_LAYER, style_image, content_image)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = model(image)
        loss_ = loss(y_true=outputs['style'], y_pred=outputs['content'])
        loss_ += CONFIG.TOTAL_VARIATION_WEIGHT * tf.image.total_variation(image)

    grad = tape.gradient(loss_, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


start = time.time()

step = 0
for n in range(CONFIG.EPOCHS):
    for m in range(CONFIG.STEPS_PER_EPOCH):
        step += 1
        train_step(image)
        print(".", end='')
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end - start))
