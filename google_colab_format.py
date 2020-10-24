import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import IPython.display as display


def tensor_to_image(tensor):
    """Convert a tensor to an image"""
    # Revert the normalization of the pixel did in the NN
    tensor = tensor * 255
    # Can't use a tensor for PIL modules
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_img(image_path):
    """Load an image and resize it"""
    max_dim = 512
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(img, title=None):
    """Show an image"""
    if len(img.shape) > 3:
        img = tf.squeeze(img, axis=0)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def clip_0_1(image):
    """
    :param image:
    :return: Tensor of image with value between 0.0 and 1.0
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    vgg_model = tf.keras.Model([vgg.input], outputs)
    return vgg_model


class StyleContentModel(tf.keras.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float inputs in range[0, 1]"""
        inputs = inputs * 255.0
        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


class StyleContentLoss(tf.keras.losses.Loss):

    def __init__(self, style_layers, content_layers, style_image, content_image):
        super(StyleContentLoss, self).__init__()
        self.style_weight = 1e-2
        self.content_weight = 1e4
        extractor = StyleContentModel(style_layers, content_layers)
        self.style_targets = extractor(style_image)['style']
        self.content_targets = extractor(content_image)['content']
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

    def call(self, y_true, y_pred):
        style_outputs = y_true  #
        content_outputs = y_pred
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss


class CONFIG:
    NOISE_RATIO = 0.6
    EPOCHS = 10
    STEPS_PER_EPOCH = 100
    STYLE_IMAGE = '/content/983794168.jpg'  # Style image to use.
    CONTENT_IMAGE = '/content/background-image.png'  # Content image to use.
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
