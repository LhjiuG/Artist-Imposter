from utils import *
import tensorflow as tf


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
