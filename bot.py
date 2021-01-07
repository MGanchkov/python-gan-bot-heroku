import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
import os.path
import requests
from PIL import Image

import numpy as np
import time
import IPython.display
import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow as tf  # %tensorflow_version 1.x
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models


PORT = int(os.environ.get('PORT', 5000))

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Вспомогтальный класс
class h:
  isColabRun = False

  @staticmethod
  def dir_create_all(path: str):
    if not os.path.exists(path):
      os.makedirs(path)

  @staticmethod
  def isFile(path: str):
    b = os.path.isfile(path)
    if b:
        b = os.path.exists(path)
    return b

  @staticmethod
  def isDir(path: str):
    b = os.path.isdir(path)
    if b:
        b = os.path.exists(path)
    return b

  @staticmethod
  def files(path: str):
    s = os.listdir(path)
    fs = filter(lambda x: os.path.isfile(x), s)
    return fs

  @staticmethod
  def dirs(path: str):
    s = os.listdir(path)
    ds = filter(lambda x: os.path.isdir(x), s)
    return ds


if not tf.executing_eagerly():
    tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

h.dir_create_all('data')
# Загрузка и сохранение базового значения контета и стиля.
image_content_default = None
if not h.isFile('data/content.jpg'):
    # открываем файл для записи, в режиме wb
    f = open('data/content.jpg', "wb")
    # делаем запрос
    ufr = requests.get('https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg')
    # записываем содержимое в файл; как видите - content запроса
    f.write(ufr.content)
    f.close()
image_content_default = Image.open('data/content.jpg')
image_style_default = None
if not h.isFile('data/style.jpg'):
    # открываем файл для записи, в режиме wb
    f = open('data/style.jpg', "wb")
    # делаем запрос
    ufr = requests.get('https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg')
    # записываем содержимое в файл; как видите - content запроса
    f.write(ufr.content)
    f.close()
image_style_default = Image.open('data/style.jpg')

# загрузка и сохранения базовой пред обученной модели
h.dir_create_all('data/model')
model_default = None
if not h.isFile('data/model/VGG19.h5'):
    model_default = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    tf.keras.models.save_model(model_default, 'data/model/VGG19.h5')


# Клиент

class Client:
    def __init__(self, teleBot, chat_id):
        print(f'Client[{chat_id}].Initialize')
        self.TeleBot = teleBot
        self.ChatID = chat_id
        self.Content = image_content_default
        self.Style = image_style_default

        self.One = None
        self.Two = None
        self.waitCountImage = 2
        self.countImage = 0

        # Слой содержимого, на котором будут вытягиваться наши карты функций
        self.content_layers = ['block5_conv2']
        # Слой стилей, который нас интересует
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1'
                             ]
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)

        self.content_weight = 1e3
        self.style_weight = 1e-2

        # Output
        self.myGAN_Images = []
        self.myGAN_Best = None
        self.myGAN_file_content = None
        self.myGAN_file_style = None
        self.content_image = None
        self.style_image = None
        self.myGAN_model_old = None

        self.__load_model__()
        print(f'Client[{self.ChatID}].Initialize OK')

    def __load_model__(self):
        print(f'Client[{self.ChatID}].Load models')
        path = f'data/{self.ChatID}/model'
        h.dir_create_all(path)
        files = h.files(path)
        self.models = dict()
        for model in files:
            s = model
            k = s.index('/')
            while k > 0:
                s = s[k, ]
            s = s[0, -5]
            self.models[s] = tf.keras.models.load_model(model)
        if not ("_" in self.models):
            self.models["_"] = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        print(f'Client[{self.ChatID}].Load {len(self.models)}')

    @staticmethod
    def load_img(file_name_image):
        max_dim = 512
        img = Image.open(file_name_image)
        long = max(img.size)
        scale = max_dim / long
        img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
        img = kp_image.img_to_array(img)
        # Нам нужно транслировать массив изображений так, чтобы он имел размер партии
        img = np.expand_dims(img, axis=0)
        return img

    def imshow(self, img, title=None):
        if not self.isColabRun:
            return
        # Remove the batch dimension
        out = np.squeeze(img, axis=0)
        # Normalize for display
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)

    def load_and_process_img(self, file_name_image):
        img = self.load_img(file_name_image)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    @staticmethod
    def deprocess_img(processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")

        # perform the inverse of the preprocessing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def get_model(self):
        """ Creates our model with access to intermediate layers.

        This function will load the VGG19 model and access the intermediate layers.
        These layers will then be used to create a new model that will take input image
        and return the outputs from these intermediate layers from the VGG model.

        Returns:
          returns a keras model that takes image inputs and outputs the style and
            content intermediate layers.
        """
        # Load our model. We load pretrained VGG, trained on imagenet data
        if not h.isFile('data/model/VGG19.h5'):
            vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
            tf.keras.models.save_model(vgg, f'data/model/VGG19.h5')
        else:
            vgg = tf.keras.models.load_model('data/model/VGG19.h5')

        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs
        # Build model
        return models.Model(vgg.input, model_outputs)

    @staticmethod
    def get_content_loss(base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    @staticmethod
    def gram_matrix(input_tensor):
        # We make the image channels first
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    @staticmethod
    def get_style_loss(base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.get_shape().as_list()
        gram_style = Client.gram_matrix(base_style)

        return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)

    def get_feature_representations(self, model, content_path, style_path):
        """Helper function to compute our content and style feature representations.

        This function will simply load and preprocess both the content and style
        images from their path. Then it will feed them through the network to obtain
        the outputs of the intermediate layers.

        Arguments:
          model: The model that we are using.
          content_path: The path to the content image.
          style_path: The path to the style image

        Returns:
          returns the style features and the content features.
        """
        # Load our images in
        self.content_image = self.load_and_process_img(content_path)
        self.style_image = self.load_and_process_img(style_path)

        # batch compute content and style features
        style_outputs = model(self.style_image)
        content_outputs = model(self.content_image)

        # Get the style and content feature representations from our model
        style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
        content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
        return style_features, content_features

    def compute_loss(self, model, loss_weights, init_image, gram_style_features, content_features):
        """This function will compute the loss total loss.
        Arguments:
          model: The model that will give us access to the intermediate layers
          loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
          init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
          gram_style_features: Precomputed gram matrices corresponding to the
            defined style layers of interest.
          content_features: Precomputed outputs from defined content layers of
            interest.

        Returns:
          returns the total loss, style loss, content loss, and total variational loss
        """
        style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = model(init_image)

        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        weight_per_style_layer = 1.0 / float(self.num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)

        # Accumulate content losses from all layers
        weight_per_content_layer = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * self.get_content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        # Get total loss
        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_grads(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    def run_style_transfer(self,
                           content_path,
                           style_path,
                           num_iterations=1000,
                           content_weight=1e3,
                           style_weight=1e-2):
        # We don't need to (or want to) train any layers of our model, so we set their
        # trainable to false.

        print(f'content_path = {content_path}')
        print(f'style_path = {style_path}')
        print(f'Visible: {isColabRun}')

        model = self.get_model()
        for layer in model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        style_features, content_features = self.get_feature_representations(model, content_path, style_path)
        gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]

        # Set initial image
        init_image = self.load_and_process_img(content_path)
        init_image = tf.Variable(init_image, dtype=tf.float32)
        # Create our optimizer
        opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

        # For displaying intermediate images
        iter_count = 1

        # Store our best result
        best_loss, best_img = float('inf'), None

        # Create a nice config
        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }

        # For displaying
        num_rows = 2
        num_cols = 5
        display_interval = num_iterations / (num_rows * num_cols)
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        message_id = None
        self.myGAN_Images = []
        for i in range(num_iterations):
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            end_time = time.time()

            if loss < best_loss:
                # Update best loss and best image from total loss.
                best_loss = loss
                best_img = self.deprocess_img(init_image.numpy())

            if i % display_interval == 0:
                start_time = time.time()

                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = self.deprocess_img(plot_img)

                self.myGAN_Images.append(plot_img)

                image_visible_and_send = Image.fromarray(plot_img)
                if isColabRun:
                    IPython.display.clear_output(wait=True)
                    IPython.display.display_png(image_visible_and_send)
                image_visible_and_send.save(f'data/{self.ChatID}/Output.jpg')

                f = Image.open(f'data/{self.ChatID}/Output.jpg')
                message_id = self.send_run(f, message_id, text=f'Итерация: {i}/{num_iterations}')
                
                print('Iteration: {}'.format(i))
                print('Total loss: {:.4e}, '
                      'style loss: {:.4e}, '
                      'content loss: {:.4e}, '
                      'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
        print('Total time: {:.4f}s'.format(time.time() - global_start))

        # для отображения подбора.
        plt.figure(figsize=(14, 4))
        for i, img in enumerate(self.myGAN_Images):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.savefig(f'data/{self.ChatID}/Outputs.jpg')

        image_visible_and_send = Image.open(f'data/{self.ChatID}/Outputs.jpg')
        self.send_run(image_visible_and_send, message_id, 'Обработка закончена: ')

        if isColabRun:
            IPython.display.clear_output(wait=True)
            plt.show()

        image_visible_and_send = Image.fromarray(best_img)
        image_visible_and_send.save(f"image/{self.ChatID}/Output.jpg")
        self.TeleBot.send_photo(self.ChatID, image_visible_and_send)

        self.myGAN_Best = best_img
        self.myGAN_file_content = None
        self.myGAN_file_style = None

        return best_img, best_loss

    def show_results(self, best_img, file_content, file_style, show_large_final=True):
        if not isColabRun:
            return
        plt.figure(figsize=(10, 5))
        content = self.load_img(file_content)
        style = self.load_img(file_style)

        plt.subplot(1, 2, 1)
        self.imshow(content, 'Content Image')

        plt.subplot(1, 2, 2)
        self.imshow(style, 'Style Image')

        if show_large_final:
            plt.figure(figsize=(10, 10))
            plt.imshow(best_img)
            plt.title('Output Image')
            plt.show()

    def send_run(self, image, messageID=None, text=None):
        if text is not None:
            self.TeleBot.send_message(self.ChatID, text)
        
        if messageID is None:
            message = self.TeleBot.send_photo(self.ChatID, image)
            return message.message_id
        else:
            # self.TeleBot.edit_message_media(media=telebot.types.InputMedia(type='photo', media=image),
            #                                 chat_id=self.ChatID, message_id=messageID)
            message = self.TeleBot.send_photo(self.ChatID, image)
            self.TeleBot.delete_message(self.ChatID, messageID)
            return message.message_id
        return messageID

    def clear(self):
        self.waitCountImage = 2
        self.countImage = 0

    def post_photo(self, photo_file):  # применение стиля
        (self.One, self.Two) = (self.Two, photo_file)
        self.countImage += 1

        self.TeleBot.send_message(self.ChatID, f'Изображений: {self.countImage} из {self.waitCountImage}.')
        
        print(f'[{self.ChatID}]: Wait={self.waitCountImage}; Count={self.countImage};')
        if self.waitCountImage == self.countImage:
            self.GAN_Style(self.One, self.Two)
            self.waitCountImage = 2
            self.countImage = 0

    def GAN_Style(self, file_content, file_style):
        if self.waitCountImage != 2:
            self.TeleBot.send_message(self.ChatID, f'Изменение стиля не возможно при {self.waitCountImage} изображениях.')
            return
        # myGAN
        print(f'[{self.ChatID}]:Run style transfer')
        self.run_style_transfer(file_content, file_style, num_iterations=1000)


# Сервер
# Загрузка справки
if os.path.isfile('help.txt'):
    file = open('help.txt', 'r')
    helpText = '\r\n'.join(file.readlines())
    file.close()
else:
    helpText = 'Здесь должна быть справка...'
    file = open('help.txt', 'w')
    file.write(helpText)
    file.close()

TOKEN = '1423071373:AAFH4pMoWGsGdGKYZGGMOC6n_8uZzT5TEis'
Clients = dict()



# Определите несколько обработчиков команд. Обычно они требуют обновления двух аргументов и
# context. Обработчики ошибок также получают поднятый объект TelegramError по ошибке.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')

def help(update, context):
    """Send a message when the command /help is issued."""
    //update.message.reply_text('Help!')
    update.message.send_message(helpText)

def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main():
    """Start the bot."""
    # Создайте Updater и передайте ему токен вашего бота.
    # Обязательно установите use_context = True, чтобы использовать новые обратные вызовы на основе контекста
    # После версии 12 в этом больше не будет необходимости
    updater = Updater(TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)
    updater.bot.setWebhook('https://python-gan-bot.herokuapp.com/' + TOKEN)

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()


