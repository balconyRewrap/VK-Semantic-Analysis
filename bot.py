import nltk

nltk.download('punkt')
import telebot
import tensorflow as tf
import pickle
import scipy.sparse as sp


# Define a function to preprocess the data
def preprocess_data(data):
    X = []
    y = []
    for text, label in data:
        X.append(text)
        y.append(label)
    return X, y


# Define a function to tokenize the text
def tokenize(text):
    return text.split()


# Define a function to read the dataset
def read_dataset(file_path):
    data = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' & ')
            if len(parts) != 2:
                print(f"Ignoring line: {line}")
                count += 1
                print(count)
                continue
            text, label = parts
            data.append((text, label))

    return data


# создаем объект бота
bot = telebot.TeleBot('5979403687:AAEiP9XwXrzFB0yL3FNyVEw9dZiCHTBeECg')
# Объект клавиатуры
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
keyboard.add(KeyboardButton('Начать'))
keyboard.add(KeyboardButton('Помощь'))
text_of_message ="d"
# обработчик команды /start
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(message, 'Привет! Я бот, который определяет эмоциональный окрас сообщений.', reply_markup=keyboard)

# Load the trained model from file
model = tf.keras.models.load_model('text_classifier.h5')

# Load the vectorizer and transformer objects from file
with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)
with open('transformer.pickle', 'rb') as handle:
    transformer = pickle.load(handle)


# обработчик текстовых сообщений
@bot.message_handler(content_types=['text'])
def send_prediction(message):
    # получаем имя пользователя и текст сообщения
    username = message.chat.first_name
    text = message.text
    global text_of_message
    text_of_message=text
    print("User:", username, "\nmessage:", text)
    # Tokenize the text
    tokens = tokenize(message.text)

    # Vectorize and transform the text
    X = vectorizer.transform([' '.join(tokens)])
    X = sp.csr_matrix(X)
    X = transformer.transform(X)

    # Convert X to a dense tensor
    X = X.toarray()

    # Convert X to a Tensor
    X = tf.convert_to_tensor(X, dtype=tf.float32, name='X')

    # Make predictions
    predictions = model.predict(X)
    markup = telebot.types.InlineKeyboardMarkup()
    markup.row(
        telebot.types.InlineKeyboardButton("Positive", callback_data='Positive'),
        telebot.types.InlineKeyboardButton("Negative", callback_data='Negative'),
        telebot.types.InlineKeyboardButton("Neutral", callback_data='Neutral'),
        telebot.types.InlineKeyboardButton("Бот прав", callback_data='RightPred')
    )


    # Analyze the predictions
    if ((predictions[0][0] / predictions[0][1]) < 0.78) or ((predictions[0][1] / predictions[0][0]) < 0.78):
        if predictions[0][0] > predictions[0][1]:
            # отправляем ответ пользователю

            sentiment = 'positive'
            bot.reply_to(message,
                         f"Привет, {username}!" 
                         f"По вашему сообщению '{text}' мы определили положительный эмоциональный окрас. Как здорово! 😃\n"
                         f"Вероятность, что это позитивное сообщение = {round(predictions[0][0]*100,2)}%")
            # Send the message with the two buttons
            bot.send_message(message.chat.id, "Прав ли Я?\nНапишите правильный вариант, если уверены, что я не прав:",
                             reply_markup=markup)
            print(sentiment)
        else:
            # отправляем ответ пользователю

            sentiment = 'negative'
            bot.reply_to(message,
                         f"Привет, {username}. "
                         f"К сожалению, по вашему сообщению '{text}' мы определили отрицательный эмоциональный окрас. "
                         f"Не расстраивайтесь, ведь завтра точно будет лучше! 😉\n"
                         f"Вероятность, что это негативное сообщение = {round(predictions[0][1]*100,2)}%")
            # Send the message with the two buttons
            bot.send_message(message.chat.id, "Прав ли Я?\nНапишите правильный вариант, если уверены, что я не прав:",
                             reply_markup=markup)
            print(sentiment)
    else:
        bot.reply_to(message,
                     f"Привет, {username}. По вашему сообщению '{text}' мы пришли к следующим выводам: \n"
                     f"1) Сообщение имеет нейтральный эмоциональный окрас\n"
                     f"2) Сообщение имеет смешанный эмоциональный окрас\n"
                     f"3) Данное сообщение не имеет смысла вовсе, например, содержит лишь определенные символы."
                     f"Вероятность, что это позитивное сообщение = {round(predictions[0][0]*100,2)}%\n"
                     f"Вероятность, что это негативное сообщение = {round(predictions[0][1]*100,2)}%")
        # Send the message with the two buttons
        bot.send_message(message.chat.id, "Прав ли Я?\nНапишите правильный вариант, если уверены, что я не прав:", reply_markup=markup)

        print(f"Neutral")
    print("predictions[0][0]", predictions[0][0])
    print("predictions[0][1]", predictions[0][1])
    print("(predictions[0][0] / predictions[0][1])",(predictions[0][0] / predictions[0][1]))
    print("(predictions[0][1] / predictions[0][0])",(predictions[0][1] / predictions[0][0]))


# Handler function for button clicks
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    # Send the "It's wrong" message when either of the buttons is clicked
    if call.data == "Positive":
        bot.send_message(call.message.chat.id, "Спасибо, эти данные помогут стать боту лучше")
        with open('newdata.txt', 'a',encoding='utf-8') as f:
            f.write(str(text_of_message) + " & positive" '\n')
    if call.data == "Negative":
        bot.send_message(call.message.chat.id, "Спасибо, эти данные помогут стать боту лучше")
        with open('newdata.txt', 'a',encoding='utf-8') as f:
            f.write(str(text_of_message) + " & negative" '\n')
    if call.data == "Neutral":
        bot.send_message(call.message.chat.id, "Увы, для нейросети этот ответ бесполезен, но спасибо и так)")
    if call.data == "RightPred":
        bot.send_message(call.message.chat.id, "Слава богу")
import threading
import os
import time



# ID чата, куда будут отправляться сообщения
chat_id = '5222268833'

# Путь к файлу, который будем проверять на изменение
file_path = r'text_classifier.h5'

# Время последнего изменения файла
last_modified = os.path.getmtime(file_path)
print(last_modified)

# Функция, которая будет выполняться в фоновом потоке
def polling():
    # Запускаем бесконечный цикл получения обновлений
    while True:
        try:
            bot.polling(none_stop=True, interval=0, timeout=30)
        except Exception as e:
            print(e)
            time.sleep(15)


# Создаем и запускаем поток для метода polling()
t = threading.Thread(target=polling)
t.start()

# Основной цикл программы для проверки изменений файла
while True:
    # Получаем время последнего изменения файла
    current_modified = os.path.getmtime(file_path)

    # Если файл был изменен
    if current_modified - last_modified > 300:
        # Отправляем сообщение в Telegram
        bot.send_message(chat_id, 'File has been modified!')
        bot.send_message(chat_id, 'Current:'+str(current_modified)+"\nLast:"+str(last_modified)+"\n difference:"+str(current_modified - last_modified))
    # Обновляем время последнего изменения файла
    last_modified = current_modified

    # Ждем 5 минут перед следующей проверкой
    time.sleep(300)