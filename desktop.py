from tkinter import *
from tkinter import messagebox
from PIL import ImageTk
from tkinter import filedialog as fd

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

def analyze_sentences():
    predictions_tf.delete(1.0, END)
    sentences = sentences_tf.get(1.0, END).split('\n')

    sentences_correctness = []
    for i in range(len(sentences) - 1):
        iscorrect = bool(sentences[i].strip())
        sentences_correctness.append(iscorrect)
        if not iscorrect:
            messagebox.showinfo('Некорректное предложение',
                f'Предложение {i+1} введено некорректно')

    if not all(sentences_correctness):
        messagebox.showinfo('Ошибка анализа',
            'Введите корректный текст для анализа')
    else:
        if model_type.get() == 0:
            predictions = model_ru.predict(np.array(sentences[:-1]))
        else:
            predictions = model.predict(np.array(sentences[:-1]))

        if output_type.get() == 0:
            for x in range(0, len(predictions)):
                predictions_tf.insert(float(x+1), str(predictions[x]) + '\n') 

            messagebox.showinfo('Среднее',
                str(sum(predictions) / len(predictions)))

        else:
            for x in range(0, len(predictions)):
                if predictions[x] < -0.1:
                    predictions_tf.insert(float(x+1), 'Негативный' + '\n')
                elif predictions[x] > 0.1:
                    predictions_tf.insert(float(x+1), 'Положительный' + '\n')
                else:
                    predictions_tf.insert(float(x+1), 'Нейтральный' + '\n')
            pred_avg = sum(predictions) / len(predictions)
            if pred_avg < -0.1:
                messagebox.showinfo('Среднее', 'Негативно')
            elif pred_avg > 0.1:
                messagebox.showinfo('Среднее', 'Положительно')
            else:
                messagebox.showinfo('Среднее', 'Нейтрально')
            

def analyze_csv():
    file_name = fd.askopenfilename(filetypes=(('CSV files', '*.csv'),))

    df = pd.read_csv(file_name, names=['text'], skiprows=1)
    text_list = df['text'].to_list()

    if model_type.get() == 0:
        predictions = model_ru.predict(np.array(text_list))
    else:
        predictions = model.predict(np.array(text_list))

    sentiment_df = pd.DataFrame(
        [[text_list[i], predictions[i][0]] for i in range(len(text_list))],
        columns=['text', 'sentiment'],
    )

    sentiment_df.to_csv(file_name, index=False)

    messagebox.showinfo('Результат', 'Успешно')

def show_help():
    messagebox.showinfo('Помощь', 'Язык анализа следует выбирать исходя '
        'из языка анализируемого текста.\n'
        'Формат вывода определяет в каком виде будет '
        'показан результат: числовом или словесном.\n'
        'Для получения результата необходимо в поле '
        'анализа ввести текст и нажать на кнопку "Анализировать".\n'
        'При выборе ".csv" файла результат, после анализа, '
        'автоматически запишется в этот файл.\n'
        'Файл должен содержать единственный столбец с анализируемым текстом.\n'
        'После анализа, в тот же файл, добавится второй '
        'столбец с результатами.\n'
        'При работе с ".csv" файлом всегда используется формат вывода "Точный"'
    )

import_model_path = './model_multi_lstm'
import_model_ru_path = './model_multi_lstm_ru'

model = tf.keras.models.load_model(
    import_model_path,
    compile=False,
    custom_objects={'KerasLayer': hub.KerasLayer})

model_ru = tf.keras.models.load_model(
    import_model_ru_path,
    compile=False,
    custom_objects={'KerasLayer': hub.KerasLayer})

window = Tk()
window.geometry('900x450')
window.title("Анализ тональности")

main_menu = Menu(window) 
window.config(menu=main_menu)

filemenu = Menu(main_menu, tearoff=0)
filemenu.add_command(
    label="Открыть",
    command=analyze_csv,
)

main_menu.add_cascade(label="Файл",
                     menu=filemenu)
                     
frame = Frame(
    window,
    padx = 10,
)
frame.pack(expand=True)

sentences_lb = Label(
    frame,
    text="Введите текст дла анализа\n"
        "(следующий элемент начинать c новой строки)",
    font=20,
)

predictions_lb = Label(
    frame,
    text="Результаты анализа",
    font=20,
)

sentences_tf = Text(
    frame,
    width=80,
    height=16,
)

predictions_tf = Text(
    frame,
    width=16,
    height=16,
)

analyze_btn = Button(
    frame,
    text='Анализировать',
    font=20,
    command=analyze_sentences,
)

model_type = IntVar()
model_type.set(0)

output_type = BooleanVar()
output_type.set(0)

analyze_language_lb = Label(
    frame,
    text='Язык анализа',
    font=14,
)

analyze_ru_rbtn = Radiobutton(
    frame,
    text='Русский',
    width=10,
    height=1,
    variable=model_type,
    value=0,
)

analyze_en_rbtn = Radiobutton(
    frame,
    text='Английский',
    width=10,
    height=1,
    variable=model_type,
    value=1,
)

output_type_lb = Label(
    frame,
    text='Формат вывода',
    font=14,
)

precise_out_rbtn = Radiobutton(
    frame, 
    text='Точный', 
    width=10,
    height=1,
    variable=output_type,
    value=0, 
)

simplified_out_rbtn = Radiobutton(
    frame, 
    text='Упрощённый', 
    width=10,
    height=1,
    variable=output_type, 
    value=1,
)

help_btn = Button(
    frame,
    text='Помощь',
    width=8,
    height=2,
    command=show_help
)

analyze_language_lb.place(x=70)
analyze_en_rbtn.place(x=30, y=25)
analyze_ru_rbtn.place(x=150, y=25)

output_type_lb.place(x=310)
precise_out_rbtn.place(x=280, y=25)
simplified_out_rbtn.place(x=380, y=25)

help_btn.grid(row=1, column=2)

sentences_lb.grid(row=2, column=1, pady=5)
sentences_tf.grid(row=3, column=1)

predictions_lb.grid(row=2, column=2, padx=10, pady=5)
predictions_tf.grid(row=3, column=2, padx=10)

analyze_btn.grid(row=4, column=1, pady=5)

scroll_sentences = Scrollbar(command=sentences_tf.yview)
sentences_tf.config(yscrollcommand=scroll_sentences.set)

scroll_pred = Scrollbar(command=sentences_tf.yview)
sentences_tf.config(yscrollcommand=scroll_pred.set)

window.mainloop()