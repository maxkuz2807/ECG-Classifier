import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from ECGWindow import Ui_MainWindow

model = load_model(r'C:\Users\Максим\Desktop\Аналитика данных\Нейросети\cnn_ecg_clf.h5')
BATCH_SIZE = 32
img_height = 80
img_width = 120
fileName = None
# Мне не удалось понять, что означает 1 класс изображений, поэтому я назвал его неопознанным
ecg_dict = {'Смешанный ритм (fusion beat)': 0,
            'Неопознанный ритм': 1,
            'Нормальный ритм': 2,
            'Неопознанный ритм': 3,
            'Наджелудочковая экстрасистолия': 4,
            'Желудочковая экстрасистолия': 5}

class Window_1(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Window_1, self).__init__(parent)
        self.setupUi(self)

    def loadImage(self):
        global fileName
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            pixmap = QPixmap(fileName)
            self.ecgImage.setPixmap(pixmap)
            self.ecgImage.resize(pixmap.width(), pixmap.height())
            self.ecgImage.setStyleSheet("border: 1px solid blue; background-color: white")
            self.predictionLabel.setText('')
            print(fileName)

    def classification(self):
        if fileName:
            img = image.load_img(r'{}'.format(fileName), target_size=(img_height, img_width), color_mode='grayscale')
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            prediction = np.argmax(model.predict(img))
            rhythm_type = next((k for k, v in ecg_dict.items() if v == prediction), None)
            self.predictionLabel.setText(f"Результат классификации: {rhythm_type}")
        else:
            self.ecgImage.setStyleSheet("border: 3px solid red; background-color: white")
            self.ecgImage.setText('Выберите изображение!')
            self.ecgImage.setAlignment(Qt.AlignCenter)


def main():
    app = QApplication(sys.argv)
    window1 = Window_1()

    window1.loadButton.clicked.connect(window1.loadImage)
    window1.predictButton.clicked.connect(window1.classification)

    window1.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
