
import numpy as np
import sys
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QPushButton,  QLineEdit, QMessageBox
from qimage2ndarray import recarray_view
import cv2
import nnleaner
from PIL import Image


class DrawingField(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drawing = False
        self.image = QImage(256, 256, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.last_point = None
        self.line_size = 16

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, self.line_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(QPoint(0, 0), self.image)


class Window(QMainWindow):
    def __init__(self, weights):
        super().__init__()
        self.setWindowTitle('Распознавание цифр')
        self.setFixedSize(630, 420)
        self.numImg = 0
        self.numImgPunish = 0
        self.predict = 0

        self.input = DrawingField(self)
        self.input.setGeometry(20, 80, 256, 256)

        self.equal_lbl = QLabel(self, text="=")
        self.equal_lbl.setFont(QFont("", 100))
        self.equal_lbl.setGeometry(280, 140, 100, 100)

        self.output = QLineEdit(self)
        self.output.setFont(QFont("", 160))
        self.output.setGeometry(360, 80, 256, 256)

        self.clear_button = QPushButton(self, text="Очистить")
        self.clear_button.setGeometry(20, 10, 256, 50)
        self.clear_button.clicked.connect(self.clear_input)

        self.translate_button = QPushButton(self, text="Распознать")
        self.translate_button.setGeometry(360, 10, 256, 50)
        self.translate_button.clicked.connect(self.translate_number)

        self.punish_button = QPushButton(self, text="Наказать")
        self.punish_button.setGeometry(20, 330, 600, 50)
        self.punish_button.clicked.connect(self.punish)

        self.show()

    def clear_input(self):
        self.input.image.save("Dataset\i03.png", "PNG")
        img = Image.open('Dataset\i03.png')
        resized_img = img.resize((32, 32), Image.Resampling.LANCZOS)
        resized_img.save('Dataset\i03.png')
        self.input.image.fill(Qt.white)
        self.repaint()

    def punish(self):
        path = "Dataset\p" + str(self.numImgPunish) + ".png"
        self.input.image.save(path, "PNG")
        self.numImgPunish += 1
        img = Image.open(path)
        resized_img = img.resize((32, 32), Image.Resampling.LANCZOS)
        resized_img.save(path)
        prediction = nnleaner.punish(path, weights, self.predict)
        self.predict = prediction
        self.output.setText(str(prediction))



    def translate_number(self):
        path = "Dataset\m" + str(self.numImg) + ".png"
        self.input.image.save(path, "PNG")
        self.numImg += 1
        img = Image.open(path)
        resized_img = img.resize((32, 32), Image.Resampling.LANCZOS)
        resized_img.save(path)
        prediction = nnleaner.get_value(path, weights)
        self.predict = prediction
        self.output.setText(str(prediction))


if __name__ == "__main__":
    weights = nnleaner.lean()
    app = QApplication([])
    window = Window(weights)

    sys.exit(app.exec_())