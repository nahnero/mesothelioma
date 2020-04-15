from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow.ui', self)

        self.button = self.findChild (QtWidgets.QPushButton, 'pushButton')

        self.res = []
        for i in range (3):
            self.res.append (self.findChild (QtWidgets.QLabel, 'res'+str(i+1)))

        self.var = []
        for i in range (3):
            self.var.append (self.findChild (QtWidgets.QLineEdit, 'var'+str(i+1)+'LE'))

        self.button.clicked.connect (self.printButtonPressed)
        self.show()

    def printButtonPressed (self):

        [res.setText (var.text ()) for var, res in zip (self.var, self.res)]
        [res.adjustSize () for res in self.res]
        [print (var.text ()) for var in self.var]

        [var.clear () for var in self.var]


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
