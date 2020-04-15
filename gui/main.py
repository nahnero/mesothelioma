from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow.ui', self)

        self.button = self.findChild (QtWidgets.QPushButton, 'pushButton')

        self.res1 = self.findChild (QtWidgets.QLabel, 'res1')
        self.res2 = self.findChild (QtWidgets.QLabel, 'res2')
        self.res3 = self.findChild (QtWidgets.QLabel, 'res3')
        self.titl = self.findChild (QtWidgets.QLabel, 'label_4')

        self.var1 = self.findChild (QtWidgets.QLineEdit, 'var1LE')
        self.var2 = self.findChild (QtWidgets.QLineEdit, 'var2LE')
        self.var3 = self.findChild (QtWidgets.QLineEdit, 'var3LE')

        self.button.clicked.connect (self.printButtonPressed)
        self.show()

    def printButtonPressed (self):
        self.res1.setText (self.var1.text ())
        self.res2.setText (self.var2.text ())
        self.res3.setText (self.var3.text ())
        self.titl.setText ("you gon' die boy")
        self.res1.adjustSize()
        self.res2.adjustSize()
        self.res3.adjustSize()
        self.titl.adjustSize ()
        print (self.var1.text ())
        print (self.var2.text ())
        print (self.var3.text ())

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
