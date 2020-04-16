from PyQt5 import QtWidgets, uic
import sys

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        """
        Housekeeping startup.
        """
        super (Ui, self).__init__()

        # load XML file with gui definition (made in Qt Creator)
        # https://en.wikipedia.org/wiki/Qt_Creator <- gui designer
        uic.loadUi('mainwindow.ui', self)

        # Find button
        self.button = self.findChild (QtWidgets.QPushButton,\
                'pushButton')

        # Find labels
        self.res = []
        for i in range (3):
            self.res.append (self.findChild (QtWidgets.QLabel,\
                'res'+str(i+1)))

        # Find text fields
        self.var = []
        for i in range (3):
            self.var.append (self.findChild (QtWidgets.QLineEdit,\
                    'var'+str(i+1)+'LE'))

        # Set default text on text fields, also set max number of digits
        [var.setPlaceholderText ('input var') for var in self.var]
        [var.setMaxLength (15) for var in self.var]

        # Connect Enter key on text field to button action
        [var.returnPressed.connect (self.printButtonPressed)\
                for var in self.var]

        # Connect button to desired action
        self.button.clicked.connect (self.printButtonPressed)

        self.show()

    def printButtonPressed (self):
        """
        Actions to perform when button (or Enter key) are pressed:
            - Sets QLabels to the number read from the variable
            QLineEdit (var).
            - If parse fails it print on console 'Parsing Exception'
            - If parse succeeds it prints the variables read on the
                console.
        """

        # Read QLineEdit (var) text and set it to the QLabels
        try:
            [res.setNum (float (var.text () or float (res.text ())))\
                    for var, res in zip (self.var, self.res)]
        except:
            print ('Parsing Exception')

        # Print on console
        [print (var.text ()) for var in self.var]

        # Clear and reset QLineEdit text fields
        [(var.clear (), var.repaint ()) for var in self.var]

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
