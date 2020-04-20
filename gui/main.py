from PyQt5 import QtWidgets, uic
import sys
import importlib.util
import csv
import pandas as pd

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        """
        Housekeeping startup.
        """
        super (Ui, self).__init__()

        # load XML file with gui definition (made in Qt Creator)
        # https://en.wikipedia.org/wiki/Qt_Creator <- gui designer
        uic.loadUi('mainwindow.ui', self)

        # Menu
        self.menuFile = self.findChild (QtWidgets.QMenu,\
                'menuFile')

        # Open File
        self.menuFile.triggered[QtWidgets.QAction].connect (self.openpressed)

        # Radio Buttons
        self.rad = []
        for i in range (1,8):
            self.rad.append (self.findChild (QtWidgets.QRadioButton,\
                'radioButton_'+str(i)))

        [rad.toggled.connect (self.radioButtonPressed)\
                for rad in self.rad]

        # Find Buttons
        self.button = self.findChild (QtWidgets.QPushButton,\
                'readvars')

        # Find labels
        self.res = []
        for i in range (30):
            self.res.append (self.findChild (QtWidgets.QLabel,\
                'res_'+str(i+1)))

        self.out = []
        for i in range (1, 10):
            self.out.append (self.findChild (QtWidgets.QLabel,\
                'out'+str(i)))

        # Find text fields
        self.var = []
        for i in range (30):
            self.var.append (self.findChild (QtWidgets.QLineEdit,\
                    'var_'+str(i+1)))

        # Set default text on text fields, also set max number of digits
        [var.setPlaceholderText ('input var') for var in self.var]
        [var.setMaxLength (15) for var in self.var]

        # Connect Enter key on text field to button action
        [var.returnPressed.connect (self.printButtonPressed)\
                for var in self.var]

        # Connect button to desired action
        self.button.clicked.connect (self.printButtonPressed)

        self.show()

    def radioButtonPressed (self):
        b = self.sender ()
        if b.isChecked ():
            print (b.text ())
            #  modelos.train (b.text ())
            try:
                spec = importlib.util.spec_from_file_location (\
                        "modelos", self.modelospath)
                modelos = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(modelos)
                (self.modelo, res) = modelos.train (b.text ())
                pred = self.modelo.predict (self.paciente)

                print (self.paciente)
                print (res)
                print (pred)
                if pred: res.append ('enfermo')
                else: res.append ('sano')

                for var, out in zip (res, self.out):
                    if type (var) is str:
                        out.setText (var)
                    else:
                        out.setText ('%.2f'%(var))
            except:
                print ('exception')
                pass


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

        # Clear and reset QLineEdit text fields
        [(var.clear (), var.repaint ()) for var in self.var]


    def openpressed (self, q):
        if q.text () == 'Open Patient':
            fname = QtWidgets.QFileDialog.getOpenFileName (self,\
                    '', './', '*.csv')
            if fname[0] != '':
                with open(fname[0], newline='') as csvfile:
                    patient = list (csv.reader(csvfile))
                #  print (patient)

                [var.setText (pat) for var, pat in\
                        zip (self.var, patient[0])]

                self.paciente = pd.DataFrame (data = patient)
                [var.repaint () for var in self.var]


        if q.text () == 'Open Models':
                fname = QtWidgets.QFileDialog.getOpenFileName (self,\
                        '', './', '*.py')
                if fname[0] != '':
                    self.modelospath = fname[0]

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
