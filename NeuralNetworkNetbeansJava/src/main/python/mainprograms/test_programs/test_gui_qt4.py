#! /usr/bin/python2.7
# -*- coding: utf-8 -*-
#

import sys
import PyQt4.QtGui as qt

class PictureEvaluation(qt.QWidget):
    def __init__(self):
        qt.QWidget.__init__(self)
        # self.move(10, 20)
        self.resize(320, 240)
        self.setWindowTitle("Hello World!")

        self.gridLayout = qt.QGridLayout()
        self.vboxLayout = qt.QVBoxLayout()
        self.hboxLayouts = [qt.QHBoxLayout(), qt.QHBoxLayout(), qt.QHBoxLayout()]

        # self.hboxLayouts[1].setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)

        # self.centralWidget = qt.QWidget()
        self.buttons = []

        self.button = qt.QPushButton()
        self.button.setText("click me!")
        self.button.move(10, 10)
        self.button.resize(200, 50)
        self.button.setParent(self)
        self.button.show()

        self.button.clicked.connect(self.on_click)
        self.button.pressed.connect(self.on_press)
        self.button.released.connect(self.on_release)

        self.button2 = qt.QPushButton()
        self.button2.setText("click me too!")
        self.button2.resize(150, 100)
        self.button2.setParent(self)
        self.button2.show()

        self.buttons.append(qt.QPushButton())
        self.buttons[0].setText("Button 1")
        self.buttons[0].resize(20, 20)
        self.buttons[0].setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)
        self.buttons.append(qt.QPushButton())
        self.buttons[1].setText("Button 2")
        self.buttons[1].setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)
        self.buttons.append(qt.QPushButton())
        self.buttons[2].setText("Button 3")
        self.buttons[2].setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)

        self.buttons.append(qt.QPushButton())
        self.buttons[3].setText("Button 4")
        self.buttons[3].resize(10,25)
        self.buttons[3].setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Expanding)

        self.buttons.append(qt.QPushButton())
        self.buttons[4].setText("Button 5")
        self.buttons.append(qt.QPushButton())
        self.buttons[5].setText("Button 6")

        self.hboxLayouts[0].addWidget(self.buttons[0])
        self.hboxLayouts[0].addWidget(self.buttons[1])
        self.hboxLayouts[0].addWidget(self.buttons[2])
        self.vboxLayout.addLayout(self.hboxLayouts[0])

        self.hboxLayouts[1].addWidget(self.buttons[3])
        self.vboxLayout.addLayout(self.hboxLayouts[1])

        self.hboxLayouts[2].addWidget(self.buttons[4])
        self.hboxLayouts[2].addWidget(self.buttons[5])
        self.vboxLayout.addLayout(self.hboxLayouts[2])

        # self.setCentralWidget(self.centralWidget)
        # self.centralWidget.setLayout(self.gridLayout)
        self.setLayout(self.vboxLayout)
    # def __init__

    def on_click(self):
        print("clicked: "+str(self.sender().text())+"   pos = "+str(self.button.pos().y()))
        self.button.move(10, self.button.pos().y() + 10)
    # def on_clicked

    def on_press(self):
        print("press")
    # def on_clicked

    def on_release(self):
        print("release")
    # def on_clicked

# class

if __name__ == "__main__":
    a = qt.QApplication(sys.argv)

    w = PictureEvaluation() #qt.QWidget()

    # w.resize(320, 240)

    # w.setWindowTitle("Hello World!")

    w.show()

    sys.exit(a.exec_())
# if
