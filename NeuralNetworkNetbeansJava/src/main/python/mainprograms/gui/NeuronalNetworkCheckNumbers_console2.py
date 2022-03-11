#! /usr/bin/python2.7

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NeuronalNetworkCheckNumbers.ui'
#
# Created: Sat Feb  6 22:25:10 2016
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

import sys
import os

from PyQt4 import QtCore, QtGui, QtGui

qt = QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class PictureFrame(qt.QGraphicsView):
    def __init__(self, widget):
        qt.QGraphicsView.__init__(self, widget)
        self.Painter = qt.QPainter()
        self.can_draw = False
        self.x = 0
        self.y = 0

    def paintEvent(self, event):
        if self.can_draw:
            self.Painter.begin(self) #.viewport())
            self.Painter.setBrush(qt.QColor(0, 0, 0))
            self.Painter.setPen(qt.QColor(0, 0, 0))

            size = 20
            self.Painter.drawEllipse(self.x-int(size/2), self.y-int(size/2), 20, 20)

            self.Painter.end()

        # print("event.x() = "+str(event.x()))
        # print("event.y() = "+str(event.y()))
        pass

    def mouseMoveEvent(self, QMouseEvent):
        if self.can_draw:
            self.x, self.y = QMouseEvent.x(), QMouseEvent.y()
            # self.update()

    def mousePressEvent(self, QMouseEvent):
        self.can_draw = True
        print QMouseEvent.pos()

    def mouseReleaseEvent(self, QMouseEvent):
        self.can_draw = False
        cursor =QtGui.QCursor()
        print("mouse pos: "+str(QMouseEvent.pos())+"   cursor pos: "+str(cursor.pos()))

class Ui_MainWindow(qt.QMainWindow):
    def __init__(self):
        qt.QMainWindow.__init__(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(828, 509)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        self.lneX = QtGui.QLineEdit(self.centralwidget)
        self.lneX.setGeometry(QtCore.QRect(170, 30, 41, 21))
        self.lneX.setObjectName(_fromUtf8("lneX"))

        self.lneY = QtGui.QLineEdit(self.centralwidget)
        self.lneY.setGeometry(QtCore.QRect(230, 30, 41, 21))
        self.lneY.setObjectName(_fromUtf8("lneY"))

        self.lblIndex = QtGui.QLabel(self.centralwidget)
        self.lblIndex.setGeometry(QtCore.QRect(150, 10, 81, 16))
        self.lblIndex.setObjectName(_fromUtf8("lblIndex"))
        self.btnIndex = QtGui.QPushButton(self.centralwidget)
        self.btnIndex.setGeometry(QtCore.QRect(150, 60, 121, 31))
        self.btnIndex.setObjectName(_fromUtf8("btnIndex"))
        self.btnIndex.clicked.connect(self.on_click_change_index_label)

        self.btnLoadPicture = QtGui.QPushButton(self.centralwidget)
        self.btnLoadPicture.setGeometry(QtCore.QRect(150, 90, 121, 31))
        self.btnLoadPicture.setObjectName(_fromUtf8("btnLoadPicture"))
        self.btnLoadPicture.clicked.connect(self.on_click_load_picture)

        self.lblMainLabel = QtGui.QLabel(self.centralwidget)
        self.lblMainLabel.setGeometry(QtCore.QRect(5, 5, 140, 140))
        self.lblMainLabel.setFrameShape(QtGui.QFrame.Box)
        self.lblMainLabel.setFrameShadow(QtGui.QFrame.Plain)
        self.lblMainLabel.setLineWidth(1)
        self.lblMainLabel.setText(_fromUtf8(""))
        self.lblMainLabel.setScaledContents(True)
        self.lblMainLabel.setObjectName(_fromUtf8("lblMainLabel"))

        self.txtEdit = QtGui.QTextEdit(self.centralwidget)
        self.txtEdit.setGeometry(QtCore.QRect(3, 157, 411, 201))
        self.txtEdit.setObjectName(_fromUtf8("txtEdit"))

        self.lblIndex_2 = QtGui.QLabel(self.centralwidget)
        self.lblIndex_2.setGeometry(QtCore.QRect(150, 29, 20, 21))
        self.lblIndex_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblIndex_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lblIndex_2.setObjectName(_fromUtf8("lblIndex_2"))
        self.lblIndex_3 = QtGui.QLabel(self.centralwidget)
        self.lblIndex_3.setGeometry(QtCore.QRect(210, 25, 21, 31))
        self.lblIndex_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblIndex_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lblIndex_3.setObjectName(_fromUtf8("lblIndex_3"))

        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(440, 30, 361, 311))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.btnTestTab = QtGui.QPushButton(self.tab)
        self.btnTestTab.setGeometry(QtCore.QRect(30, 30, 85, 27))
        self.btnTestTab.setObjectName(_fromUtf8("btnTestTab"))
        self.btnTestTab.clicked.connect(self.on_click_copy_line)

        self.lneTestInput = QtGui.QLineEdit(self.tab)
        self.lneTestInput.setGeometry(QtCore.QRect(30, 70, 113, 27))
        self.lneTestInput.setObjectName(_fromUtf8("lneTestInput"))
        self.txtTextOutput = QtGui.QTextEdit(self.tab)
        self.txtTextOutput.setGeometry(QtCore.QRect(30, 100, 131, 101))
        self.txtTextOutput.setObjectName(_fromUtf8("txtTextOutput"))
        self.tabWidget.addTab(self.tab, _fromUtf8(""))

        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))

        self.gvGraphic1 = PictureFrame(self.centralwidget) #QtGui.QGraphicsView(self.centralwidget)
        self.gvGraphic1.setGeometry(QtCore.QRect(280, 20, 120, 120))
        self.gvGraphic1.setObjectName(_fromUtf8("gvGraphic1"))
        # self.gvGraphic1.

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 707, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuMenu = QtGui.QMenu(self.menubar)
        self.menuMenu.setObjectName(_fromUtf8("menuMenu"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionExit = QtGui.QAction(MainWindow)

        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.actionExit.triggered.connect(self.on_trigger_exit)

        self.actionInfo = QtGui.QAction(MainWindow)
        self.actionInfo.setObjectName(_fromUtf8("actionInfo"))
        self.menuMenu.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionInfo)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Painter = qt.QPainter()
        self.should_draw = False

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.lblIndex.setText(_translate("MainWindow", "Index", None))
        self.btnIndex.setText(_translate("MainWindow", "Calc Num", None))
        self.btnLoadPicture.setText(_translate("MainWindow", "Load Picture", None))
        self.lblIndex_2.setText(_translate("MainWindow", "x", None))
        self.lblIndex_3.setText(_translate("MainWindow", "y", None))
        self.btnTestTab.setText(_translate("MainWindow", "Test Tab", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Number Evaluation", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2", None))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu", None))
        self.menuHelp.setTitle(_translate("MainWindow", "Help", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionInfo.setText(_translate("MainWindow", "Info", None))

    def paintEvent(self, event):
        # paint = qt.QPainter()
        if self.should_draw:
            self.Painter.begin(self) #.viewport())
            self.Painter.setBrush(qt.QColor(0, 0, 0))
            self.Painter.setPen(qt.QColor(0, 0, 0))

            size = 20
            self.Painter.drawEllipse(self.xm-int(size/2), self.ym-int(size/2), 20, 20)

            self.Painter.end()
    # def paintEvent

    def mouseMoveEvent(self, event):
        # self.Brush =
        # self.Painter = qt.QPainter()

        # g = self.gvGraphic1
        # x1, y1 = g.x(), g.y()
        # x2, y2 = x1 + g.width(), y1 + g.height()
        xm, ym = event.x(), event.y()
        self.xm, self.ym = xm, ym
        # if x1 < xm and x2 > xm and y1 < ym and y2 > ym or True:
        if True:
            # self.gvGraphic1.update()
            self.should_draw = True
            self.update()
            # self.should_draw = False
            print("paint")
            # paint = qt.QPainter()
            # self.Painter.begin(self.gvGraphic1) #.viewport())
            # self.Painter.setBrush(qt.QColor(0, 0, 0))
            # self.Painter.setPen(qt.QColor(0, 0, 0))

            # self.Painter.drawEllipse(xm, ym, 2, 2)

            # self.Painter.end()
            pass
        # if

        # print("event = "+str(event))
        print("x = "+str(event.x()))
        print("y = "+str(event.y()))
        # print("pos of gvGraphic1: "+str(self.gvGraphic1.x())+", "+str(self.gvGraphic1.y())+", "+str(self.gvGraphic1.width())+", "+str(self.gvGraphic1.height()))
        print("")
    # def mouseMoveEvent

    def on_click_change_index_label(self):
        self.lblIndex.setText("Hallo!!!")
        x = int(self.lneX.text())
        y = int(self.lneY.text())
        img = self.lblMainLabel.pixmap().toImage()

        # print("x = "+str(x)+"   y = "+str(y)+"   color = "+str(img.pixel(x, y))+"   color hex = "+str(hex(img.pixel(x, y))))

        self.txtEdit.setText(str(hex(img.pixel(10*x, 10*y))))
        self.txtEdit.setText(self.txtEdit.toPlainText()+" Test!")

        img = self.lblMainLabel.pixmap().toImage()
        img = img.rgbSwapped()
        self.lblMainLabel.setPixmap(qt.QPixmap.fromImage(img))
        # self.txtEdit.setText(self.txtEdit.toPlainText()+"asfasfsfadsfjasdhflajkshfajks\nasfadsfadsfadsfs\nasdfajshfiouh5oiu43\n")

    def on_click_change_index_text(self):
        self.lblIndex.setText("Hallo!!!")

    def on_trigger_exit(self):
        QtCore.QCoreApplication.instance().quit()

    def on_click_load_picture(self):
        # pixmap = qt.QPixmap(os.getcwd() + "/test.png")
        self.lblMainLabel.setPixmap(qt.QPixmap(0, 0))
        # img_orig = qt.QPixmap(os.getcwd() + "/test_14x14.png").toImage()
        img_orig = qt.QPixmap(os.getcwd() + "/test_2.png").toImage()

        pmap_new = qt.QPixmap(140, 140)
        img = pmap_new.toImage()

        for x in xrange(0, 14):
            for y in xrange(0, 14):
                c = img_orig.pixel(x, y)
                for dx in xrange(0, 10):
                    for dy in xrange(0, 10):
                        img.setPixel(x*10+dx, y*10+dy, c)

        self.lblMainLabel.setPixmap(qt.QPixmap.fromImage(img))

    def on_click_copy_line(self):
        self.txtTextOutput.setText(str(self.lneTestInput.text()+"\n")*5)

if __name__ == "__main__":
    a = qt.QApplication(sys.argv)
    w = Ui_MainWindow() #qt.QWidget()
    w.setupUi(w)

    # w.resize(320, 240)
    # w.setWindowTitle("Hello World!")
    w.show()
    sys.exit(a.exec_())
# if
