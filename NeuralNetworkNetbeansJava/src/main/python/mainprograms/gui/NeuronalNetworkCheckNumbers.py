# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NeuronalNetworkCheckNumbers.ui'
#
# Created: Thu Feb 11 22:48:05 2016
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(828, 509)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.lneX = QtGui.QLineEdit(self.centralwidget)
        self.lneX.setGeometry(QtCore.QRect(170, 30, 41, 21))
        self.lneX.setObjectName(_fromUtf8("lneX"))
        self.lblIndex = QtGui.QLabel(self.centralwidget)
        self.lblIndex.setGeometry(QtCore.QRect(150, 10, 81, 16))
        self.lblIndex.setObjectName(_fromUtf8("lblIndex"))
        self.btnIndex = QtGui.QPushButton(self.centralwidget)
        self.btnIndex.setGeometry(QtCore.QRect(150, 60, 121, 31))
        self.btnIndex.setObjectName(_fromUtf8("btnIndex"))
        self.btnLoadPicture = QtGui.QPushButton(self.centralwidget)
        self.btnLoadPicture.setGeometry(QtCore.QRect(150, 90, 121, 31))
        self.btnLoadPicture.setObjectName(_fromUtf8("btnLoadPicture"))
        self.lblMainLabel = QtGui.QLabel(self.centralwidget)
        self.lblMainLabel.setGeometry(QtCore.QRect(5, 5, 140, 140))
        self.lblMainLabel.setFrameShape(QtGui.QFrame.Box)
        self.lblMainLabel.setFrameShadow(QtGui.QFrame.Plain)
        self.lblMainLabel.setLineWidth(1)
        self.lblMainLabel.setText(_fromUtf8(""))
        self.lblMainLabel.setObjectName(_fromUtf8("lblMainLabel"))
        self.txtEdit = QtGui.QTextEdit(self.centralwidget)
        self.txtEdit.setGeometry(QtCore.QRect(3, 157, 411, 201))
        self.txtEdit.setObjectName(_fromUtf8("txtEdit"))
        self.lneY = QtGui.QLineEdit(self.centralwidget)
        self.lneY.setGeometry(QtCore.QRect(230, 30, 41, 21))
        self.lneY.setObjectName(_fromUtf8("lneY"))
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
        self.lneTestInput = QtGui.QLineEdit(self.tab)
        self.lneTestInput.setGeometry(QtCore.QRect(30, 70, 113, 27))
        self.lneTestInput.setObjectName(_fromUtf8("lneTestInput"))
        self.txtTextOutput = QtGui.QTextEdit(self.tab)
        self.txtTextOutput.setGeometry(QtCore.QRect(30, 100, 131, 101))
        self.txtTextOutput.setObjectName(_fromUtf8("txtTextOutput"))
        self.hsldX = QtGui.QSlider(self.tab)
        self.hsldX.setGeometry(QtCore.QRect(30, 240, 251, 18))
        self.hsldX.setOrientation(QtCore.Qt.Horizontal)
        self.hsldX.setObjectName(_fromUtf8("hsldX"))
        self.lblX = QtGui.QLabel(self.tab)
        self.lblX.setGeometry(QtCore.QRect(40, 220, 16, 17))
        self.lblX.setObjectName(_fromUtf8("lblX"))
        self.label_2 = QtGui.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(65, 219, 41, 20))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.gvGraphic1 = QtGui.QGraphicsView(self.centralwidget)
        self.gvGraphic1.setGeometry(QtCore.QRect(280, 20, 120, 120))
        self.gvGraphic1.setObjectName(_fromUtf8("gvGraphic1"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 828, 27))
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
        self.actionInfo = QtGui.QAction(MainWindow)
        self.actionInfo.setObjectName(_fromUtf8("actionInfo"))
        self.menuMenu.addAction(self.actionExit)
        self.menuHelp.addAction(self.actionInfo)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.lblIndex.setText(_translate("MainWindow", "Index", None))
        self.btnIndex.setText(_translate("MainWindow", "Calc Num", None))
        self.btnLoadPicture.setText(_translate("MainWindow", "Load Picture", None))
        self.lblIndex_2.setText(_translate("MainWindow", "x", None))
        self.lblIndex_3.setText(_translate("MainWindow", "y", None))
        self.btnTestTab.setText(_translate("MainWindow", "Test Tab", None))
        self.lblX.setText(_translate("MainWindow", "X", None))
        self.label_2.setText(_translate("MainWindow", "0", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Number Evaluation", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2", None))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu", None))
        self.menuHelp.setTitle(_translate("MainWindow", "Help", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        self.actionInfo.setText(_translate("MainWindow", "Info", None))

