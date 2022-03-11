#! /usr/bin/python2.7

from javax.swing import *

frame = JFrame("Hello Jython")
label = JLabel("Hello Jython!", JLabel.CENTER)
frame.add(label)
frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
frame.setSize(300, 300)
frame.show()
