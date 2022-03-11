#!/usr/bin/python2.7

from gi.repository import Gtk
import time
import sys

class TableWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="Table Example")
        self.set_size_request(600, 400)

        self.table = Gtk.Table(4, 4)
        self.add(self.table)

        self.buttons = []
        self.index = 4

        self.button1 = Gtk.Button(label="Button 1")
        self.button2 = Gtk.Button(label="Button 2")
        self.button3 = Gtk.Button(label="Button 3")
        self.button4 = Gtk.Button(label="Button 4")
        self.button5 = Gtk.Button(label="Button 5")
        self.button6 = Gtk.Button(label="Button 6")

        self.button1.connect("clicked", self.on_button1_clicked)
        self.button3.connect("clicked", self.on_button3_clicked)

        # self.table.attach(self.button1, 0, 1, 0, 1)
        # self.table.attach(self.button2, 1, 3, 0, 1)
        # self.table.attach(self.button3, 0, 1, 1, 3)
        # self.table.attach(self.button4, 1, 3, 1, 2)
        # self.table.attach(self.button5, 1, 2, 2, 3)
        # self.table.attach(self.button6, 2, 3, 2, 3)

        self.table.attach(self.button1, 0, 1, 0, 1)
        self.table.attach(self.button2, 3, 4, 1, 3)
        # self.table.attach(self.button3, 0, 1, 1, 3)
        # self.table.attach(self.button4, 1, 3, 1, 2)
        # self.table.attach(self.button5, 1, 2, 2, 3)
        # self.table.attach(self.button6, 2, 3, 2, 3)

        # bt1.set_property("width-request", 85)
        self.button1.set_size_request(30, 30)
        self.button2.set_size_request(60, 30)

        # self.buttons.append(Gtk.Button(label="ButtonN 1"))
        # self.table.attach(self.buttons[-1], 2, 3, 3, self.index)
        # self.index += 1


    def on_button1_clicked(self, button):
        print("change label of button2: "+str(time.strftime("%H:%M:%S")))
        self.button2.set_label(str(time.strftime("%H:%M:%S")))
        self.table.attach(self.button3, 2, 3, 1, 3)
    # def on_button1_clicked

    def on_button3_clicked(self, button):
        self.buttons.append(Gtk.Button(label="ButtonN "+str(self.index)))
        self.table.attach(self.buttons[-1], 2, 3, self.index-1, self.index)
        self.index += 1
    # def on_button3_clicked
# class

class SizeWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Size Window")
        self.set_size_request(200, 200)

        self.grid = Gtk.Grid()
        self.add(self.grid)

        self.index = 0

        self.grid.set_row_spacing(5)
        self.grid.set_column_spacing(5)

        self.button1 = Gtk.Button(label="Button 1")
        self.button1.set_size_request(150, 150)
        self.grid.attach(self.button1, 0, 0, 1, 2)

        self.button2 = Gtk.Button(label="Button 2")
        self.grid.attach(self.button2, 1, 0, 1, 1)

        self.button3 = Gtk.Button(label="Button 3")
        self.grid.attach(self.button3, 2, 0, 1, 1)

        self.button4 = Gtk.Button(label="Button 4")
        self.button4.set_size_request(60, 54)
        self.button4.connect("clicked", self.on_button_resize_click)
        self.grid.attach(self.button4, 1, 1, 2, 1)
    # def __init__

    def on_button_resize_click(self, button):
        width = button.get_allocation().width
        height = button.get_allocation().height
        print(str(width))
        print(str(height))
        button.set_size_request(width, height + 10)
    # def on_button_resize_click
# class

win = TableWindow()
win.connect("delete-event", Gtk.main_quit)
win.show_all()

win2 = SizeWindow()
win2.connect("delete-event", Gtk.main_quit)
win2.show_all()

Gtk.main()
