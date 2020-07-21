from PyQt5 import QtCore
from PyQt5.QtGui import QPalette, QColor

def apply_dark_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QPalette.Text, QtCore.Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QtCore.Qt.black)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QtCore.Qt.gray)
    palette.setColor(QPalette.Disabled, QPalette.Text, QtCore.Qt.gray)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QtCore.Qt.gray)
    app.setPalette(palette)

    #Thanks to: https://stackoverflow.com/questions/48256772/dark-theme-for-qt-widgets @ Michael Herrmann

