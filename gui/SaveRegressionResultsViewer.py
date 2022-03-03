# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SaveRegressionResultsViewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPoint, pyqtSignal, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QWidget, QApplication, QDialog, QFileDialog, QColorDialog


class SaveRegressionResultsViewer(QDialog):
    ## results_prefix, target_dir, is_save_reg_flim_image, is_save_blending_image, is_save_loss_curve
    save_results_confirmed = pyqtSignal(object, object, object, object, object, object, object, object)

    def __init__(self, parent=None, *args, **kwargs):
        super(SaveRegressionResultsViewer, self).__init__(parent)
        self.setObjectName("SaveRegressionResultsViewer")
        self.resize(488, 201)
        self.setWindowModality(Qt.ApplicationModal)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.resultPrefixLineEdit = QtWidgets.QLineEdit(self)
        self.resultPrefixLineEdit.setObjectName("resultPrefixLineEdit")
        self.gridLayout.addWidget(self.resultPrefixLineEdit, 0, 1, 1, 2)

        self.targetDirLineEdit = QtWidgets.QLineEdit(self)
        self.targetDirLineEdit.setObjectName("targetDirLineEdit")
        self.gridLayout.addWidget(self.targetDirLineEdit, 2, 1, 1, 1)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.selectDirButton = QtWidgets.QPushButton(self)
        self.selectDirButton.setObjectName("selectDirButton")
        self.selectDirButton.clicked.connect(self.toggle_selectDirButton)
        self.gridLayout.addWidget(self.selectDirButton, 2, 2, 1, 1)

        self.paddingColourGroupbox = QtWidgets.QGroupBox(self)
        self.paddingColourGroupbox.setObjectName("paddingColourGroupbox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.paddingColourGroupbox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.histoPaddingColourButton = QtWidgets.QPushButton(self.paddingColourGroupbox)
        self.histoPaddingColourButton.setObjectName("histoPaddingColourButton")
        self.histoPaddingColourButton.clicked.connect(self.toggle_histoPaddingColourButton)
        self.gridLayout_4.addWidget(self.histoPaddingColourButton, 0, 6, 1, 1)
        self.histoPaddingColourLabel = QtWidgets.QLabel(self.paddingColourGroupbox)
        self.histoPaddingColourLabel.setMinimumSize(QtCore.QSize(50, 0))
        self.histoPaddingColourLabel.setStyleSheet("border: 1px solid black;\n"
                                                   "background-color: rgb(213, 212, 235);")
        self.histoPaddingColourLabel.setText("")
        self.histoPaddingColourLabel.setObjectName("histoPaddingColourLabel")
        self.histoPaddingColourLabel.setAutoFillBackground(True)
        self.gridLayout_4.addWidget(self.histoPaddingColourLabel, 0, 5, 1, 1)

        self.label_5 = QtWidgets.QLabel(self.paddingColourGroupbox)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 0, 4, 1, 1)
        self.flimPaddingColourButton = QtWidgets.QPushButton(self.paddingColourGroupbox)
        self.flimPaddingColourButton.setObjectName("flimPaddingColourButton")
        self.flimPaddingColourButton.clicked.connect(self.toggle_flimPaddingColourButton)
        self.gridLayout_4.addWidget(self.flimPaddingColourButton, 0, 2, 1, 1)
        self.flimPaddingColourLabel = QtWidgets.QLabel(self.paddingColourGroupbox)
        self.flimPaddingColourLabel.setMinimumSize(QtCore.QSize(50, 0))
        self.flimPaddingColourLabel.setStyleSheet("border: 1px solid black;\n"
                                                  "background-color: rgb(0, 0, 0);")
        self.flimPaddingColourLabel.setText("")
        self.flimPaddingColourLabel.setObjectName("flimPaddingColourLabel")
        self.flimPaddingColourLabel.setAutoFillBackground(True)
        self.gridLayout_4.addWidget(self.flimPaddingColourLabel, 0, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.paddingColourGroupbox)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 0, 3, 1, 1)
        self.gridLayout.addWidget(self.paddingColourGroupbox, 3, 0, 1, 3)

        self.widget = QtWidgets.QWidget(self)
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.widget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.saveFlimImageRadioButton = QtWidgets.QCheckBox(self.groupBox)
        self.saveFlimImageRadioButton.setObjectName("saveFlimImageRadioButton")
        self.saveFlimImageRadioButton.setChecked(True)
        self.gridLayout_3.addWidget(self.saveFlimImageRadioButton, 0, 0, 1, 1)

        self.saveHistologyPatchRadioButton = QtWidgets.QCheckBox(self.groupBox)
        self.saveHistologyPatchRadioButton.setObjectName("saveHistologyPatchRadioButton")
        self.saveHistologyPatchRadioButton.setChecked(True)
        self.gridLayout_3.addWidget(self.saveHistologyPatchRadioButton, 0, 1, 1, 1)

        self.saveBlendingImageCheckbox = QtWidgets.QCheckBox(self.groupBox)
        self.saveBlendingImageCheckbox.setObjectName("saveBlendingImageCheckbox")
        self.saveBlendingImageCheckbox.setChecked(True)
        self.gridLayout_3.addWidget(self.saveBlendingImageCheckbox, 1, 0, 1, 1)

        self.saveLossCurveCheckbox = QtWidgets.QCheckBox(self.groupBox)
        self.saveLossCurveCheckbox.setObjectName("saveLossCurveCheckbox")
        self.gridLayout_3.addWidget(self.saveLossCurveCheckbox, 1, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 2)
        self.gridLayout.addWidget(self.widget, 4, 0, 1, 3)

        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.accepted.connect(self.toggle_saveButton)
        self.buttonBox.rejected.connect(self.close)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 5, 0, 1, 3)

        self.retranslateUi()

        if "flim_image_file" in kwargs:
            self.flim_image_file = kwargs.pop("flim_image_file")
            self.resultPrefixLineEdit.setText(self.flim_image_file)
        else:
            self.flim_image_file = None

        if "default_target_dir" in kwargs:
            self.target_dir = kwargs.pop("default_target_dir")
            self.targetDirLineEdit.setText(self.target_dir)
        else:
            self.target_dir = None

        self.flim_padding_colour = (0, 0, 0)
        self.histo_padding_colour = (213, 212, 235)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("self", "Save Regression Results"))
        self.label.setText(_translate("self", "Results prefix: "))
        self.label_2.setText(_translate("self", "Target dir: "))
        self.selectDirButton.setText(_translate("self", "Select Dir"))
        self.groupBox.setTitle(_translate("self", "Select images to be saved"))
        self.saveFlimImageRadioButton.setText(_translate("self", "Save Registered FLIM image"))
        self.saveHistologyPatchRadioButton.setText(_translate("self", "Save Registered Histology Patch"))
        self.saveBlendingImageCheckbox.setText(_translate("self", "Save Blending Image"))
        self.saveLossCurveCheckbox.setText(_translate("self", "Save Loss Curve"))
        self.paddingColourGroupbox.setTitle(_translate("Dialog", "Padding Colour"))
        self.histoPaddingColourButton.setText(_translate("Dialog", "Choose Colour"))
        self.label_4.setText(_translate("Dialog", "FLIM Image: "))
        self.flimPaddingColourButton.setText(_translate("Dialog", "Choose Colour"))
        self.label_5.setText(_translate("Dialog", "Histology Patch: "))

    def toggle_flimPaddingColourButton(self):
        self.flim_padding_colour = self._set_colour(self.flimPaddingColourLabel)

    def toggle_histoPaddingColourButton(self):
        self.histo_padding_colour = self._set_colour(self.histoPaddingColourLabel)

    def _set_colour(self, label):
        selected_colour = QColorDialog.getColor(QColor("black"), parent=self)
        label.setStyleSheet("border: 1px solid black;\nbackground-color: rgb({}, {}, {})"
                            .format(selected_colour.red(), selected_colour.green(), selected_colour.blue()))
        return (selected_colour.red(), selected_colour.green(), selected_colour.blue())


    def toggle_selectDirButton(self):
        dir = QFileDialog.getExistingDirectory(self, "Open Directory", ".",
                                               QFileDialog.ShowDirsOnly|QFileDialog.DontResolveSymlinks)
        if dir:
            self.target_dir = dir
            self.targetDirLineEdit.setText(dir)

    def toggle_saveButton(self):
        is_save_reg_flim = self.saveFlimImageRadioButton.isChecked()
        is_save_reg_histo = self.saveHistologyPatchRadioButton.isChecked()
        is_save_blending_image = self.saveBlendingImageCheckbox.isChecked()
        is_save_loss_curve = self.saveLossCurveCheckbox.isChecked()
        self.save_results_confirmed.emit(self.resultPrefixLineEdit.text(), self.targetDirLineEdit.text(),
                                         is_save_reg_flim, is_save_reg_histo, is_save_blending_image, is_save_loss_curve,
                                         self.flim_padding_colour, self.histo_padding_colour)
        self.accept()

if __name__ == "__main__":
    flim_image_file = "20210405_CR71A_4_FOV515_testing--2_3x3_Row_1_col_2"
    default_target_dir = "G:/inverted/processed/gen_images/fake_histo/lifetime"
    app = QApplication(sys.argv)
    win = SaveRegressionResultsViewer(flim_image_file=flim_image_file, default_target_dir=default_target_dir)
    win.show()
    sys.exit(app.exec_())