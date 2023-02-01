QT       += core gui printsupport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    Model/graphnetwork.cc \
    Model/graphneuron.cc \
    Model/matrixnetwork.cc \
    Model/neuralnetwork.cc \
    Model/weightsreader.cc \
    Model/weights.cc \
    Model/weightssaver.cc \
    Model/datasetreader.cc \
    Model/facade.cc \
    View/crossvalidationdialog.cc \
    View/fileviewer.cc \
    View/drawer.cc \
    View/mainview.cc \
    View/qcustomplot.cpp \
    View/testdialog.cc \
    View/traindialog.cc \
    Controller/controller.cc \
    View/trainresultsdialog.cc \
    View/graphdialog.cc \
    main.cc

HEADERS += \
    Model/facade.h \
    Model/neuron.h \
    Model/neuralnetwork.h \
    Model/graphnetwork.h \
    Model/matrixnetwork.h \
    Model/filereader.h \
    Model/weights.h \
    View/crossvalidationdialog.h \
    View/fileviewer.h \
    View/drawer.h \
    View/mainview.h \
    View/qcustomplot.h \
    View/testdialog.h \
    View/traindialog.h \
    View/trainresultsdialog.h \
    Controller/controller.h \
    View/graphdialog.h \
    useful_data.h

FORMS += \
    View/Ui/crossvalidationdialog.ui \
    View/Ui/graphdialog.ui \
    View/Ui/mainview.ui \
    View/Ui/testdialog.ui \
    View/Ui/traindialog.ui \
    View/Ui/trainresultsdialog.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
