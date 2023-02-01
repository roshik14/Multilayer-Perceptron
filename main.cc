#include <QApplication>

#include "View/mainview.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  s21::MainView main_window;
  main_window.show();
  return a.exec();
}
