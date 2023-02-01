#ifndef TESTDIALOG_H
#define TESTDIALOG_H

#include <QDialog>
#include <memory>

namespace Ui {
class TestDialog;
}

namespace s21 {
class TestDialog;
}

class s21::TestDialog : public QDialog {
  Q_OBJECT

 public:
  struct TestData {
    QString file_name;
    double percent;
  };

 public:
  explicit TestDialog(QWidget *parent = nullptr);
  ~TestDialog();
  TestData data();

 private slots:
  void SpinboxSetValueSlot(int);
  void ChooseFileSlot();

 private:
  std::unique_ptr<Ui::TestDialog> ui_;
  TestData data_;
};

#endif  // TESTDIALOG_H
