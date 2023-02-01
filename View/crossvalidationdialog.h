#ifndef CROSSVALIDATIONDIALOG_H
#define CROSSVALIDATIONDIALOG_H

#include <QDialog>
#include <memory>

namespace Ui {
class CrossValidationDialog;
}
namespace s21 {
class CrossValidationDialog;
}

class s21::CrossValidationDialog : public QDialog {
  Q_OBJECT

 public:
  explicit CrossValidationDialog(QWidget *parent = nullptr);
  ~CrossValidationDialog();
  QString file_name() const noexcept;
  int groups_count() const;

 private:
  void ChooseFileBtnClicked();

 private:
  std::unique_ptr<Ui::CrossValidationDialog> ui_;
  QString file_name_;
};

#endif  // CROSSVALIDATIONDIALOG_H
