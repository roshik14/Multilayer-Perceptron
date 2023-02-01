#ifndef TRAINDIALOG_H_
#define TRAINDIALOG_H_

#include <QAbstractButton>
#include <QDialog>
#include <QString>
#include <memory>

#include "../useful_data.h"

namespace Ui {
class TrainDialog;
}

namespace s21 {
class TrainDialog;
}

class s21::TrainDialog : public QDialog {
  Q_OBJECT

 public:
  explicit TrainDialog(QWidget* parent = nullptr);
  ~TrainDialog();
  TrainData data() const;

 private slots:
  void ApplySlot();
  void ChooseFileSlot();

 private:
  QAbstractButton* GetApplyButton();

 private:
  std::unique_ptr<Ui::TrainDialog> ui_;
  QString file_name_;
};

#endif  // TRAINDIALOG_H_
