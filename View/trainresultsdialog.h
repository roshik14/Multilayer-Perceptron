#ifndef TRAINRESULTSDIALOG_H
#define TRAINRESULTSDIALOG_H

#include <QDialog>
#include <memory>
#include <vector>

namespace Ui {
class TrainResultsDialog;
}

namespace s21 {
class TrainResultsDialog;
}

class s21::TrainResultsDialog : public QDialog {
  Q_OBJECT

 public:
  explicit TrainResultsDialog(QWidget *parent = nullptr);
  ~TrainResultsDialog();
  void SetRows(int rows);
  void SetErrors(std::vector<std::vector<double>> &&errors) noexcept;
  void CreateTable();
  void ShowGraph();

 private:
  void InitColumnsHeader();
  void SetErrorsToRow(int row);

 private:
  std::unique_ptr<Ui::TrainResultsDialog> ui_;
  std::vector<std::vector<double>> errors_;
};

#endif  // TRAINRESULTSDIALOG_H
