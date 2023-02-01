#ifndef GRAPHDIALOG_H
#define GRAPHDIALOG_H

#include <QDialog>
#include <QVector>
#include <memory>
#include <vector>

#include "View/qcustomplot.h"

namespace Ui {
class GraphDialog;
}

class GraphDialog : public QDialog {
  Q_OBJECT

 public:
  explicit GraphDialog(const std::vector<std::vector<double>> &errors,
                       QWidget *parent = nullptr);
  ~GraphDialog();

 private:
  QVector<double> InitEpochVector(size_t epoch_count);
  QVector<double> InitErrorsVector(size_t epoch_count);
  double CountErrorsSum(int i);

 private:
  std::unique_ptr<Ui::GraphDialog> ui_;
  std::vector<std::vector<double>> errors_;

  void DrawGraph();
};

#endif  // GRAPHDIALOG_H
