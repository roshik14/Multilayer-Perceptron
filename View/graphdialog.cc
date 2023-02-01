#include "graphdialog.h"

#include <QDebug>
#include <QVector>
#include <vector>

#include "ui_graphdialog.h"

GraphDialog::GraphDialog(const std::vector<std::vector<double>> &errors,
                         QWidget *parent)
    : QDialog(parent),
      ui_(std::make_unique<Ui::GraphDialog>()),
      errors_(errors) {
  ui_->setupUi(this);

  DrawGraph();
  connect(ui_->done_button, &QPushButton::clicked, this, &QDialog::accept);
}

GraphDialog::~GraphDialog() {}

void GraphDialog::DrawGraph() {
  auto epoch_count = errors_.size();
  auto epoch_v = InitEpochVector(epoch_count);
  auto error_v = InitErrorsVector(epoch_count);

  ui_->graph_area->addGraph();
  ui_->graph_area->graph(0)->setData(epoch_v, error_v);

  ui_->graph_area->xAxis->setLabel("Epoch");
  ui_->graph_area->yAxis->setLabel("Error");

  ui_->graph_area->xAxis->setRange(0, epoch_count);
  ui_->graph_area->yAxis->setRange(0, 1);

  ui_->graph_area->replot();
}

QVector<double> GraphDialog::InitEpochVector(size_t epoch_count) {
  QVector<double> epoch_v(epoch_count);
  for (size_t i = 0; i != epoch_count; ++i) {
    epoch_v[i] = i;
  }
  return epoch_v;
}

QVector<double> GraphDialog::InitErrorsVector(size_t epoch_count) {
  QVector<double> error_v(epoch_count);
  for (size_t i = 0; i != epoch_count; ++i) {
    error_v[i] = CountErrorsSum(i);
  }
  return error_v;
}

double GraphDialog::CountErrorsSum(int i) {
  auto sum = 0.0;
  for (size_t j = 0, current_errors_sz = errors_[i].size();
       j != current_errors_sz; ++j) {
    sum += errors_[i][j];
  }
  return sum;
}
