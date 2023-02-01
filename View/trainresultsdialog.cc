#include "trainresultsdialog.h"

#include "graphdialog.h"
#include "ui_trainresultsdialog.h"

using s21::TrainResultsDialog;

TrainResultsDialog::TrainResultsDialog(QWidget *parent)
    : QDialog(parent), ui_(std::make_unique<Ui::TrainResultsDialog>()) {
  ui_->setupUi(this);
  connect(ui_->quit_button, &QPushButton::clicked, this, &QDialog::accept);
  connect(ui_->graph_button, &QPushButton::clicked, this,
          &TrainResultsDialog::ShowGraph);
}

TrainResultsDialog::~TrainResultsDialog() {}

void TrainResultsDialog::SetRows(int rows) { ui_->table->setRowCount(rows); }

void TrainResultsDialog::SetErrors(
    std::vector<std::vector<double>> &&errors) noexcept {
  errors_ = std::move(errors);
}

void TrainResultsDialog::CreateTable() {
  InitColumnsHeader();
  for (auto i = 0, rows = ui_->table->rowCount(); i != rows; ++i) {
    auto current_epoch = new QTableWidgetItem(QString::number(i + 1));
    ui_->table->setItem(i, 0, current_epoch);
    SetErrorsToRow(i);
  }
}

void TrainResultsDialog::InitColumnsHeader() {
  auto cols = errors_[0].size();
  ui_->table->setColumnCount(cols);
  for (size_t i = 0; i != cols; ++i) {
    ui_->table->setHorizontalHeaderItem(
        i + 1, new QTableWidgetItem("Ошибка " + QString::number(i + 1)));
  }
}

void TrainResultsDialog::SetErrorsToRow(int row) {
  for (size_t col = 0, sz = errors_[row].size(); col != sz; ++col) {
    ui_->table->setItem(
        row, col + 1, new QTableWidgetItem(QString::number(errors_[row][col])));
  }
}

void TrainResultsDialog::ShowGraph() {
  auto graph_dialog = std::make_unique<GraphDialog>(errors_);
  graph_dialog->exec();
}
