#include "crossvalidationdialog.h"

#include "fileviewer.h"
#include "ui_crossvalidationdialog.h"

using s21::CrossValidationDialog;

CrossValidationDialog::CrossValidationDialog(QWidget *parent)
    : QDialog(parent), ui_(std::make_unique<Ui::CrossValidationDialog>()) {
  ui_->setupUi(this);
  ui_->apply_button->setEnabled(false);
  connect(ui_->apply_button, &QPushButton::clicked, this, &QDialog::accept);
  connect(ui_->cancel_button, &QPushButton::clicked, this, &QDialog::reject);
  connect(ui_->choose_file_button, &QPushButton::clicked, this,
          &CrossValidationDialog::ChooseFileBtnClicked);
}

CrossValidationDialog::~CrossValidationDialog() {}

void CrossValidationDialog::ChooseFileBtnClicked() {
  auto file_viewer = std::make_unique<FileViewer>();
  auto readed = file_viewer->ReadDataset();
  if (readed) {
      file_name_ = file_viewer->FileName();
      ui_->file_name_label->setText(file_name_.section('/', -1));
      ui_->apply_button->setEnabled(true);
  }
}

QString CrossValidationDialog::file_name() const noexcept { return file_name_; }

int CrossValidationDialog::groups_count() const {
  return ui_->group_count_spinbox->currentText().toInt();
}
