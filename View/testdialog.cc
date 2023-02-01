#include "testdialog.h"

#include "fileviewer.h"
#include "ui_testdialog.h"

using s21::TestDialog;

TestDialog::TestDialog(QWidget *parent)
    : QDialog(parent), ui_(std::make_unique<Ui::TestDialog>()) {
  ui_->setupUi(this);
  connect(ui_->value_slider, &QSlider::valueChanged, this,
          &TestDialog::SpinboxSetValueSlot);
  connect(ui_->choose_file_button, &QAbstractButton::clicked, this,
          &TestDialog::ChooseFileSlot);
  ui_->button_box->button(QDialogButtonBox::Ok)->setDisabled(true);
}

TestDialog::~TestDialog() {}

void TestDialog::SpinboxSetValueSlot(int value) {
  ui_->value_spinbox->setValue(value * 1.0 / 100);
}

void TestDialog::ChooseFileSlot() {
  auto file_viewer = std::make_unique<FileViewer>();
  auto readed = file_viewer->ReadDataset();
  if (readed) {
    data_.file_name = file_viewer->FileName();
    auto index = data_.file_name.lastIndexOf('/') + 1;
    ui_->file_name_label->setText(data_.file_name.sliced(index));
    ui_->button_box->button(QDialogButtonBox::Ok)->setEnabled(true);
  }
}

TestDialog::TestData TestDialog::data() {
  data_.percent = ui_->value_spinbox->value();
  return data_;
}
