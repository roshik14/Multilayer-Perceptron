#include "traindialog.h"

#include "fileviewer.h"
#include "ui_traindialog.h"
#include "useful_data.h"

using s21::TrainDialog;

TrainDialog::TrainDialog(QWidget* parent)
    : QDialog(parent), ui_(std::make_unique<Ui::TrainDialog>()) {
  ui_->setupUi(this);
  connect(ui_->button_box, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(ui_->button_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  connect(GetApplyButton(), &QAbstractButton::clicked, this,
          &TrainDialog::ApplySlot);
  connect(ui_->choose_file_button, &QAbstractButton::clicked, this,
          &TrainDialog::ChooseFileSlot);
  GetApplyButton()->setEnabled(false);
}

TrainDialog::~TrainDialog() {}

QAbstractButton* TrainDialog::GetApplyButton() {
  auto buttons = ui_->button_box->buttons();
  for (int i = 0, sz = buttons.size(); i != sz; ++i) {
    if (ui_->button_box->buttonRole(buttons[i]) == QDialogButtonBox::ApplyRole)
      return buttons[i];
  }
  return nullptr;
}

s21::TrainData TrainDialog::data() const {
  return TrainData{
      file_name_.toStdString(), ui_->epoch_spinbox->value(),
      ui_->neuron_spinbox->value(), ui_->hidden_layers_spinbox->value(),
      ui_->matrix_radio_button->isChecked() ? PerceptronType::kMatrix
                                            : PerceptronType::kGraph};
}

void TrainDialog::ApplySlot() { emit ui_->button_box->accepted(); }

void TrainDialog::ChooseFileSlot() {
  auto reader = std::make_unique<FileViewer>();
  auto opened = reader->ReadDataset();
  if (opened) {
    file_name_ = reader->FileName();
    ui_->file_name_label->setText(file_name_.section('/', -1));
    GetApplyButton()->setEnabled(true);
  }
}
