#ifndef SRC_DRAWER_H_
#define SRC_DRAWER_H_

#include <QGraphicsView>
#include <QMouseEvent>
#include <QPointer>

namespace s21 {
class Drawer;
}

class s21::Drawer final : public QGraphicsView {
  Q_OBJECT
 public:
  explicit Drawer(QWidget* parent = nullptr);
  void Clear();
  void SetImage(QImage&& image) noexcept;
  QImage image();

 signals:
  void Release();

 private:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  bool IsLeftButton(QMouseEvent* event) const;

 private:
  const int width_;
  const int height_;
  QPointer<QGraphicsScene> scene_;
  QPoint last_pos_;
  const int kPointSize = 50;
};

#endif  // SRC_DRAWER_H_
