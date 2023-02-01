# Multilayer-Perceptron
Проект выполнен совместно с [@chwews](https://github.com/chwews) и [@Sokol264](https://github.com/Sokol264)
## Introduction
Реализация многослойного перцептрона[^1], который может обучаться на EMNIST-датасете. Кроме того, проект предоставляет возможность протестировать перцептрон на заданном тестовом [датасете](https://www.kaggle.com/datasets/crawford/emnist), а также запустить метод кросс-валидации. Для того чтобы каждый раз не переобучать нейросеть (или сохранить подходящий результат обучения) предусмотрена возможность сохранения и загрузки весов. Реализация перцептрона предусмотрена в двух вариантах: матричная и графовая. Варианты реализации перцептрона можно переключать во время работы программы.

Кроме разработки перцептрона и программмного интерфейса, было проведено исследование для сравнения матричной и графовой реализаций, прилагаемое в файле  [research.md](./research.md).

## Example
![](https://github.com/roshik14/Multilayer-Perceptron/blob/main/preview.gif)

## Note
Для запуска требуется Qt 5.15 LTS и выше

## Install
```
make install
make run
```

или

```
make all
```

## Run tests
```
make tests
```

## Rebuild

```
make uninstall
make install
make run
```


[^1]: [Многослойный перцептрон](https://ru.wikipedia.org/wiki/%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%D1%81%D0%BB%D0%BE%D0%B9%D0%BD%D1%8B%D0%B9_%D0%BF%D0%B5%D1%80%D1%86%D0%B5%D0%BF%D1%82%D1%80%D0%BE%D0%BD_%D0%A0%D1%83%D0%BC%D0%B5%D0%BB%D1%8C%D1%85%D0%B0%D1%80%D1%82%D0%B0)
