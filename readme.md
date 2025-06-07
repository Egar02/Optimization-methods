# Метод последовательного квадратичного программирования (SQP)

> **Авторы**:

> Томских Егор, Шульков Роман

> студенты группы СМ2-101

## Краткое описание

В данном репозитории представлены теоретические основы метода и представлен 
пример применения метода для тестовой функции с использованием языка python.

---

Метод применяется для нахождения экстремума произвольной функции $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 
при заданном нелинейном ограничении в виде $g(x)=0$, где $g: \mathbb{R}^n \rightarrow \mathbb{R}^m$ 

Метод основан на квадратичной аппроксимации целевой функции и линейной аппроксимации ограничивающей функции.
На каждом шаге алгоритма решается задача поиска минимума ***квадратичной формы*** составленной на основе аппроксимированных функций.
Именно этим фактом обусловлено название метода.

---
---

## Организация проекта

- В папке [src](https://github.com/Egar02/Optimization-methods/tree/main/src) представлена собственная реализация метода SQP на языке python.
- В папке [notebooks](https://github.com/Egar02/Optimization-methods/blob/main/notebooks/) в файле [SQP.ipynb](https://github.com/Egar02/Optimization-methods/blob/main/notebooks/SQP.ipynb) представлено подробное теоретическое описание метода
    и приведена демонстрация метода на примере функции Розенброка с ограничением в виде кругового цилиндра.
- В папке [references](https://github.com/Egar02/Optimization-methods/tree/main/references) представлена [статья](https://github.com/Egar02/Optimization-methods/blob/main/references/Sequential-Quadratic-Programming.pdf) о методе SQP. Также есть ссылка на google-диск со статьями [Ссылка](https://drive.google.com/drive/folders/1_1hbQDE_bEgR-_61vv1KuW26V7OnFEj_?usp=sharing)

---
---