import numpy as np
from typing import Callable

# Вспомогательные функции для программной реализации метода SQP 
# (последовательного квадратичного программирования)
#
# Здесь рассматриваются целевые функции 2 аргументов f: R^2 -> R
# с системой ограничений в виде системы неравенств вида g_i(X) <= (<) 0.
# 
# Соответствующая ограничениям функция задана в виде векторной ФНП g: R^2 -> R^m,
# где m - количество наложенных ограничений

def B(x: float, y:float, Lambda: np.array, 
      Hf: Callable[[float, float], np.array], 
      Hg: Callable[[float, float], np.array]) -> np.array:
    '''
    Вычисляет левый верхний блок матрицы Якоби для градиента функции Лагранжа
    в заданной точке
    
    Аргументы: 

    x, y - координаты текущей точки
    Lambda - вектор размерностью (m, 1) из текущих значений множителей Лагранжа
    Hf - функция, вычисляющая матрицу Гессе целевой функции f в точке
    Hg - функция, вычисляющая матрицу Гессе ограничивающей g функции в точке
    
    Возвращаемое значение:
    
    Матрица размерностью (2, 2)
    
    '''
    return Hf(x, y) + np.squeeze(np.dot(Lambda.T, Hg(x, y)), axis=0)


def JdL(x: float, y: float, Lambda: np.array, 
        Hf: Callable[[float, float], np.array],
        Hg: Callable[[float, float], np.array],
        Jg: Callable[[float, float], np.array]) -> np.array:
    '''
    Вычисляет матрицу Якоби для градиента функции Лагранжа
    в заданной точке
    
    Аргументы: 

    x, y - координаты текущей точки
    Lambda - вектор размерностью (m, 1) из текущих значений множителей Лагранжа
    Hf - функция, вычисляющая матрицу Гессе целевой функции f в точке
    Hg - функция, вычисляющая матрицу Гессе ограничивающей функции g в точке
    Jg - функция, вычисляющая матрицу Якоби ограничивающей функции g в точке
    
    Возвращаемое значение:
    
    Матрица размерностью (2 + m, 2 + m)
    
    '''
    
    B_ = B(x, y, Lambda, Hf, Hg)
    Jg_ = Jg(x, y)
    m = Jg_.shape[0]
    
    JdL_up = np.hstack([B_, Jg_.T])
    JdL_bottom = np.hstack([Jg_, np.zeros((m, m))])
    
    return np.vstack([JdL_up, JdL_bottom])


def SQP_step(x: float, y: float, Lambda: np.array, 
             Hf: Callable[[float, float], np.array],
             Hg: Callable[[float, float], np.array],
             Jg: Callable[[float, float], np.array],
             dL: Callable[[float, float, np.array], np.array]) -> tuple[float, float, np.array]:
    '''
    Делает шаг алгоритма в заданной точке, на выходе получая шаги в сторону
    уменьшения градиента функции Лагранжа по координатам и по множителям Лагранжа
    
    Аргументы: 

    x, y - координаты текущей точки
    Lambda - вектор размерностью (m, 1) из текущих значений множителей Лагранжа
    Hf - функция, вычисляющая матрицу Гессе целевой функции в точке
    Hg - функция, вычисляющая матрицу Гессе ограничивающей функции в точке
    Jg - функция, вычисляющая матрицу Якоби ограничивающей функции в точке
    dL - функция, вычисляющая градиент функции Лагранжа в точке
    
    Возвращаемые значения:
    
    step_x, step_y - шаги по координатам
    step_lambda - вектор шагов по множителям Лагранжа размерностью (m, 1)
    
    '''
    JdL_ = JdL(x, y, Lambda, Hf, Hg, Jg)
    dL_ = dL(x, y, Lambda)
    
    step = np.linalg.solve(JdL_, -1 * dL_)
    
    step_x = step[0].item()
    step_y = step[1].item()
    
    step_Lambda = step[2:, :]
    
    return step_x, step_y, step_Lambda


def SQP_method(x_0: float, y_0:float, Lambda_0: np.array,
               f: Callable[[float, float], np.array], 
               Hf: Callable[[float, float], np.array], 
               Hg: Callable[[float, float], np.array], 
               Jg: Callable[[float, float], np.array], 
               dL: Callable[[float, float, np.array], tuple[float, float, np.array]],
               goal: float = 1e-5,
               lr: float = 0.1,
               show: bool = False) -> tuple[float, float, list[tuple[float, float]]]:
    '''
    Ищет локальный минимум целевой функции при заданных ограничениях
    
    Аргументы: 

    x_0, y_ - координаты начальной точки
    Lambda - вектор размерностью (m, 1) из начальных значений множителей Лагранжа
    f - целевая функция
    Hf - функция, вычисляющая матрицу Гессе целевой функции в точке
    Hg - функция, вычисляющая матрицу Гессе ограничивающей функции в точке
    Jg - функция, вычисляющая матрицу Якоби ограничивающей функции в точке
    dL - функция, вычисляющая градиент функции Лагранжа в точке
    goal - значение нормы градиента функции Лагранжа, при котором поиск останавливается
    show - параметр, отвечающий за то, выводить ли данные на каждом шаге
    
    Возвращаемые значения:
    
    x, y - координаты локального минимума
    log - данные по точкам, пройденным на каждом шаге алгоритма
    
    '''
    
    log = []
    
    x = x_0
    y = y_0
    Lambda = Lambda_0
    
    grad_L = np.sqrt(np.sum(dL(x, y, Lambda) ** 2))
    
    log.append((x, y))
    
    if show:
        print(f"Шаг 0:\t x = {x},\t y = {y},\t f = {f(x, y)},\t |grad L| = {np.sqrt(np.sum(dL(x, y, Lambda) ** 2))}")
    
    i = 0
    
    while abs(grad_L) > goal:
        
        i += 1
        
        step_x, step_y, step_Lambda = SQP_step(x, y, Lambda, Hf, Hg, Jg, dL)
        
        x += lr * step_x
        y += lr * step_y
        Lambda += lr * step_Lambda
        
        grad_L = np.sqrt(np.sum(dL(x, y, Lambda) ** 2))
        
        log.append((x, y))
        
        if show:
            print(f"Шаг {i + 1}:\t x = {x:.5f},\t y = {y:.5f},\t f = {f(x, y):.4f},\t |grad L| = {grad_L:.5e}")
    
    print("\n_________________________________________________________________________________\n")
    print(f"\n// Найден локальный минимум за число шагов, равное {i + 1} //\n\n\
В точке:\t\t({x:.3f}, {y:.3f})\n\
Значение функции:\t\t {f(x, y):.3f}\n")
        
    return x, y, log