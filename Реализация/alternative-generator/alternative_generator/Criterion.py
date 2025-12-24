from typing import List

class Criterion:
    def __init__(
        self,
        name: str,
        absolute: bool,
        maximize: bool,
        valid_values: List[str] = None,
        min_value: float = None,
        max_value: float = None,
    ):
        """
        Инициализирует объект Критерия.

        Параметры:
        - name: Имя критерия.
        - absolute: True, если критерий количественный (абсолютный), False, если порядковый.
        - maximize: True, если цель — максимизация критерия, False — минимизация.
        - valid_values: Для порядковых критериев — упорядоченный список допустимых строковых значений.
        - min_value: Для абсолютных критериев — минимальное допустимое значение.
        - max_value: Для абсолютных критериев — максимальное допустимое значение.
        """
        self.name = name
        self.absolute = absolute
        self.maximize = maximize
        self.weight = None  # Вес критерия, заполняется позже

        if self.is_ordinal():
            if valid_values is None or not isinstance(valid_values, list):
                raise ValueError(
                    f"Необходимо предоставить допустимые значения для порядкового критерия '{self.name}'"
                )
            self.valid_values = valid_values
            self.min_value = 0  # Минимальное значение после кодирования
            self.max_value = len(valid_values) - 1  # Максимальное значение после кодирования
        elif self.is_absolute():
            if min_value is None or max_value is None:
                raise ValueError(
                    f"Необходимо предоставить min и max значения для абсолютного критерия '{self.name}'"
                )
            self.min_value = min_value
            self.max_value = max_value
        else:
            raise ValueError("Критерий должен быть либо абсолютным, либо порядковым")

    def is_absolute(self):
        return self.absolute

    def is_ordinal(self):
        return not self.absolute

    def is_maximize(self):
        return self.maximize

    def is_minimize(self):
        return not self.maximize

    def is_value_permissible(self, value):
        if self.is_absolute():
            return self.min_value <= value <= self.max_value
        elif self.is_ordinal():
            return value in self.valid_values
        else:
            return False

    def __str__(self):
        """
        Возвращает строковое представление объекта Criterion.
        """
        if self.is_absolute():
            type_str = "Абсолютный"
            values_str = f"Минимальное значение: {self.min_value}, Максимальное значение: {self.max_value}"
        else:
            type_str = "Порядковый"
            values_str = f"Допустимые значения: {self.valid_values}"
        goal_str = "Максимизация" if self.is_maximize() else "Минимизация"
        return (f"Критерий '{self.name}':\n"
                f"  Тип: {type_str}\n"
                f"  Цель: {goal_str}\n"
                f"  {values_str}\n")