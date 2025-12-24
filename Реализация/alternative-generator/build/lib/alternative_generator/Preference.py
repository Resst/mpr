from alternative_generator import Criterion

class Preference:
    def __init__(self, criterion1: Criterion, criterion2: Criterion, equivalent: bool):
        """
        Инициализирует объект Предпочтения между двумя критериями.

        Параметры:
        - criterion1: Первый критерий.
        - criterion2: Второй критерий.
        - equivalent: True, если критерии эквивалентны, False, если criterion1 важнее criterion2.
        """
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.equivalent = equivalent

    def __str__(self):
        """
        Возвращает строковое представление объекта Preference.
        """
        if self.equivalent:
            relation = "="
        else:
            relation = ">"
        return f"{self.criterion1.name} {relation} {self.criterion2.name}"