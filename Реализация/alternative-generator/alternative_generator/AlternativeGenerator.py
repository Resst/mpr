from typing import List, Tuple
import xml.etree.ElementTree as ET
import re
import numpy as np
import pandas as pd
import random
import json

from alternative_generator import Criterion, Preference


class AlternativeGenerator:
    def __init__(
            self,
            num_criteria: int,
            num_alternatives: int,
            fixed_scales: bool = False,
            abs_ratio: float = 0.8,
            abs_value_range: Tuple[float, float] = (-10.0, 10.0),
            ord_value_range: int = 5,
            abs_distribution: str = "uniform",
            ord_distribution: str = "uniform",
            a_min: int = 0,
            a_max: int = 60,
            b_min: int = 0,
            b_max: int = 60,
            iterations: int = 10,
            fixed_group_sizes: bool = False,
            disconnected_group_components: bool = False,
            avg_group_size: int = 2,
            group_size_std: int = 1,
            fuzzy_numbers: bool = False,  # Новый параметр для включения нечетких чисел
    ):
        """
        Инициализация генератора альтернатив с параметрами генерации.

        :param num_criteria:                    Количество критериев, которые будут сгенерированы.

        :param num_alternatives:                Количество альтернатив, которые будут сгенерированы.

        :param fixed_scales:                    При True фиксирует минимальные и максимальные значения абсолютных шкал
                                                в виде значений, которые передаются в abs_value_range, и фиксирует
                                                количество допустимых значений для порядковых шкал в виде ord_value_range.
                                                При False эти поля работают как в их описании.

        :param abs_ratio:                       Доля абсолютных критериев среди общего числа (от 0 до 1).
                                                Например, при значении 0.8 — 80% критериев будут абсолютными,
                                                а остальные 20% — порядковыми.

        :param abs_value_range:                 Кортеж (min, max), задающий диапазон для генерации
                                                минимальных и максимальных значений абсолютных критериев.

        :param ord_value_range:                 Максимальное количество допустимых значений для порядковых критериев.
                                                Количество допустимых значений для каждого критерия выбирается случайно,
                                                от 1 до указанного максимального значения.

        :param abs_distribution:                Распределение, используемое при генерации значений абсолютных критериев.
                                                Поддерживаются:
                                                - 'uniform' (равномерное распределение)
                                                - 'normal' (нормальное распределение)
                                                - 'uneven' (неравномерное распределение)
                                                - 'peak' (распределение с пиком в центре графика плотности)

        :param ord_distribution:                Распределение для генерации значений порядковых критериев.
                                                Поддерживаются:
                                                - 'uniform' (равномерное распределение)
                                                - 'normal' (нормальное распределение)
                                                - 'uneven' (неравномерное распределение)
                                                - 'peak' (распределение с пиком в центре графика плотности)

        :param fixed_group_sizes:               При True размер групп фиксированный, задается avg_group_size.
                                                При False размер групп случайный со стандартным отклонением group_size_std.

        :param disconnected_group_components:   При True граф связей между группами получается несвязный.
                                                При False граф будет связный.

        :param avg_group_size:                  Средний размер групп эквивалентных критериев. Используется для генерации
                                                предпочтений между критериями.

        :param group_size_std:                  Стандартное отклонение для размера групп эквивалентных критериев.
                                                Позволяет варьировать размер групп в пределах заданного отклонения.
                                                Учитывается только при fixed_group_sizes = False

        :param fuzzy_numbers: Если True, генерирует нечеткие числа (тройки a, b, c)
        """
        if (abs_distribution == 'uneven' or ord_distribution == 'uneven') and num_alternatives % iterations != 0:
            raise ValueError("Количество альтернатив должно делиться на количество итераций.")

        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.iterations = iterations
        self.num_criteria = num_criteria
        self.num_alternatives = num_alternatives
        self.fixed_scales = fixed_scales
        self.abs_ratio = abs_ratio
        self.abs_value_range = abs_value_range
        self.ord_value_range = ord_value_range
        self.abs_distribution = abs_distribution
        self.ord_distribution = ord_distribution
        self.fixed_group_sizes = fixed_group_sizes
        self.disconnected_group_components = disconnected_group_components
        self.avg_group_size = avg_group_size
        self.group_size_std = group_size_std
        self.fuzzy_numbers = fuzzy_numbers  # Сохраняем параметр
        self.criteria = []
        self.alternatives = pd.DataFrame()
        self.preferences = []
        self.criterion_groups = []

    def generate_criteria(self):
        """
        Генерация критериев с учетом распределения типов.
        """
        num_absolute = int(self.num_criteria * self.abs_ratio)
        num_ordinal = self.num_criteria - num_absolute

        # Создаем список критериев с заданным распределением типов
        criteria_types = [True] * num_absolute + [False] * num_ordinal
        random.shuffle(criteria_types)

        for i, absolute in enumerate(criteria_types):
            name = f"criterion{i + 1}"
            if self.fixed_scales:
                maximize = True  # Цель (максимизация или минимизация)
            else:
                maximize = np.random.choice([True, False])  # Цель (максимизация или минимизация)

            if absolute:
                # Генерация абсолютного критерия
                if self.fixed_scales:
                    min_value = round(self.abs_value_range[0], 2)
                    max_value = round(self.abs_value_range[1], 2)
                else:
                    min_value = round(np.random.uniform(*self.abs_value_range), 2)
                    max_value = round(np.random.uniform(min_value, self.abs_value_range[1]), 2)
                criterion = Criterion(
                    name=name,
                    absolute=True,
                    maximize=maximize,
                    min_value=min_value,
                    max_value=max_value,
                )
            else:
                # Генерация порядкового критерия
                if self.fixed_scales:
                    valid_values = [str(x) for x in range(1, self.ord_value_range + 1)]
                else:
                    valid_values = [str(x) for x in range(1, random.randint(2, self.ord_value_range))]
                criterion = Criterion(
                    name=name,
                    absolute=False,
                    maximize=maximize,
                    valid_values=valid_values,
                )

            self.criteria.append(criterion)

    def _generate_fuzzy_triplet(self, base_value: float, min_val: float, max_val: float, 
                               spread_ratio: float = 0.2) -> Tuple[float, float, float]:
        """
        Генерация нечеткого числа на основе базового значения.
        
        :param base_value: Базовое значение (наиболее вероятное)
        :param min_val: Минимальное возможное значение
        :param max_val: Максимальное возможное значение
        :param spread_ratio: Относительный разброс границ (0-1)
        :return: Тройка (a, b, c)
        """
        # Определяем максимальный разброс
        total_range = max_val - min_val
        max_spread = total_range * spread_ratio
        
        # Генерируем разбросы для левой и правой границ
        left_spread = random.uniform(0, min(max_spread, base_value - min_val))
        right_spread = random.uniform(0, min(max_spread, max_val - base_value))
        
        a = round(base_value - left_spread, 2)  # Нижняя граница
        b = round(base_value, 2)                # Наиболее вероятное значение
        c = round(base_value + right_spread, 2) # Верхняя граница
        
        # Гарантируем, что a ≤ b ≤ c
        return (min(a, b), b, max(b, c))

    def _generate_fuzzy_ordinal_triplet(self, base_value: str, valid_values: List[str]) -> Tuple[str, str, str]:
        """
        Генерация нечеткого порядкового числа.
        
        :param base_value: Базовое значение (наиболее вероятное)
        :param valid_values: Допустимые значения
        :return: Тройка (a, b, c)
        """
        try:
            # Пытаемся преобразовать значения в числа для сравнения
            values_as_nums = [float(v) for v in valid_values]
            base_num = float(base_value)
            
            # Находим индекс базового значения
            base_idx = values_as_nums.index(base_num)
            
            # Определяем разброс
            max_left_spread = min(2, base_idx)  # Не более 2 шагов влево
            max_right_spread = min(2, len(values_as_nums) - base_idx - 1)  # Не более 2 шагов вправо
            
            left_spread = random.randint(0, max_left_spread)
            right_spread = random.randint(0, max_right_spread)
            
            lower_idx = base_idx - left_spread
            upper_idx = base_idx + right_spread
            
            return (
                valid_values[lower_idx],
                valid_values[base_idx],
                valid_values[upper_idx]
            )
        except (ValueError, AttributeError):
            # Если значения не преобразуются в числа, используем простой подход
            base_idx = valid_values.index(base_value)
            lower_idx = max(0, base_idx - 1)
            upper_idx = min(len(valid_values) - 1, base_idx + 1)
            return (
                valid_values[lower_idx],
                valid_values[base_idx],
                valid_values[upper_idx]
            )



    def generate_alternatives(self) -> pd.DataFrame:
        """
        Генерация альтернатив на основе критериев.

        :return: DataFrame с альтернативами.
        """
        if not self.criteria:
            raise ValueError("Критерии не сгенерированы. Вызовите generate_criteria() перед генерацией альтернатив.")

        # Создаем массив с типом object для поддержки разных типов данных
        alternatives = np.empty((self.num_alternatives, self.num_criteria), dtype=object)

        for i, criterion in enumerate(self.criteria):
            if criterion.absolute:
                alternatives[:, i] = self._generate_absolute_values(criterion)
            else:
                if (self.fuzzy_numbers):
                    alternatives[:, i] = self._generate_ordinal_values(criterion)
                else:
                    alternatives[:, i] = self._generate_ordinal_values(criterion).astype(str)

        # Создаем DataFrame, сохраняя имена критериев
        self.alternatives = pd.DataFrame(alternatives, columns=[criterion.name for criterion in self.criteria])
        return self.alternatives

    def _generate_absolute_values(self, criterion: Criterion):
        """
        Генерация значений для абсолютных критериев с учетом распределения.
        """
        if self.abs_distribution == "uniform":
            base_values = np.round(
                np.random.uniform(
                    criterion.min_value,
                    criterion.max_value,
                    self.num_alternatives,
                ), 2
            )
        elif self.abs_distribution == "normal":
            mean = (criterion.max_value + criterion.min_value) / 2
            std = (criterion.max_value - criterion.min_value) / 6
            base_values = np.round(
                np.clip(
                    np.random.normal(mean, std, self.num_alternatives),
                    criterion.min_value,
                    criterion.max_value,
                ), 2
            )
        elif self.abs_distribution == 'peak':
            a, b = 50, 50
            data = np.random.beta(a, b, size=self.num_alternatives)
            base_values = data * (criterion.max_value - criterion.min_value) + criterion.min_value
            base_values = np.round(
                np.clip(base_values, criterion.min_value, criterion.max_value), 2
            )
        elif self.abs_distribution == "uneven":
            all_values = np.empty((0,))
            for i in range(self.iterations):
                peak = np.random.beta(a=np.random.uniform(self.a_min, self.a_max),
                                      b=np.random.uniform(self.b_min, self.b_max),
                                      size=int(self.num_alternatives / self.iterations)) * (
                                   criterion.max_value - criterion.min_value) + criterion.min_value
                all_values = np.hstack((all_values, peak))
            base_values = np.round(np.clip(all_values, criterion.min_value, criterion.max_value), 2)
            np.random.shuffle(base_values)
        
        # Если режим нечетких чисел, преобразуем каждое значение в тройку
        if self.fuzzy_numbers:
            result = np.empty(self.num_alternatives, dtype=object)
            for i, base_val in enumerate(base_values):
                result[i] = self._generate_fuzzy_triplet(
                    base_val, 
                    criterion.min_value, 
                    criterion.max_value
                )
            return result
        
        return base_values


    def _generate_ordinal_values(self, criterion: Criterion):
        """
        Генерация значений для порядковых критериев с учетом распределения.
        """
        if self.ord_distribution == "uniform":
            base_values = np.random.choice(criterion.valid_values, self.num_alternatives)
        elif self.ord_distribution == "normal":
            value_indices = np.random.normal(
                loc=len(criterion.valid_values) / 2,
                scale=len(criterion.valid_values) / 6,
                size=self.num_alternatives,
            )
            # Корректируем индексы, чтобы избежать выхода за пределы
            value_indices = np.clip(value_indices, 0, len(criterion.valid_values) - 1).astype(int)
            
            # Генерация альтернатив
            base_values = np.array(criterion.valid_values)[value_indices]
        elif self.ord_distribution == 'peak':
            a, b = 50, 50  # Чем больше, тем уже пик
            data = np.random.beta(a, b, size=self.num_alternatives)  # Генерация beta-распределения
            # Масштабирование индексов под длину valid_values
            scaled_indices = np.clip(np.round(data * (len(criterion.valid_values) - 1)), 0,
                                     len(criterion.valid_values) - 1).astype(int)
            # Преобразование индексов в значения valid_values
            base_values = np.array(criterion.valid_values)[scaled_indices]
        elif self.ord_distribution == "uneven":
            all_values = np.empty((0,))
            valid_indices = np.arange(len(criterion.valid_values))  # Индексы допустимых значений
            for i in range(self.iterations):
                # Генерация весов с использованием beta-распределения
                weights = np.random.beta(a=np.random.uniform(self.a_min, self.a_max),
                                         b=np.random.uniform(self.b_min, self.b_max),
                                         size=int(self.num_alternatives / self.iterations))
                # Масштабирование индексов с округлением
                peak_indices = np.clip(np.round(weights * (len(valid_indices) - 1)), 0, 
                                       len(valid_indices) - 1).astype(int)
                peak = np.array(criterion.valid_values)[peak_indices]  # Преобразование индексов в значения
                all_values = np.hstack((all_values, peak))
            np.random.shuffle(all_values)  # Перемешивание значений
            base_values = all_values
        
        # Если режим нечетких чисел, преобразуем каждое значение в тройку
        if self.fuzzy_numbers:
            result = np.empty(self.num_alternatives, dtype=object)
            for i, base_val in enumerate(base_values):
                result[i] = self._generate_fuzzy_ordinal_triplet(
                    base_val,
                    criterion.valid_values
                )
            return result
        
        return base_values

    def generate_preferences(self) -> List[Preference]:
        """
        Генерация предпочтений для сгенерированных ранее критериев с учетом параметров генератора.
        """
        if not self.criteria:
            raise ValueError("Критерии не сгенерированы. Вызовите generate_criteria() перед генерацией предпочтений.")

        # Список индексов всех критериев
        remaining_criteria = self.criteria.copy()
        random.shuffle(remaining_criteria)

        # Генерация групп эквивалентных критериев
        groups = []
        while remaining_criteria:
            if self.fixed_group_sizes:
                group_size = max(1, int(self.avg_group_size))  # Фиксированный размер группы
            else:
                group_size = int(
                    max(
                        1,
                        round(np.random.normal(self.avg_group_size, self.group_size_std)),
                    )
                )
            group = remaining_criteria[:group_size]
            groups.append(group)
            remaining_criteria = remaining_criteria[group_size:]

        # Сохраняем порядок групп
        random.shuffle(groups)
        self.criterion_groups = groups

        # Генерация эквивалентных предпочтений внутри групп
        for group in groups:
            for i in range(1, len(group)):
                self.preferences.append(Preference(group[0], group[i], equivalent=True))

        # Генерация предпочтений между группами
        for i in range(len(groups) - 1):
            self.preferences.append(
                Preference(groups[i][0], groups[i + 1][0], equivalent=False)
            )

        # Удаление связей между группами для создания несвязного графа
        if self.disconnected_group_components:
            # Количество компонент: критерии^0.4, округляем до целого
            num_components = max(2, round(len(self.criteria) ** 0.4))
            edges_to_remove = num_components - 1

            # Удаляем случайные предпочтения между группами
            if len(groups) >= num_components:
                removable_edges = [p for p in self.preferences if not p.equivalent]
                random.shuffle(removable_edges)
                for i in range(edges_to_remove):
                    self.preferences.remove(removable_edges[i])

        return self.preferences

    def export_to_xml(self, filename: str = "alternatives.xml"):
        """
        Экспорт данных в XML формате.

        :param filename: Имя файла, в который нужно записать данные
        """
        # Проверка наличия данных
        if not self.criteria or not self.preferences:
            raise ValueError("Критерии или предпочтения не сгенерированы.")

        # Создаем корневой элемент XML
        root = ET.Element("DecisionModel")

        # Создаем секцию критериев
        criteria_section = ET.SubElement(root, "Criteria")
        for criterion in self.criteria:
            crit_element = ET.SubElement(criteria_section, "Criterion", {
                "name": criterion.name,
                "type": "absolute" if criterion.is_absolute() else "ordinal",
                "goal": "maximize" if criterion.is_maximize() else "minimize"
            })
            if criterion.is_absolute():
                ET.SubElement(crit_element, "MinValue").text = str(criterion.min_value)
                ET.SubElement(crit_element, "MaxValue").text = str(criterion.max_value)
            else:
                values_element = ET.SubElement(crit_element, "ValidValues")
                for value in criterion.valid_values:
                    ET.SubElement(values_element, "Value").text = value

        # Создаем секцию альтернатив
        alternatives_section = ET.SubElement(root, "Alternatives")
        alternatives = self.alternatives

        for i in range(len(alternatives)):
            alt_element = ET.SubElement(alternatives_section, "Alternative", {
                "id": str(i + 1)
            })

            # Для каждой альтернативы и критерия записываем значение в XML
            for criterion in self.criteria:
                value = alternatives.at[i, criterion.name] if criterion.name in alternatives.columns else ""
                ET.SubElement(alt_element, criterion.name).text = str(value)

        # Создаем секцию предпочтений
        preferences_section = ET.SubElement(root, "Preferences")
        for pref in self.preferences:
            ET.SubElement(preferences_section, "Preference", {
                "criterion1": pref.criterion1.name,
                "criterion2": pref.criterion2.name,
                "equivalent": str(pref.equivalent).lower()
            })

        # Преобразуем дерево в строку и сохраняем в файл
        tree = ET.ElementTree(root)
        with open(filename, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

        print(f"XML файл сохранен как {filename}")

    def export_to_xml_pydass(self, filename: str = "pydass_alternatives.xml"):
        """
        Экспорт данных в XML формате для pydass.

        :param filename: Имя файла, в который нужно записать данные
        """
        import xml.etree.ElementTree as ET

        # Проверка наличия данных
        if not self.criteria or not self.preferences:
            raise ValueError("Критерии или предпочтения не сгенерированы.")

        # Создаем корневой элемент XML
        root = ET.Element("decision")

        # Описание задачи
        problem_section = ET.SubElement(root, "problem")
        ET.SubElement(problem_section, "pname").text = "Выбор наилучшей альтернативы"
        ET.SubElement(problem_section, "pdescr").text = "Автоматически сгенерированные данные"

        # Секция критериев
        criteria_section = ET.SubElement(root, "criterionList")
        for criterion in self.criteria:
            ET.SubElement(criteria_section, "criterion", {
                "cname": criterion.name
            }).text = criterion.name

        # Секция важности
        importance_section = ET.SubElement(root, "importance", {
            "active": "order" if not criterion.absolute else "interval"
        })
        order_section = ET.SubElement(importance_section, "order")
        positions_section = ET.SubElement(order_section, "positions")

        for i, group in enumerate(self.criterion_groups):
            for criterion in group:
                number = int(re.search(r'\d+$', criterion.name).group()) - 1
                ET.SubElement(positions_section, "pos").text = str(number)

        rel_importance_section = ET.SubElement(order_section, "relativeImportance")

        # Обход групп эквивалентных критериев в порядке важности
        for i, group in enumerate(self.criterion_groups):
            first = True
            for criterion in group:
                if first:
                    ET.SubElement(rel_importance_section, "ri").text = "less"
                    first = False
                else:
                    ET.SubElement(rel_importance_section, "ri").text = "equal"

        importance_coefs = ET.SubElement(importance_section, "importanceCoefs")

        first_elem = True
        # Обход групп эквивалентных критериев в порядке важности
        for i, group in enumerate(self.criterion_groups):
            first = True
            for criterion in group:
                if first:
                    if first_elem:
                        first = False
                        first_elem = False
                        continue
                    ET.SubElement(importance_coefs, "ic").text = "2"
                    first = False
                else:
                    ET.SubElement(importance_coefs, "ic").text = "1"

        # Секция шкал (только для порядковых критериев)
        scale_section = ET.SubElement(root, "scale", {"active": "plain"})
        ET.SubElement(scale_section, "gradeCount").text = str(len(self.criteria))
        grade_values_section = ET.SubElement(scale_section, "gradeValues")

        if self.criteria[0].valid_values:
            for value in criterion.valid_values:
                ET.SubElement(grade_values_section, "gv").text = str(value)

        # Секция альтернатив
        alternatives_section = ET.SubElement(root, "variantList")
        for i, row in self.alternatives.iterrows():
            alt_element = ET.SubElement(alternatives_section, "variant", {
                "vname": f"n{i + 1}",
                "nondominated": "yes"
            })
            scores_section = ET.SubElement(alt_element, "scores")
            for value in row:
                ET.SubElement(scores_section, "sc").text = str(value)
            ET.SubElement(alt_element, "linkedTo").text = f"n{i + 1}"

        # Преобразуем дерево в строку и сохраняем в файл
        tree = ET.ElementTree(root)
        with open(filename, "wb") as f:
            f.write(b'\xff\xfe')  # BOM для UTF-16 little-endian
            tree.write(f, encoding="UTF-16", xml_declaration=True)

        print(f"XML файл для PYDASS сохранен как {filename}")

    def print_equivalent_groups_by_components(self):
        """
        Вывод групп эквивалентных критериев с разделением по компонентам связности
        на основе предпочтений.
        """
        components = []  # Список компонентов связности

        # Распределение групп по компонентам связности
        for preference in self.preferences:
            if not preference.equivalent:
                # Определяем группы, которым принадлежат критерии
                group1 = next(group for group in self.criterion_groups if preference.criterion1 in group)
                group2 = next(group for group in self.criterion_groups if preference.criterion2 in group)

                # Проверяем, есть ли группы уже в компонентах
                component1 = next((comp for comp in components if group1 in comp), None)
                component2 = next((comp for comp in components if group2 in comp), None)

                if component1 and component2:
                    # Если обе группы уже в компонентах, объединяем компоненты
                    if component1 != component2:
                        component1.extend(component2)
                        components.remove(component2)
                elif component1:
                    # Если первая группа уже в компоненте, добавляем вторую
                    component1.append(group2)
                elif component2:
                    # Если вторая группа уже в компоненте, добавляем первую
                    component2.append(group1)
                else:
                    # Если ни одна группа не в компоненте, создаём новый компонент
                    components.append([group1, group2])

        # Добавляем группы, которые не попали в компоненты
        for group in self.criterion_groups:
            if not any(group in component for component in components):
                components.append([group])

        # Вывод компонент связности
        component_counter = 1
        for component in components:
            print(f"\nКомпонент связности {component_counter}:")
            for i, group in enumerate(component, start=1):
                group_names = [criterion.name for criterion in group]
                print(f"Группа {i} :", group_names)
            component_counter += 1
                
    def export_to_json(self, filename: str = "alternatives.json"):
        """
        Экспорт данных в JSON формате для нечетких чисел.
        
        Формат:
        {
          "alternatives": ["A1", "A2", ...],
          "criteria": [
            {"name": "criterion1", "type": "positive/negative", "weight": 1},
            ...
          ],
          "dms": [
            {
              "scores": [
                [[a,b,c], [a,b,c], ...],  # для альтернативы 1
                [[a,b,c], [a,b,c], ...],  # для альтернативы 2
                ...
              ]
            }
          ]
        }
        """
        if not self.criteria or self.alternatives.empty:
            raise ValueError("Критерии или альтернативы не сгенерированы.")
        
        # Проверяем, что включен режим нечетких чисел
        if not self.fuzzy_numbers:
            print("Предупреждение: экспорт в JSON предназначен для нечетких чисел. "
                  "Режим fuzzy_numbers=False, данные будут преобразованы в тройки.")
        
        # Формируем список имен альтернатив
        alternatives_names = [f"A{i+1}" for i in range(self.num_alternatives)]
        
        # Формируем список критериев
        criteria_list = []
        for criterion in self.criteria:
            # Определяем тип критерия (positive для максимизации, negative для минимизации)
            criterion_type = "positive" if criterion.maximize else "negative"
            criteria_list.append({
                "name": criterion.name,
                "type": criterion_type,
                "weight": 1  # Все веса по умолчанию 1
            })
        
        # Формируем матрицу оценок
        scores = []
        
        for alt_idx in range(self.num_alternatives):
            alt_scores = []
            
            for criterion in self.criteria:
                value = self.alternatives.at[alt_idx, criterion.name]
                
                # Преобразуем значение в тройку
                if self.fuzzy_numbers and isinstance(value, tuple):
                    # Если уже тройка
                    if len(value) == 3:
                        alt_scores.append(list(value))
                    else:
                        # Если что-то не так, создаем тройку из одинаковых значений
                        alt_scores.append([value, value, value])
                else:
                    # Для обычных чисел создаем тройку из одинаковых значений
                    if isinstance(value, (int, float)):
                        alt_scores.append([float(value), float(value), float(value)])
                    else:
                        # Для строковых значений пытаемся преобразовать в число
                        try:
                            num_value = float(value)
                            alt_scores.append([num_value, num_value, num_value])
                        except:
                            # Если не удается, используем 0,0,0
                            alt_scores.append([0.0, 0.0, 0.0])
            
            scores.append(alt_scores)
        
        # Формируем финальную структуру данных
        data = {
            "alternatives": alternatives_names,
            "criteria": criteria_list,
            "dms": [
                {
                    "scores": scores
                }
            ]
        }
        
        # Сохраняем в файл
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"JSON файл сохранен как {filename}")
        
        return data