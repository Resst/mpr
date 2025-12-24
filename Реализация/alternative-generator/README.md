# alternative-generator
Генератор альтернатив для подачи входных данных в методы принятия решений PYDASS

**Генератор альтернатив** – это инструмент для автоматизированного создания критериев, альтернатив и предпочтений для задач многокритериального выбора. Он позволяет формировать данные для систем принятия решений и многокритериального анализа, в первую очередь в формате XML для PYDASS или реализации метода t-упорядочивания.

## Возможности
- Генерация **абсолютных** и **порядковых** критериев.
- Создание альтернатив с использованием различных распределений (`uniform`, `normal`, `uneven`, `peak`).
- Генерация предпочтений между критериями, включая эквивалентные и неэквивалентные связи.
- Поддержка экспорта в **XML**-формат для дальнейшей обработки в системах принятия решений.
- Возможность контролировать параметры генерации (количество критериев, альтернатив, плотность, распределение и т.д.).
- Генерация связанных и несвязных наборов предпочтений.
- Поддержка генерации критериев с фиксированным или случайным размером групп.

## Установка пакета

Потребуется установленный пакетный менеджер pip. Откройте папку t_ordering в консоли и выполните команду:
` pip install .` Пакет установится вместе с указанными в `setup.py` зависимостями, после чего будет доступн для использования в других проектах.

## Участники
Основным разработчиком проекта является Владимир С. Лебедев, студент ИКНК СПбПУ.

Руководитель и соавтор проекта – Владимир А. Пархоменко, старший преподаватель ИКНК СПбПУ.

## Гарантии
Разработчики не дают никаких гарантий по поводу использования данного программного обеспечения.

## Лицензия
Эта программа открыта для использования и распространяется под лицензией MIT.

# Alternative Generator  

**Alternative Generator** is a tool for automated creation of criteria, alternatives, and preferences for multi-criteria decision-making tasks. It enables the generation of data for decision-making systems and multi-criteria analysis, primarily in XML format for PYDASS or for the implementation of the t-ordering method.  

## Features  
- Generation of **absolute** and **ordinal** criteria.  
- Creation of alternatives using various distributions (`uniform`, `normal`, `uneven`, `peak`).  
- Generation of preferences between criteria, including equivalent and non-equivalent connections.  
- Support for **XML** export for further processing in decision-making systems.  
- Ability to control generation parameters (number of criteria, alternatives, density, distribution, etc.).  
- Generation of connected and disconnected sets of preferences.  
- Support for generating criteria with fixed or random group sizes.  

## Package Installation  
A pre-installed package manager pip is required. Open the `t_ordering` folder in the console and run the command:  
` pip install .` The package will be installed along with the dependencies specified in `setup.py` and will be available for use in other projects.

## Persons
The main contributor of the project is Vladimir S. Lebedev, a student of SPbPU ICSC.

The advisor and minor contributor is Vladimir A. Parkhomenko a seniour lecturer of SPbPU ICSC.

## Warranty
The contributors give no warranty for the using of the software.

## License
This program is open to use anywhere and is licensed under the MIT license.