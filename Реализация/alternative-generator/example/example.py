from alternative_generator import AlternativeGenerator


if __name__ == "__main__":
    # Пример генерации данных для ТВК из PYDASS
    generator = AlternativeGenerator(
        num_criteria=9,
        num_alternatives=10,
        fixed_scales=True, # все шкалы одинаковые
        abs_ratio=0, # только порядковые шкалы у критериев критерии
        ord_value_range=6,
        ord_distribution="uneven",
        fixed_group_sizes=True,
        disconnected_group_components=False, # ТВК не поддерживает несвязные компоненты, информация о предпочтениях должна быть полной
        avg_group_size=3,
        group_size_std=1,
    )

    generator.generate_criteria()
    alternatives = generator.generate_alternatives()
    preferences = generator.generate_preferences()

    print("\nСгенерированные критерии:")
    for criterion in generator.criteria:
        print(vars(criterion))

    print("\nСгенерированные альтернативы:\n", alternatives)

    print("\nСгенерированные предпочтения:")
    for preference in generator.preferences:
        print(preference)

    print("\nСгенерированные группы эквивалентных критериев в порядке уменьшения их важности внутри компонент связности:")
    generator.print_equivalent_groups_by_components()

    print("\n")
    generator.export_to_xml_pydass()