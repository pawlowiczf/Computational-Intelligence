.PHONY: rastrigin-optimization binary-classification

rastrigin-optimization:
	python -m clonalg.experiments.rastrigin.main

binary-classification:
	python -m clonalg.experiments.binary_classification.main