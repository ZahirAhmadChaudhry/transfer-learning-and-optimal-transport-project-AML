# utils.py

def get_class_names(y):
    return [f'Class {label}' for label in sorted(set(y))]
