# fuzzy_logic_utils.py

def fuzz_energy(val):
    if val < 35:
        return {'low': 1.0, 'medium': 0.0, 'high': 0.0}
    elif 35 <= val <= 45:
        return {'low': (45 - val) / 10, 'medium': (val - 35) / 10, 'high': 0.0}
    elif 45 < val <= 55:
        return {'low': 0.0, 'medium': (55 - val) / 10, 'high': (val - 45) / 10}
    else:
        return {'low': 0.0, 'medium': 0.0, 'high': 1.0}

def fuzz_income(val):
    if val < 10:
        return {'low': 1.0, 'medium': 0.0, 'high': 0.0}
    elif 10 <= val <= 20:
        return {'low': (20 - val) / 10, 'medium': (val - 10) / 10, 'high': 0.0}
    elif 20 < val <= 30:
        return {'low': 0.0, 'medium': (30 - val) / 10, 'high': (val - 20) / 10}
    else:
        return {'low': 0.0, 'medium': 0.0, 'high': 1.0}
