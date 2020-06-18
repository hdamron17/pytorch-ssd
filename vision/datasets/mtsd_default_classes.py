## Taken from https://github.com/jacky841102/HD_map_updating/blob/master/sign_detector/preprocess_script/trafficsigns_to_detectron_6_class.py

#                 0      1            2         3: black and white    4:red circle           5
CATEGORIES =  ['stop', 'yield', 'do_not_enter', 'other_regulatory', 'other_prohibitory', 'warning']

def convert_label(label):
    class_id = -1  # empty value
    if 'regulatory--stop--g1' in label:
        class_id = 0
    elif 'regulatory--yield--g1' in label:
        class_id = 1
    elif 'regulatory--no-entry--g1' in label:
        class_id = 2
    elif 'regulatory--no-parking--g2' in label:
        class_id = 4
    elif 'regulatory--maximum-speed-limit' in label:
        if 'led' in label:
            class_id = -1
        elif 'g1' in label:
            class_id = -1
        elif '90' in label:
            class_id = -1
        elif '100' in label:
            class_id = -1
        elif '110' in label:
            class_id = -1
        elif '120' in label:
            class_id = -1
        elif '130' in label:
            class_id = -1
        else:
            class_id = 3
    elif 'regulatory--turn-right--g3' in label:
        class_id = 3
    elif 'regulatory--go-straight--g3' in label:
        class_id = 3
    elif 'regulatory--turn-left--g2' in label:
        class_id = 3
    elif 'regulatory--no-right-turn--g1' in label:
        class_id = 4
    elif 'regulatory--no-straight-through--g1' in label:
        class_id = 4
    elif 'regulatory--no-left-turn--g1' in label:
        class_id = 4
    elif label.startswith('warning'):
        if 'warning--railroad-crossing--g4' in label:
            class_id = -1
        else:
            class_id = 5
    elif 'regulatory--bicycles-only--g3' in label:
        class_id = -1
    else:
        class_id = -1

    return class_id + 1 if class_id >= 0 else None
