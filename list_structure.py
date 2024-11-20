import os

def list_structure(base_path, level=3):
    for root, dirs, files in os.walk(base_path):
        depth = root[len(base_path):].count(os.sep)
        if depth < level:
            print(" " * (depth * 4) + os.path.basename(root))
            if depth < level - 1:
                for f in files:
                    print(" " * ((depth + 1) * 4) + f)

list_structure("D:/UDEMY/sleep-disorder-prediction-main/sleep-disorder-prediction-main/Sleep (2) (1)/Sleep")