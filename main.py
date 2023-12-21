from tools.file_searcher import FileSearcher
from tools.media_analyzer import NeuroCLIP
from tools.media_analyzer import NeuroYOLO
from time import time

neuro_clip = NeuroCLIP()
neuro_yolo = NeuroYOLO()
searcher = FileSearcher("images", ("jpeg", "png", "jpg"))

for neuro in (neuro_yolo, neuro_clip):
    all_time = []

    for file in searcher.paths[:100]:
        start = time()
        result_clip = neuro.analyze(file, 0.10)
        spent = time() - start
        all_time.append(spent)

        print(file, result_clip)
        print("Time:", spent)
    print(f"[{neuro.name}] Amount files: {len(all_time)}")
    print(f"[{neuro.name}] Average: {sum(all_time) / len(all_time)} sec.")
    print(f"[{neuro.name}] MAX: {max(all_time)} sec.\tMIN: {min(all_time)} sec.\n\n\n\n")
