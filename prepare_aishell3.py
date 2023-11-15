from pathlib import Path
from tqdm import tqdm


arr = [
'SSB1837',
'SSB1365',
'SSB0534',
'SSB1100',
'SSB0737',
'SSB1126',
'SSB0434'
        ]

wav_list = Path("aishell3").rglob("*.wav")


o_file = "wav.tsv"

with open(o_file,"w") as wf:
    line = "client_id\tpath\n"
    wf.write(line)

    for wav in tqdm(wav_list):
        spk = wav.parent.parent.stem
        path = wav.absolute()
        if spk in arr:        
            line = f"{spk}\t{path}\n"
            wf.write(line)
