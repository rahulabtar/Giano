import os 
from MusicDecoder import MusicDecoder, SortType

#grabs the file path from the path to this repo on your computer
file = "Baby_Shark__Nursery_Rhyme_Easy_Piano.mxl"
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "MXLFiles", file)

#creates a music decoder object
md = MusicDecoder(file_path, SortType.VOICE)
#when a music decoder object is made, fingerings and hand placements are automically decided if not detected in the score already

#prints out the notes attribute, which is a list of music 21 notes. Kinda hard to understand
print("Notes Object:")
print(str(md.notes) + "\n")

#prints out the data we actually care about for serial transmission as a string in a nicer format
print("MusicDecoder Object as string:")
print(md)
