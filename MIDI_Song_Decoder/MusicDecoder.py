from music21 import converter, note, articulations
from enum import Enum


class SortType(Enum):
    """An Enum represeting the different way ths MusicDecoder's Note's attribute can be organized in a list"""
    TIMING = "timing"
    VOICE = "voice"

class MusicDecoder:
    def __init__(self, file_path: str, sort: SortType = SortType.TIMING):
        self.score = None
        self.notes: list[note.Note] = []

        self._setScore(file_path)
        self._getNotes(sort)

    def _setScore(self, path):
        """Gets and Sets the score from the path provided"""
        try:
            self.score = converter.parse(path.encode('unicode_escape').decode())
        except Exception as e:
            raise RuntimeError("Failed to open file path, check validity of path: " + str(e))

    def _getNotes(self, sort: SortType = SortType.TIMING):
        """Gets all the notes form the score and and puts them into a list. Order of notes in list is determined by sort enum arg"""
        try:
            #Get the Notes into a list
            notes = []
            for n in self.score.recurse().notes:
                if isinstance(n, note.Note):
                    notes.append(n)
            
            #fill in note fingering just in case not labeled for every notes
            notes = self._fill_missing_fingerings(notes)
            
            #sort the Notes as specified
            self.notes = notes
            if not sort == SortType.VOICE:
                self.sortNotes(sort)
            
            #Assign Hand Values for each note
            self._setHands()
            
        except Exception as e:
            raise RuntimeError("Failed to get notes from score: " + str(e))
    
    def _fill_missing_fingerings(self, notes: list[note.Note]):
        """
        Goes through a list of Note objects and assigns missing fingering
        from the previous Note that had one.
        Modifies the notes in place.
        """
        last_fingering = None

        for n in notes:
            # Check if this note has a Fingering
            fingering_obj = None
            for art in n.articulations:
                if isinstance(art, articulations.Fingering):
                    fingering_obj = art
                    break

            if fingering_obj is None:
                # Assign the previous fingering if it exists
                if last_fingering is not None:
                    n.articulations.append(articulations.Fingering(last_fingering))
            else:
                # Update last_fingering
                last_fingering = fingering_obj.fingerNumber
        return notes
    
    def _setHands(self, note_cross:int = 29):
        """Goes through the score and assigns a Hand into the ._editorial property of each note.
            note_cross: integer representing the pitch barrier between L/R hand Assignment. 29 corresponds to Middle C
            hand_right is assigned to this property, with True representing the note should be played by the Right hand"""

        for n in self.notes:

            n._editorial = n._editorial or {}
            n._editorial['Hand'] = "Right" if n.pitch.diatonicNoteNum >= note_cross else "Left"

            #This code checks if there are multiple voices playing at once on the same finger, and assigns the higher one to be on the right hand
            # for tmp in self.notes:
            #     #if the notes are at the same time
            #     if (n.measureNumber, n.offset) == (tmp.measureNumber, n.offset):
            #         self.comparePitches(n,tmp)._editorial.hand_right = False

        #Todo Assign hand if same fingers found at once and at the same time

    def comparePitches(note1:note.Note, note2:note.Note) -> note.Note:
        """Takes in two notes and returns the one with the higher pitch"""
        if note1.pitch.frequency >= note2.pitch.frequency:
            return note1 
        return note2

    def sortNotes(self, sort:SortType = SortType.TIMING):
        """ Sorts the "Notes" attribute based on sorting specified 
            Args: 
                sort : SortType Enum : Method of sorting
        """
        #check to see if score not initalized or no notes found
        if self.notes == None or self.notes == []:
            raise RuntimeError("No notes found in the Score! Score may be empty or initialized incorrectly")
        
        #sort based on specified sorting method
        if sort == SortType.TIMING:
            self.notes.sort(key = lambda n: (n.measureNumber, n.offset))
        elif sort == SortType.VOICE:
            self._getNotes(SortType.VOICE)
 
    def getNotesInfo(self) -> list[dict]:
        """Returns a list of dicts representing each note's info from self.notes."""
        notes = []
        for n in self.notes:
            # find first fingering articulation, if any
            fingering = next((art for art in n.articulations if isinstance(art, articulations.Fingering)), None)
            finger_num = fingering.fingerNumber if fingering is not None else None

            note_info = {
                "Pitch": n.pitch.nameWithOctave,         # str
                "Hand": str(n._editorial["Hand"]),    # force to str just in case
                "Midi": int(n.pitch.midi),               # int
                "Finger": finger_num,                    # str or None
                "Time": (int(n.measureNumber), float(n.offset)),  # tuple of ints/floats
            }
            notes.append(note_info)

        return notes
    
    def __str__(self) -> str:
        notes_info = self.getNotesInfo()
        # Convert list of dicts to a string
        return self.score.metadata.title + "\n" + "\n".join(str(note) for note in notes_info)


                


