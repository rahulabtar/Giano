from music21 import converter, note, articulations
from enum import Enum

class SortType(Enum):
    TIMING = "timing"
    VOICE = "voice"

class MusicDecoder:
    def __init__(self, file_path: str, sort: SortType = SortType.TIMING):
        self.score = None
        self.notes: list[note.Note] = []

        self._setScore(file_path)
        self._getNotes(sort)

    def _setScore(self, path):
        try:
            self.score = converter.parse(path.encode('unicode_escape').decode())
        except Exception as e:
            raise RuntimeError("Failed to open file path, check validity of path: " + str(e))

    def _getNotes(self, sort: SortType):
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
            
        except Exception as e:
            raise RuntimeError("Failed to get notes from score: " + str(e))
    
    def _fill_missing_fingerings(self, notes: list[note.Note]) -> None:
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
    
    def _setHands(self):
        for n in self.notes:
            n._editorial.hand = "left"

        #Todo Assign hand if same fingers found at once and at the same time

    def sortNotes(self, sort:SortType):
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
 
    def getNotesInfo(self):
        return
        notes = [] 
        for n in self.notes:
            note_info = {}

            # Print pitch
            note_info["Pitch"] = n.pitch.nameWithOctave
            note_info["Measure"] = n.measureNumber
            note_info["Onset"] = n.offset

            # Check for fingering (in articulations or notations)
            for art in n.articulations:
                if isinstance(art, articulations.Fingering):
                    fingering = art.fingerNumber

            if fingering:
                note_info["Fingering"] = fingering

            notes.append(note_info)
        return notes
                


