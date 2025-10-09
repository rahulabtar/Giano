import cv2 as cv
import numpy as np
import pdf2image
import os
from typing import List, Tuple, Union
from src.core.constants import PAGE_DPI, ASSETS_DIR

class ArucoMarkerGenerator:
    """A system for generating ArUco markers and marker sheets."""
    
    def __init__(self, dictionary_type=cv.aruco.DICT_6X6_250):
        """
        Initialize the ArUco marker generator.
        
        Args:
            dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
        """
        self.dictionary = cv.aruco.getPredefinedDictionary(dictionary_type)
    
    def generate_marker(self, marker_id: int, size: int = 200) -> np.ndarray:
        """
        Generate an ArUco marker.
        
        Args:
            marker_id: ID of the marker (0-249 for DICT_6X6_250)
            size: Size of the marker in pixels
            
        Returns:
            numpy array containing the marker image
        """
        marker_img = cv.aruco.generateImageMarker(self.dictionary, marker_id, size)
        return marker_img
    
    def save_marker(self, marker_id: int, filename: str, size: int = 200):
        """
        Generate and save an ArUco marker to file.
        
        Args:
            marker_id: ID of the marker
            filename: Output filename (with .png extension)
            size: Size of the marker in pixels
        """
        marker_img = self.generate_marker(marker_id, size)
        cv.imwrite(filename, marker_img)
        print(f"Marker {marker_id} saved as {filename}")
    
    def create_marker_sheet(
        self,
        input_sheet: np.ndarray = None,   # If None, create a new blank sheet
        paper_size_inches: Tuple[float, float] = (8.5, 11.0),  
        dpi: int = 300,
        marker_size_inches: float = 1.0,   
        marker_locations: List[Tuple[float, float]] = None,  # (x, y) offsets in inches from top-left
        marker_ids: List[int] = None,  # Custom marker IDs
        filename: str = "aruco_marker_sheet.png"
    ) -> str:
        """
        Create a sheet with markers at specified locations or corners.
        
        Args:
            input_sheet: Existing image to overlay markers on (optional)
            paper_size_inches: Paper dimensions in inches (width, height)
            dpi: Resolution in dots per inch (300 is good for printing)
            marker_size_inches: Size of each marker in inches
            marker_locations: List of (x, y) positions in inches from top-left corner.
                            If None, defaults to 4 corner positions with 0.5" offset
            marker_ids: List of marker IDs to use. If None, uses sequential IDs starting from 0
            filename: Output filename
            
        Returns:
            filename: The name of the file the image was saved to
        """
        
        # Set default corner locations if none provided
        if marker_locations is None:
            corner_offset = 0.5  # Default 0.5" from edges
            marker_locations = [
                (corner_offset, corner_offset),  # Top-left
                (paper_size_inches[0] - corner_offset - marker_size_inches, corner_offset),  # Top-right
                (corner_offset, paper_size_inches[1] - corner_offset - marker_size_inches),  # Bottom-left
                (paper_size_inches[0] - corner_offset - marker_size_inches, 
                 paper_size_inches[1] - corner_offset - marker_size_inches)  # Bottom-right
            ]
        
        # Set default marker IDs if none provided
        if marker_ids is None:
            marker_ids = list(range(len(marker_locations)))
        
        # Validate inputs
        if len(marker_locations) != len(marker_ids):
            raise ValueError("Number of marker locations must match number of marker IDs")
        
        print(f"\nCreating marker sheet for {paper_size_inches[0]}x{paper_size_inches[1]} inch paper...")
        print(f"Placing {len(marker_locations)} markers at custom locations")
        
        # Determine paper dimensions in pixels
        if input_sheet is not None:
            paper_height_px, paper_width_px = input_sheet.shape[:2]
            # Calculate effective DPI from input sheet and specified paper size
            effective_dpi_x = paper_width_px / paper_size_inches[0]
            effective_dpi_y = paper_height_px / paper_size_inches[1]
            if paper_size_inches[0] / paper_size_inches[1] != paper_width_px / paper_height_px:
              raise ValueError("Width/height inches not equal to width/height px")
            
            print(f"Using existing sheet: {paper_width_px}x{paper_height_px} pixels")
            # Use average DPI for marker sizing
            dpi = int((effective_dpi_x + effective_dpi_y) / 2)
        else:
            # Create new sheet
            paper_width_px = int(paper_size_inches[0] * dpi)
            paper_height_px = int(paper_size_inches[1] * dpi)
        
        marker_size_px = int(marker_size_inches * dpi)
        
        print(f"Paper size: {paper_width_px}x{paper_height_px} pixels")
        print(f"Marker size: {marker_size_px}x{marker_size_px} pixels ({marker_size_inches}\")")
        
        # Create or use existing sheet
        if input_sheet is not None:
            # Convert to grayscale if it's color
            if len(input_sheet.shape) == 3:
                sheet = cv.cvtColor(input_sheet, cv.COLOR_BGR2GRAY)
            else:
                sheet = input_sheet.copy()
        else:
            # Create white sheet
            sheet = np.ones((paper_height_px, paper_width_px), dtype=np.uint8) * 255
        
        # Convert marker locations from inches to pixels
        marker_positions_px = []
        for x_inches, y_inches in marker_locations:
            x_px = int(x_inches * dpi)
            y_px = int(y_inches * dpi)
            marker_positions_px.append((x_px, y_px))
        
        # Place markers at specified positions
        placed_markers = 0
        for i, ((x_px, y_px), marker_id) in enumerate(zip(marker_positions_px, marker_ids)):
            
            print(f"Placing marker {marker_id} at ({marker_locations[i][0]:.2f}\", {marker_locations[i][1]:.2f}\") = ({x_px}, {y_px}) pixels")
            
            # Generate marker
            marker_img = cv.aruco.generateImageMarker(self.dictionary, marker_id, marker_size_px)
            
            # Check if marker fits within bounds
            if (x_px >= 0 and y_px >= 0 and 
                x_px + marker_size_px <= paper_width_px and 
                y_px + marker_size_px <= paper_height_px):
                
                # Place marker
                sheet[y_px:y_px+marker_size_px, x_px:x_px+marker_size_px] = marker_img
                
                # Add ID label
                font_scale = max(0.3, marker_size_inches * 0.4)
                thickness = max(1, int(marker_size_inches * 2))
                
                # Position label to avoid going off edges
                label_x = max(10, min(x_px, paper_width_px - 80))
                label_y = max(20, min(y_px + marker_size_px + 20, paper_height_px - 10))
                
                cv.putText(sheet, f"ID: {marker_id}", 
                          (label_x, label_y),
                          cv.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness)
                
                placed_markers += 1
                print(f"   ✓ Successfully placed marker {marker_id}")
            else:
                print(f"   ✗ Marker {marker_id} doesn't fit at ({x_px}, {y_px}) - bounds exceeded")
        
        # Save sheet
        save_dir = os.path.join(ASSETS_DIR, "aruco_output")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cv.imwrite(save_dir + os.path.sep + filename, sheet)

        print(f"✓ Marker sheet saved as {save_dir + os.path.sep + filename}")
        print(f"✓ Successfully placed {placed_markers}/{len(marker_ids)} markers")
        print(f"✓ Print at {dpi} DPI for accurate {marker_size_inches}\" markers")
        
        return filename
    
    def pdf_to_nparray(self, pdf_path:str, dpi:int = PAGE_DPI, is_grayscale:bool = True) -> Union[np.ndarray, list[np.ndarray]]:
        """ Converts pdf at path to np.ndarray 

        Args:
        pdf_path: fullpath to the pdf in question
        is_grayscale: if the pdf should be converted to grayscale color mode. Default true
        """
        pillow_pdf = pdf2image.convert_from_path(pdf_path=pdf_path, hide_annotations=True, dpi=dpi, grayscale=is_grayscale)
        
        print(f'Number of pages: {len(pillow_pdf)}')

        # Use a list to collect pages
        np_pdf = []  
        for i, page in enumerate(pillow_pdf):
            page_gray = np.array(page)
            print(page_gray.shape)
            np_pdf.append(page_gray)  # Append the 2D array to list
        

        if np_pdf:
            print(f"Each page shape: {np_pdf[0].shape}")
            print(f"Page data type: {np_pdf[0].dtype}")

        # extract page array if there's only 1 page
        if len(np_pdf) == 1:
            np_pdf = np_pdf[0]

        return np_pdf