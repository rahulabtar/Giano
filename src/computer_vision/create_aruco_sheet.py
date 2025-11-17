import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.computer_vision.aruco_marker_generator import ArucoMarkerGenerator
from src.core.constants import PAPER_SIZES, MARKER_SIZE, CORNER_OFFSET, PAGE_DPI, MARKER_IDS, ASSETS_DIR
import cv2 as cv

PAPER = PAPER_SIZES.LETTER

def parse_marker_locations(locations_str: str) -> list:
    """
    Parse marker locations from string format: "x1,y1 x2,y2 ..." or "x1,y1,x2,y2,..."
    
    Args:
        locations_str: String with locations in format "x1,y1 x2,y2" or "x1,y1,x2,y2"
        
    Returns:
        List of (x, y) tuples
    """
    if not locations_str:
        return None
    
    # Try space-separated first
    if ' ' in locations_str:
        parts = locations_str.split()
    else:
        # Comma-separated, need to pair them
        parts = locations_str.split(',')
        if len(parts) % 2 != 0:
            raise ValueError("Marker locations must have even number of values (x,y pairs)")
        parts = [f"{parts[i]},{parts[i+1]}" for i in range(0, len(parts), 2)]
    
    locations = []
    for part in parts:
        try:
            x, y = map(float, part.split(','))
            locations.append((x, y))
        except ValueError:
            raise ValueError(f"Invalid location format: {part}. Expected 'x,y'")
    
    return locations

def parse_marker_ids(ids_str: str) -> list:
    """
    Parse marker IDs from comma or space-separated string.
    
    Args:
        ids_str: String with IDs like "0,1,2" or "0 1 2"
        
    Returns:
        List of integers
    """
    if not ids_str:
        return None
    
    # Try comma-separated first, then space
    if ',' in ids_str:
        ids = [int(x.strip()) for x in ids_str.split(',')]
    else:
        ids = [int(x.strip()) for x in ids_str.split()]
    
    return ids

def main_cli():
    """CLI interface for create_marker_sheet"""
    parser = argparse.ArgumentParser(
        description='Create ArUco marker sheets with custom configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a blank sheet with default 4 corner markers:
  python create_aruco_sheet.py --filename output.png

  # Create sheet from existing image with custom markers:
  python create_aruco_sheet.py --input image.png --filename output.png \\
      --marker-ids "0,1,2" --marker-locations "0.5,0.5 7.5,0.5 0.5,10.5"

  # Custom paper size and marker size:
  python create_aruco_sheet.py --filename output.png \\
      --paper-size "11,8.5" --marker-size 1.5 --dpi 300
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input image file path (PNG, JPG, etc.). If not provided, creates blank sheet.'
    )
    
    parser.add_argument(
        '--filename', '-f',
        type=str,
        default='aruco_marker_sheet.png',
        help='Output filename (default: aruco_marker_sheet.png)'
    )
    
    parser.add_argument(
        '--paper-size',
        type=str,
        help='Paper size in inches as "width,height" (default: 8.5,11.0 for Letter)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution in dots per inch (default: 300)'
    )
    
    parser.add_argument(
        '--marker-size',
        type=float,
        default=1.0,
        help='Marker size in inches (default: 1.0)'
    )
    
    parser.add_argument(
        '--marker-locations',
        type=str,
        help='Marker locations in inches from top-left. Format: "x1,y1 x2,y2 ..." or "x1,y1,x2,y2,..."'
    )
    
    parser.add_argument(
        '--marker-ids',
        type=str,
        help='Marker IDs as comma or space-separated list (e.g., "0,1,2" or "0 1 2")'
    )
    
    args = parser.parse_args()
    
    # Parse paper size
    paper_size = (8.5, 11.0)  # Default Letter size
    if args.paper_size:
        try:
            width, height = map(float, args.paper_size.split(','))
            paper_size = (width, height)
        except ValueError:
            parser.error("--paper-size must be in format 'width,height' (e.g., '8.5,11.0')")
    
    # Parse marker locations
    marker_locations = parse_marker_locations(args.marker_locations) if args.marker_locations else None
    
    # Parse marker IDs
    marker_ids = parse_marker_ids(args.marker_ids) if args.marker_ids else None
    
    # Validate that locations and IDs match
    if marker_locations and marker_ids and len(marker_locations) != len(marker_ids):
        parser.error(f"Number of marker locations ({len(marker_locations)}) must match number of marker IDs ({len(marker_ids)})")
    
    # Load input image if provided
    input_sheet = None
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {args.input}")
        input_sheet = cv.imread(str(input_path), cv.IMREAD_GRAYSCALE)
        if input_sheet is None:
            parser.error(f"Failed to load image: {args.input}")
        print(f"Loaded input image: {input_sheet.shape}")
    
    # Create marker sheet
    marker_adder = ArucoMarkerGenerator()
    marker_adder.create_marker_sheet(
        input_sheet=input_sheet,
        paper_size_inches=paper_size,
        dpi=args.dpi,
        marker_size_inches=args.marker_size,
        marker_locations=marker_locations,
        marker_ids=marker_ids,
        filename=args.filename
    )

if __name__ == '__main__':
    # Check if CLI arguments are provided
    if len(sys.argv) > 1:
        main_cli()
    else:
        # Run original script behavior
        file_name = ("Keys1_9px_Aruco_1.2in.png", "Keys2_9px_Aruco_Edge_Markers_1.2in.png", "Keys3_9px_Aruco_1.2in.png")

        keys1_locations = [(CORNER_OFFSET, PAPER.height - CORNER_OFFSET - MARKER_SIZE), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, PAPER.height - CORNER_OFFSET - MARKER_SIZE)]
        keys1_ids = MARKER_IDS[0:2]

        #keys 2 markers will be in the middle
        keys2_locations = [(CORNER_OFFSET, CORNER_OFFSET), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, CORNER_OFFSET)]
        keys2_ids = MARKER_IDS[2:4]

        # keys 3 for 4 octaves 
        keys3_locations = [(CORNER_OFFSET, CORNER_OFFSET), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, CORNER_OFFSET)]
        keys3_ids = MARKER_IDS[4:]

        marker_adder = ArucoMarkerGenerator()
        
        # keys1_path = os.path.join(ASSETS_DIR,"aruco_input","Keys1.pdf")
        # keys2_path = os.path.join(ASSETS_DIR,"aruco_input","Keys2.pdf")

        keys1_path = Path(ASSETS_DIR,"aruco_output","widened_edges", "wider_edges_adaptive_9_Keys1.png")
        keys2_path = Path(ASSETS_DIR,"aruco_output","widened_edges", "wider_edges_adaptive_9_Keys2_with_top_line.png")

        # keys1_array = marker_adder.pdf_to_nparray(pdf_path=keys1_path)
        # keys3_array = keys1_array.copy()
        # keys2_array = marker_adder.pdf_to_nparray(pdf_path=keys2_path)

        keys1_array = cv.imread(str(keys1_path), cv.IMREAD_GRAYSCALE)
        keys2_array = cv.imread(str(keys2_path), cv.IMREAD_GRAYSCALE)
        keys3_array = keys1_array.copy()
        
        """marker_adder.create_marker_sheet(keys1_array, 
                                         filename=file_name[0], 
                                         marker_locations=keys1_locations, 
                                         marker_ids=keys1_ids,
                                         marker_size_inches=MARKER_SIZE
                                        )
        """
        marker_adder.create_marker_sheet(keys2_array, 
                                         filename=file_name[1], 
                                         marker_locations=keys2_locations, 
                                         marker_ids=keys2_ids,
                                         marker_size_inches=MARKER_SIZE,
                                         dpi=PAGE_DPI
                                        )
        """marker_adder.create_marker_sheet(keys3_array, 
                                         filename=file_name[2], 
                                         marker_locations=keys3_locations, 
                                         marker_ids=keys3_ids,
                                         marker_size_inches=MARKER_SIZE                                   
                                        )
        """
