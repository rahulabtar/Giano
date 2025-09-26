#!/usr/bin/env python3
"""
Test script for custom marker locations in inches.
"""

from giano_aruco import ArucoMarkerSystem

def test_custom_locations():
    """Test marker placement at custom locations specified in inches."""
    
    aruco_system = ArucoMarkerSystem()
    
    print("Testing custom marker locations...")
    print("=" * 50)
    
    # Test 1: Default corner locations
    print("\n📍 Test 1: Default corner locations")
    filename, cols, rows = aruco_system.create_marker_sheet(
        paper_size_inches=(8.5, 11.0),
        marker_size_inches=1.0,
        filename="test_default_corners.png"
    )
    print(f"   Created: {filename}")
    
    # Test 2: Custom locations - cross pattern
    print("\n📍 Test 2: Cross pattern in center")
    cross_locations = [
        (3.75, 4.5),   # Center
        (3.75, 2.5),   # Top of cross
        (3.75, 6.5),   # Bottom of cross
        (2.75, 4.5),   # Left of cross
        (4.75, 4.5),   # Right of cross
    ]
    cross_ids = [10, 11, 12, 13, 14]
    
    filename, cols, rows = aruco_system.create_marker_sheet(
        paper_size_inches=(8.5, 11.0),
        marker_size_inches=0.75,
        marker_locations=cross_locations,
        marker_ids=cross_ids,
        filename="test_cross_pattern.png"
    )
    print(f"   Created: {filename}")
    
    # Test 3: Grid pattern
    print("\n📍 Test 3: Regular grid pattern")
    grid_locations = []
    grid_ids = []
    
    # Create 3x4 grid with 0.5" margins
    start_x, start_y = 1.0, 1.0  # Start 1" from edges
    spacing_x, spacing_y = 2.0, 2.0  # 2" spacing between markers
    marker_size = 0.6
    
    for row in range(4):
        for col in range(3):
            x = start_x + col * spacing_x
            y = start_y + row * spacing_y
            grid_locations.append((x, y))
            grid_ids.append(row * 3 + col + 20)  # IDs starting from 20
    
    filename, cols, rows = aruco_system.create_marker_sheet(
        paper_size_inches=(8.5, 11.0),
        marker_size_inches=marker_size,
        marker_locations=grid_locations,
        marker_ids=grid_ids,
        filename="test_grid_pattern.png"
    )
    print(f"   Created: {filename} with {len(grid_locations)} markers")
    
    # Test 4: Edge markers for document scanning
    print("\n📍 Test 4: Document edge markers for scanning")
    edge_locations = [
        (0.25, 0.25),   # Top-left corner
        (8.0, 0.25),    # Top-right corner
        (0.25, 10.5),   # Bottom-left corner
        (8.0, 10.5),    # Bottom-right corner
        (4.125, 0.25),  # Top center
        (4.125, 10.5),  # Bottom center
        (0.25, 5.375),  # Left center
        (8.0, 5.375),   # Right center
    ]
    edge_ids = [100, 101, 102, 103, 104, 105, 106, 107]
    
    filename, cols, rows = aruco_system.create_marker_sheet(
        paper_size_inches=(8.5, 11.0),
        marker_size_inches=0.5,
        marker_locations=edge_locations,
        marker_ids=edge_ids,
        filename="test_edge_markers.png"
    )
    print(f"   Created: {filename}")

def test_a4_custom():
    """Test custom locations on A4 paper."""
    
    print("\n" + "=" * 50)
    print("TESTING A4 PAPER CUSTOM LOCATIONS")
    print("=" * 50)
    
    aruco_system = ArucoMarkerSystem()
    
    # A4 dimensions: 8.27" x 11.69"
    a4_width, a4_height = 8.27, 11.69
    
    # Create calibration pattern for camera calibration
    print("\n📸 Camera calibration pattern on A4")
    cal_locations = []
    cal_ids = []
    
    # 4x6 calibration grid
    margin = 0.5
    cols, rows = 4, 6
    available_width = a4_width - 2 * margin
    available_height = a4_height - 2 * margin
    
    spacing_x = available_width / (cols - 1)
    spacing_y = available_height / (rows - 1)
    
    for row in range(rows):
        for col in range(cols):
            x = margin + col * spacing_x
            y = margin + row * spacing_y
            cal_locations.append((x, y))
            cal_ids.append(row * cols + col)
    
    filename, cols, rows = aruco_system.create_marker_sheet(
        paper_size_inches=(a4_width, a4_height),
        marker_size_inches=0.4,
        marker_locations=cal_locations,
        marker_ids=cal_ids,
        filename="a4_calibration_pattern.png"
    )
    print(f"   Created: {filename} with {len(cal_locations)} markers")

def test_with_existing_image():
    """Test overlaying markers on an existing image."""
    
    print("\n" + "=" * 50)
    print("TESTING WITH EXISTING IMAGE")
    print("=" * 50)
    
    import numpy as np
    import cv2 as cv
    
    aruco_system = ArucoMarkerSystem()
    
    # Create a fake document image (simulating a scanned document)
    paper_width_px = int(8.5 * 300)  # US Letter at 300 DPI
    paper_height_px = int(11.0 * 300)
    
    # Create a document-like image with some text areas
    document = np.ones((paper_height_px, paper_width_px), dtype=np.uint8) * 255
    
    # Add some "text" blocks (gray rectangles)
    cv.rectangle(document, (300, 400), (2200, 800), 200, -1)
    cv.rectangle(document, (300, 1000), (2200, 1400), 200, -1)
    cv.rectangle(document, (300, 1600), (2200, 2000), 200, -1)
    
    print("   Created simulated document image")
    
    # Add corner markers to the existing document
    corner_locations = [
        (0.3, 0.3),      # Top-left
        (8.0, 0.3),      # Top-right
        (0.3, 10.5),     # Bottom-left
        (8.0, 10.5),     # Bottom-right
    ]
    
    filename, cols, rows = aruco_system.create_marker_sheet(
        input_sheet=document,
        paper_size_inches=(8.5, 11.0),
        marker_size_inches=0.4,
        marker_locations=corner_locations,
        marker_ids=[200, 201, 202, 203],
        filename="document_with_markers.png"
    )
    print(f"   Added markers to document: {filename}")

if __name__ == "__main__":
    test_custom_locations()
    test_a4_custom()
    test_with_existing_image()
    
    print("\n🎉 All tests completed!")
    print("\n💡 Usage Summary:")
    print("   - Specify locations as (x, y) in inches from top-left")
    print("   - X increases going right, Y increases going down")
    print("   - (0, 0) is the top-left corner of the paper")
    print("   - Locations are automatically converted to pixels based on DPI")