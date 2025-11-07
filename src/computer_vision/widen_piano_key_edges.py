import cv2 as cv
import numpy as np
import os
from pathlib import Path


def widen_black_edges(image, dilation_iterations=2, line_thickness=1):
    """
    Detect and widen the black edges between piano keys.
    
    Args:
        image: Input image (BGR or grayscale)
        dilation_iterations: Number of iterations to widen edges (default: 2)
        line_thickness: Thickness of vertical lines to detect (default: 1)
    
    Returns:
        Modified image with widened black edges
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create a copy of the original to modify
    if len(image.shape) == 3:
        result = image.copy()
    else:
        result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    
    # Detect dark lines (black edges between keys)
    # Use adaptive threshold or simple threshold for dark regions
    _, dark_mask = cv.threshold(gray, 60, 255, cv.THRESH_BINARY_INV)
    
    # Detect vertical lines (piano key separators)
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, line_thickness))
    vertical_lines = cv.morphologyEx(dark_mask, cv.MORPH_OPEN, vertical_kernel)
    
    # Dilate the lines to make them wider
    dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    dilated_lines = cv.dilate(vertical_lines, dilation_kernel, iterations=dilation_iterations)
    
    # Create a mask for the widened edges
    edge_mask = dilated_lines > 0
    
    # Make the edges darker (true black)
    if len(image.shape) == 3:
        result[edge_mask] = [0, 0, 0]  # Pure black in BGR
    else:
        result[edge_mask] = 0
    
    return result, dilated_lines


def widen_edges_advanced(image, method='adaptive', edge_width=3):
    """
    Advanced method to widen black edges between piano keys.
    Detects and widens both horizontal and vertical lines.
    
    Args:
        image: Input image
        method: 'adaptive', 'canny', or 'threshold'
        edge_width: Desired width of edges in pixels
    
    Returns:
        Modified image with widened edges and edge mask
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create a copy to modify
    result = image.copy() if len(image.shape) == 3 else cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    height, width = gray.shape
    
    # Detect lines based on method
    if method == 'adaptive':
        # Use adaptive thresholding to detect black regions
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv.THRESH_BINARY_INV, 11, 2)
        line_src = binary
        
    elif method == 'canny':
        # Use Canny edge detection
        line_src = cv.Canny(gray, 50, 150)
        
    else:  # threshold
        # Simple threshold for very dark regions
        _, line_src = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
    
    # Detect HORIZONTAL lines (between keys)
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    horizontal_lines = cv.morphologyEx(line_src, cv.MORPH_OPEN, horizontal_kernel)
    
    # Detect VERTICAL lines (key separators)
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    vertical_lines = cv.morphologyEx(line_src, cv.MORPH_OPEN, vertical_kernel)
    
    # Combine both horizontal and vertical lines
    combined_lines = cv.bitwise_or(horizontal_lines, vertical_lines)
    
    # Dilate horizontal lines to make them wider
    horizontal_dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, edge_width))
    dilated_horizontal = cv.dilate(horizontal_lines, horizontal_dilate_kernel, iterations=1)
    
    # Dilate vertical lines to make them wider
    vertical_dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT, (edge_width, 1))
    dilated_vertical = cv.dilate(vertical_lines, vertical_dilate_kernel, iterations=1)
    
    # Combine the dilated lines
    dilated = cv.bitwise_or(dilated_horizontal, dilated_vertical)
    
    # Apply the widened edges to the result
    edge_mask = dilated > 0
    if len(result.shape) == 3:
        result[edge_mask] = [0, 0, 0]  # Pure black
    else:
        result[edge_mask] = 0
    
    return result, dilated


def process_piano_key_images(input_dir, output_dir, edge_width=3, method='adaptive'):
    """
    Process all piano key images in a directory and widen the black edges.
    Detects and widens both horizontal and vertical lines between keys.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        edge_width: Width of edges in pixels
        method: Detection method to use ('adaptive', 'canny', or 'threshold')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all piano key images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(Path(input_dir).glob(ext))
    
    print(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        # Read image
        img = cv.imread(str(img_file), cv.IMREAD_COLOR)
        if img is None:
            print(f"  Error: Could not read {img_file.name}")
            continue
        
        # Process image
        result, edge_mask = widen_edges_advanced(img, method=method, edge_width=edge_width)
        
        # Save result
        output_path = os.path.join(output_dir, f"wider_edges_{method}_{edge_width}_{img_file.name}")
        cv.imwrite(output_path, result)
        
        # Save edge mask for debugging
        debug_path = os.path.join(output_dir, f"edge_mask_{method}_{edge_width}_{img_file.name}")
        cv.imwrite(debug_path, edge_mask)
        
    
    print(f"\nProcessed images saved to: {output_dir}")


def main():
    # Process piano key images with ArUco markers
    input_dir = "assets/aruco_input"
    output_dir = "assets/aruco_output/widened_edges"
    
    print("=" * 60)
    print("WIDENING BLACK EDGES BETWEEN PIANO KEYS")
    print("Detecting both horizontal and vertical lines")
    print("=" * 60)
    
    # Process all images
    process_piano_key_images(
        input_dir=input_dir,
        output_dir=output_dir,
        edge_width=9,  # Width of edges in pixels
        method='adaptive'  # Options: 'adaptive', 'canny', 'threshold'
    )
    
    print("\nPreview results:")
    print("✓ Both horizontal and vertical lines are detected and widened")
    print("✓ Edge detection method: 'adaptive' typically works best for clean key images")
    print("✓ Edge detection method: 'canny' works well for noisy or complex images")
    print("Adjust 'edge_width' parameter (1-10) to control edge thickness")
    print("\nTo test different settings, modify edge_width and method")


if __name__ == '__main__':
    main()

