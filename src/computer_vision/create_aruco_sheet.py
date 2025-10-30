from src.computer_vision.aruco_marker_generator import ArucoMarkerGenerator
from src.core.constants import PAPER_SIZES, MARKER_SIZE, CORNER_OFFSET, PAGE_DPI, MARKER_IDS, ASSETS_DIR
import os
import cv2 as cv

PAPER = PAPER_SIZES.LETTER

if __name__ == '__main__':

  file_name = ("Keys1_wide_Aruco_1.2in.png", "Keys2_wide_Aruco_1.2in.png", "Keys3_wide_Aruco_1.2in.png")


  keys1_locations = [(CORNER_OFFSET, PAPER.height - CORNER_OFFSET - MARKER_SIZE), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, PAPER.height - CORNER_OFFSET - MARKER_SIZE)]
  keys1_ids = MARKER_IDS[0:2]

  #keys 2 markers will be in the middle
  keys2_locations = [(CORNER_OFFSET, PAPER.height/2 - MARKER_SIZE/2), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, PAPER.height/2 - MARKER_SIZE/2)]
  keys2_ids = MARKER_IDS[2:4]

  # keys 3 for 4 octaves 
  keys3_locations = [(CORNER_OFFSET, CORNER_OFFSET), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, CORNER_OFFSET)]
  keys3_ids = MARKER_IDS[4:]

  marker_adder = ArucoMarkerGenerator()
  
  # keys1_path = os.path.join(ASSETS_DIR,"aruco_input","Keys1.pdf")
  # keys2_path = os.path.join(ASSETS_DIR,"aruco_input","Keys2.pdf")

  keys1_path = os.path.join(ASSETS_DIR,"aruco_output","widened_edges", "wider_edges_adaptiveKeys1.png")
  keys2_path = os.path.join(ASSETS_DIR,"aruco_output","widened_edges", "wider_edges_adaptiveKeys2.png")

  # keys1_array = marker_adder.pdf_to_nparray(pdf_path=keys1_path)
  # keys3_array = keys1_array.copy()
  # keys2_array = marker_adder.pdf_to_nparray(pdf_path=keys2_path)

  keys1_array = cv.imread(keys1_path, cv.IMREAD_GRAYSCALE)
  keys2_array = cv.imread(keys2_path, cv.IMREAD_GRAYSCALE)
  keys3_array = keys1_array.copy()
  
  marker_adder.create_marker_sheet(keys1_array, 
                                   filename=file_name[0], 
                                   marker_locations=keys1_locations, 
                                   marker_ids=keys1_ids,
                                   marker_size_inches=MARKER_SIZE
                                  )
  marker_adder.create_marker_sheet(keys2_array, 
                                   filename=file_name[1], 
                                   marker_locations=keys2_locations, 
                                   marker_ids=keys2_ids,
                                   marker_size_inches=MARKER_SIZE
                                  )
  marker_adder.create_marker_sheet(keys3_array, 
                                   filename=file_name[2], 
                                   marker_locations=keys3_locations, 
                                   marker_ids=keys3_ids,
                                   marker_size_inches=MARKER_SIZE                                   
                                  )

