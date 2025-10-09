from src.computer_vision.aruco_system import ArucoMarkerSystem
from src.core.constants import PAPER_SIZES, MARKER_SIZE, CORNER_OFFSET, PAGE_DPI, MARKER_IDS, ASSETS_DIR
import pdf2image
from PIL import Image
import numpy as np
import os
import cv2 as cv
from typing import Union

PAPER = PAPER_SIZES.LETTER

def pdf_to_nparray(pdf_path:str, dpi:int = PAGE_DPI, is_grayscale:bool = True) -> Union[np.ndarray, list[np.ndarray]]:
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

if __name__ == '__main__':

  file_name = ("Keys1_Aruco_1.5in.png", "Keys2_Aruco_1.5in.png", "Keys3_Aruco_1.5in.png")


  keys1_locations = [(CORNER_OFFSET, PAPER.height - CORNER_OFFSET - MARKER_SIZE), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, PAPER.height - CORNER_OFFSET - MARKER_SIZE)]
  keys1_ids = MARKER_IDS[0:2]

  #keys 2 markers will be in the middle
  keys2_locations = [(CORNER_OFFSET, PAPER.height/2 - MARKER_SIZE/2), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, PAPER.height/2 - MARKER_SIZE/2)]
  keys2_ids = MARKER_IDS[2:4]

  # keys 3 for 4 octaves 
  keys3_locations = [(CORNER_OFFSET, CORNER_OFFSET), (PAPER.width - CORNER_OFFSET - MARKER_SIZE, CORNER_OFFSET)]
  keys3_ids = MARKER_IDS[4:]

  marker_adder = ArucoMarkerSystem()
  
  keys1_path = ASSETS_DIR+"aruco_input"+os.path.sep+"Keys1.pdf"
  keys2_path = ASSETS_DIR+"aruco_input"+os.path.sep+"Keys2.pdf"
  
  keys1_array = pdf_to_nparray(keys1_path)
  keys3_array = keys1_array.copy()
  keys2_array = pdf_to_nparray(keys2_path)

  
  marker_adder.create_marker_sheet(keys1_array, filename=file_name[0], marker_locations=keys1_locations, marker_ids=keys1_ids)
  marker_adder.create_marker_sheet(keys2_array, filename=file_name[1], marker_locations=keys2_locations, marker_ids=keys2_ids)
  marker_adder.create_marker_sheet(keys3_array, filename=file_name[2], marker_locations=keys3_locations, marker_ids=keys3_ids)

