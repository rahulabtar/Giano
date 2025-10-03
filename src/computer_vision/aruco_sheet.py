from giano_aruco import ArucoMarkerSystem
import pdf2image
from PIL import Image
import numpy as np
import os
import cv2 as cv
from typing import Union

PAGE_DPI = 300
MARKER_SIZE = 1
CORNER_OFFSET = 0.25
MARKER_IDS = [10,11,12,13,14,15]

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

  keys1_locations = [(CORNER_OFFSET, 11 - CORNER_OFFSET - MARKER_SIZE), (8.5 - CORNER_OFFSET - MARKER_SIZE, 11 - CORNER_OFFSET - MARKER_SIZE)]
  keys1_ids = MARKER_IDS[0:2]

  #keys 2 markers will be in the middle
  keys2_locations = [(CORNER_OFFSET, 11/2 - MARKER_SIZE/2), (8.5 - CORNER_OFFSET - MARKER_SIZE, 11/2 - MARKER_SIZE/2)]
  keys2_ids = MARKER_IDS[2:4]

  # keys 3 for 4 octaves 
  keys3_locations = [(CORNER_OFFSET, CORNER_OFFSET), (8.5 - CORNER_OFFSET - MARKER_SIZE, CORNER_OFFSET)]
  keys3_ids = MARKER_IDS[4:]

  marker_adder = ArucoMarkerSystem()
  
  keys1_path = os.path.curdir+os.path.sep+"aruco_input"+os.path.sep+"Keys1.pdf"
  keys2_path = os.path.curdir+os.path.sep+"aruco_input"+os.path.sep+"Keys2.pdf"
  
  keys1_array = pdf_to_nparray(keys1_path)
  keys3_array = keys1_array.copy()
  keys2_array = pdf_to_nparray(keys2_path)

  
  marker_adder.create_marker_sheet(keys1_array, filename="Keys1withAruco.png", marker_locations=keys1_locations, marker_ids=keys1_ids)
  marker_adder.create_marker_sheet(keys2_array, filename="Keys2withAruco.png", marker_locations=keys2_locations, marker_ids=keys2_ids)
  marker_adder.create_marker_sheet(keys3_array, filename="Keys3withAruco.png", marker_locations=keys3_locations, marker_ids=keys3_ids)

