from giano_aruco import ArucoMarkerSystem
import pdf2image
from PIL import Image
import numpy as np
import os
import cv2 as cv
from typing import Union

PAGE_DPI = 300
MARKER_SIZE = 1

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


  keys1_locations = [(0.25, 10.75 - MARKER_SIZE), (8.25 - MARKER_SIZE, 10.75 - MARKER_SIZE)]
  keys1_ids = [10,11]

  #TODO: fix keys2 marker locations
  keys2_locations = [(8.5 - MARKER_SIZE - 0.25, 0.25), (8.5 - MARKER_SIZE - 0.25, 11 - MARKER_SIZE - 0.25)]
  keys2_ids = [12,13]
  marker_adder = ArucoMarkerSystem()

  keys1_path = os.path.curdir+os.path.sep+"aruco_input"+os.path.sep+"Keys1.pdf"
  keys2_path = os.path.curdir+os.path.sep+"aruco_input"+os.path.sep+"Keys2.pdf"
  
  keys1_array = pdf_to_nparray(keys1_path)
  keys2_array = pdf_to_nparray(keys2_path)
  
  #TODO: Implement marker2 marker sheet
  marker_adder.create_marker_sheet(keys1_array, filename="Keys1withAruco.png", marker_locations=keys1_locations, marker_ids=keys1_ids)
