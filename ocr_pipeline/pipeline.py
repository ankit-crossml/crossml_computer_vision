import cv2 
import os
import numpy as np
import shutil
from PIL import Image 
import tabula
import table_ocr
import csv
import pandas as pd 
import ocrmypdf
import camelot

import matplotlib.pyplot as plt 


import table_ocr.util
import table_ocr.extract_tables
import table_ocr.extract_cells
import table_ocr.ocr_image
import table_ocr.ocr_to_csv

from extract_table import process_image

class Document: 
    
    def __init__(self, location): 
        self.location = location
        self.doc_name = location.split("/")[-1].split(".")[-2]
        self.image_arrays = self.convert_doc_to_images(location)
        
        self.num_pages = len(self.image_arrays)
        self.image_locations = self.save_arrays_as_images(self.image_arrays)
        self.image_tables = self.save_table_elements(self.image_arrays)
        self.pdf = self.save_pdf_from_images("cache/saved_pdf",self.image_locations, "document")
        self.tables_pdf = self.save_pdf_from_images("cache/saved_pdf_tables",self.image_tables, "document")     
        self.tables_pdf_ocr = self.get_ocred_pdf(self.tables_pdf)
        
    def convert_doc_to_images(self,doc_location):
        
        multi_read = cv2.imreadmulti(doc_location)
        num_pages = len(multi_read[1])
        images = []
        for i in range(num_pages):
            images.append(multi_read[1][i])

        return images
    
    def save_arrays_as_images(self, list_arrays):
        
        if os.path.isdir("cache"):
            shutil.rmtree("cache")
        
        base_name = "cache/saved_images/image"
        folder_name = "/".join(base_name.split('/')[:-1])
        if not os.path.isdir(folder_name):
            os.mkdir("cache")
            os.mkdir(folder_name)
        i = 1 
        names = []
        for arr in list_arrays:
            name = base_name + str(i) + ".png"
            res = cv2.imwrite(name, arr)
            if not res:
                raise Exception("Image Not Saved!")
            names.append(name)
            i+=1
        return names 
    
    def save_table_elements(self, image_arrays):
        folder_name = "cache/table_elements"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
    
        table_elemets = []
        
        for image_arr in image_arrays:
            tables_for_one_page = process_image(image_arr)
            table_elemets.extend(tables_for_one_page)
        
        
        i = 1 
        names = []
        for image in table_elemets:
            name = os.path.join(folder_name, str(i) + ".png")
            res = cv2.imwrite(name, image)
            if not res:
                raise Exception("Image Not Saved!")
            names.append(name)
            i+=1
            
        return names
        
    
    def save_pdf_from_images(self, folder_name, img_locations, saved_name):
    
#         folder_name = "cache/saved_pdf"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
            
        images = []

        for img_loc in img_locations:
            image = Image.open(img_loc)
            image = image.convert('RGB')
            images.append(image)

        first_element = images.pop(0)       

        location = folder_name + "/" + saved_name + ".pdf"
        
        if image == []:
            first_element.save(location)
        else:
            first_element.save(location,save_all=True, append_images=images)

#         print("pdf saved!")
        return location
    
    def get_ocred_pdf(self, pdf):
        pdf_ocr = pdf.split(".")[0] + "_ocr.pdf"
        ocrmypdf.ocr(pdf, pdf_ocr, deskew=True)
        
        return pdf_ocr
    

    class OCR_pipeline():

    def __init__(self): 
        self.saved_folder = "results"        
        if not os.path.isdir("results"):
            os.mkdir("results")
    
    def run_tabula(self, doc):
        pdf = doc.tables_pdf_ocr
        tabula.read_pdf(pdf, pages="all")
        
        tabula.convert_into(pdf, os.path.join(self.saved_folder, doc.doc_name, "tabula.csv"), output_format="csv", pages='all')
        
        
    def run_camelot(self, doc):
        pdf = doc.tables_pdf_ocr
        tables = camelot.read_pdf(pdf)
        tables.export(os.path.join(self.saved_folder, doc.doc_name, "camelot.csv"), f='csv', compress=True)
    
    def run_table_ocr(self, doc):
        number = 1
        image_locs = doc.image_tables
        for image_loc in image_locs:
            csv_result = self.extract_line_items(image_loc)
            self.save_df_from_text(csv_result,"results", doc.doc_name, doc_number=number)
            number += 1
    
    def extract_line_items(self, image_path):
        image_tables = table_ocr.extract_tables.main([image_path])
        for _, tables in image_tables:
        # iterate all the detected tables
            for table in tables:
                # extract all cells in table and run OCR
                cells = table_ocr.extract_cells.main(table)
                ocr = [
                    table_ocr.ocr_image.main(cell, None)
                    for cell in cells
                    ]
                for c, o in zip(cells[:3], ocr[:3]):
                    with open(o) as ocr_file:
                        text = ocr_file.read().strip()
                        print("{}: {}".format(c, text))

                    return table_ocr.ocr_to_csv.text_files_to_csv(ocr)
                
    
    def save_df_from_text(self, csv_output,output_folder, doc_name, doc_number = 1, return_csv = False):
        if csv_output == None:
            return None

        folder_path = os.path.join(output_folder,doc_name)
        if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                print(folder_path)

        path = os.path.join(folder_path, "table_ocr")
        if not os.path.isdir(path):
                os.mkdir(path)

        output_file = os.path.join(path, str(doc_number) + ".csv")
        with open(output_file, "w") as f:
            f.write(csv_output)
            f.close()

        if return_csv:
            # read output csv as dataframe
            csvlines = csv.reader(open(output_file, 'r'))
            columns_len = max([len(line) for _, line in enumerate(csvlines)])
            df = pd.read_csv(output_file, names=[*range(0, columns_len)], header=None)

            return df
