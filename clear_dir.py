# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:15:48 2023

@author: matthew.sharr

Clear directory of _Feature folders containing masked individual bands, composites, and any predictions.
"""

import os, shutil

directory_path = r'C:\_Turbidity\Imagery'
directory = [os.path.join(directory_path, file) for file in os.listdir(directory_path)]

for folder in directory:
    for location in os.listdir(folder):
        full = os.path.join(folder, location)
        for f in os.listdir(full):
            file_path = os.path.join(full, f)
            if os.path.isdir(file_path):
                if '_Features_' in file_path and file_path.endswith('Bands'):
                    # shutil.rmtree(file_path) # remove to delete feature directories
                    print(f'Deleted {file_path}')
                elif os.path.isfile(file_path):
                    pass
    



