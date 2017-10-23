"""
  @author: Xing Hao
  @email: xinghao.1st@gmail.com
"""

import web
import os
import cv2
import numpy as np
import sys
from time import time

sys.path.append("static")
import BOW
root_dir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
static_dir = root_dir+"/static"

render = web.template.render('templates/')
urls = ('/', 'upload')

class result:
    def __init__(self, fn, l, n):
        self.filename = fn
        self.label = l
        self.neighbors = n

class upload:
    def GET(self):
        return render.index()

    def POST(self):
        x = web.input(myfile={})
        filedir = static_dir+"/upload"
        if 'myfile' in x: # to check if the file-object is created
            filepath=x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
            filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
            fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
            fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
            fout.close() # closes the file, upload complete.

            # Get description and similar images
            start = time()
            labels = BOW.classify(filedir +'/'+ filename, 5)
            end = time()
            duration = end - start
            print "Find descriptions in ",duration," secs."
            start = time()
            list = BOW.knn(filedir +'/'+ filename, labels, 6)
            end = time()
            duration = end - start
            print "Find knn in ", duration, " secs."

            return render.upload(result(filename, ', '.join(labels), list))

if __name__ == "__main__":
   app = web.application(urls, globals())
   app.run()