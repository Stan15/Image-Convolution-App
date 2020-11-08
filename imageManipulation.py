import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog
import PIL.Image, PIL.ImageTk 
import time
import numpy
import scipy
from cv2 import cv2

class App: 
    def __init__(self, root):
        self.root = root
        self.imageMatrix = numpy.array([]).astype('uint8')
        self.layers = ['red','green','blue']
        self.identicalLayers = {}
        self.layerIsMoving = False

        self.kernels = []
        self.maxKernelSize = (5,5)
        
        self.settings = tk.Frame(root)
        self.settings.grid(row=0,column=0, sticky='n')
        self.display = tk.Frame(root)
        self.display.grid(row=0,column=1, sticky='n')

        self.isTransitionEnabled = True
        self.transitionDuration = .2
        self.transitionCurve = lambda x: x**1.2

        self.paddingSmall = 5
        self.paddingMedium = 10
        self.paddingLarge = 15

        self.titleFont = tkFont.Font(family="Helvetica", size=10)
        self.normalFont = tkFont.Font(family='Helvetica', size=8)
        self.smallFont = tkFont.Font(family='Helvetica', size=7)
        self.boldSmallFont = tkFont.Font(family='Helvetica', size=7, weight='bold')

        self.numberEntrySize = 3

        self.packDisplayWidgets(300,200)
        self.packSettingsWidgets()
    
    def packDisplayWidgets(self, defaultWidth, defaultHeight, minSize=100, maxSize=600):
        self.minCanvasSize = minSize
        self.maxCanvasSize = maxSize

        self.canvas = tk.Canvas(self.display, width=defaultWidth, height=defaultHeight, background='white')
        self.canvas.pack(side='top', padx=self.paddingSmall, pady=(self.paddingSmall,self.paddingSmall/2))

        selectImageBttn = tk.Button(self.display, text="Select Image", command=self.loadImage)
        selectImageBttn.pack(pady=(0,self.paddingSmall/2))
    
    def packSettingsWidgets(self):
        self.layersFrame = tk.Frame(self.settings)
        self.layersFrame.grid(row=0, column=0, padx=self.paddingSmall, pady=self.paddingSmall/2, sticky='nw')
        self.kernelFrame = tk.Frame(self.settings)
        self.kernelFrame.grid(row=1, column=0, padx=self.paddingSmall, pady=self.paddingSmall/2, sticky='nw')
        self.error_msg = tk.Label(self.settings, text="", foreground='red')
        self.error_msg.grid(row=1, column=0, padx=self.paddingSmall, pady=self.paddingSmall/2, sticky='s')

        self.packLayersWidgets(self.layersFrame)
        self.packKernelWidgets(self.kernelFrame)
    
    def packLayersWidgets(self, parent):
        self.layersTitle = tk.Frame(parent)
        self.layersTitle.grid(row=0, column=0, pady=self.paddingSmall/2, sticky='nw')
        layers_text = tk.Label(self.layersTitle, text="Layers", font=self.titleFont)
        layers_text.grid(row=0, column=0, sticky='nw')

        self.layersControls = tk.Frame(parent)
        self.layersControls.grid(row=1, column=0, pady=self.paddingSmall/2)

        self.layersListBox = tk.Listbox(self.layersControls, selectmode='multiple')
        self.layersListBox.pack(side='left')
        self.updateLayersListBox()

        self.buttons = tk.Frame(self.layersControls)
        self.buttons.pack(side='right', padx=(self.paddingSmall,0))
        move_up = tk.Button(self.buttons, text="▲", command=lambda: self.moveLayers(self.getSelectedLayers(), 'up'))
        move_up['font'] = self.boldSmallFont
        move_up.grid(row=0, column=0, pady=self.paddingSmall/2)
        move_down = tk.Button(self.buttons, text="▼", command=lambda: self.moveLayers(self.getSelectedLayers(), 'down'))
        move_down['font'] = self.boldSmallFont
        move_down.grid(row=1, column=0, pady=self.paddingSmall/2)
        flatten = tk.Button(self.buttons, text="Flatten", command=lambda: self.flattenLayers(self.getSelectedLayers()))
        flatten.grid(row=0, column=1, rowspan=2, padx=(self.paddingSmall*2,0))

    def packKernelWidgets(self, parent):
        self.kernelSizeFrame = tk.Frame(parent)
        self.kernelSizeFrame.grid(row=0, column=0, sticky='nw')
        self.kernelGridFrame = tk.Frame(parent)
        self.kernelGridFrame.grid(row=1, column=0, sticky='nw')

        self.packKernelSizeWidgts(self.kernelSizeFrame)
    
    def packKernelSizeWidgts(self, parent):
        self.kernelTitle = tk.Frame(parent)
        self.kernelTitle.grid(row=0, column=0, pady=self.paddingSmall/2, sticky='nw')
        kernel_text = tk.Label(self.kernelTitle, text="Kernel", font=self.titleFont)
        kernel_text.grid(row=0, column=0, sticky='nw')

        self.kernelSize = tk.Frame(parent)
        self.kernelSize.grid(row=1, column=0, sticky='nw')
        kernel_size_text = tk.Label(self.kernelSize, text="Size:", font=self.normalFont)
        kernel_size_text.grid(row=0, column=0, padx=(0,self.paddingSmall))

        self.kernel_num_rows = tk.Entry(self.kernelSize, width=self.numberEntrySize)
        self.kernel_num_rows.grid(row=0, column=1)
        
        timesSymbol = tk.Label(self.kernelSize, text="×", font=self.boldSmallFont)
        timesSymbol.grid(row=0, column=2, padx=self.paddingSmall)
        
        self.kernel_num_cols = tk.Entry(self.kernelSize, width=self.numberEntrySize)
        self.kernel_num_cols.grid(row=0, column=3)
        
        self.empty_kernel_button = tk.Button(self.kernelSize, text="🡆", height=1, command=lambda: self.parseAddEmptyKernel(self.kernel_num_rows.get(), self.kernel_num_cols.get()))
        self.empty_kernel_button['font'] = self.smallFont
        self.empty_kernel_button.grid(row=0, column=4, padx=self.paddingSmall)

    def parseAddEmptyKernel(self, numRows, numCols):
        if (self.kernels)>2:
            self.error("You can only have a maximum of 2 kernels at once")
            return
        try:
            numRows = int(numRows)
            numCols = int(numCols)
            if numRows<=0 or numCols<=0 or numRows>self.maxKernelSize[0] or numCols>self.maxKernelSize[1]:
                self.error("Kernel size must be in the given range [1,{}] x [1,{}]".format(self.maxKernelSize[0], self.maxKernelSize[1]))
                return
        except Exception:
            self.error("Kernel size must be an integer")
            return

    def error(self, message):
        self.error_msg.config(text=message)

    def packKernelGridWidgets(self, parent):
        parent = None
        return
    
    def loadImage(self):
        filePath = tk.filedialog.askopenfilename(filetypes=[("Image File","*.jpg *.png")])
        self.imageMatrix = cv2.imread(filePath)
        self.imageMatrix = cv2.cvtColor(self.imageMatrix, cv2.COLOR_BGR2RGB)
        self.layers = ['red','green','blue']
        self.resetIdenticalLayers()
        self.updateLayersListBox()
        self.clearLayersSelection()
        self.drawCanvas()
        pass

    def flattenLayers(self,layersToCombine):
        if len(layersToCombine)<=1 or len(self.identicalLayers['red'])==3: #that means that all three (red, green and blue) are equal, so flattening does nothing
            self.clearLayersSelection()
            return
        
        layersIdx = [self.layers.index(layer) for layer in layersToCombine]
        
        newImg = self.imageMatrix.copy()
        matLayers = newImg[:,:,layersIdx].astype('float32')
        matLayers = numpy.power(matLayers, 2)
        matLayers = numpy.sum(matLayers, axis=2)
        matLayers = numpy.divide(matLayers, len(layersIdx))
        matLayers = numpy.power(matLayers, 0.5).astype('uint8')
        matLayers = numpy.repeat(matLayers[:,:,numpy.newaxis], len(layersIdx),axis=2)
        
        newImg[:,:,layersIdx] = matLayers

        newIdenticalLayers = {'red':['red'], 'green': ['green'], 'blue':['blue']}
        for key in newIdenticalLayers:
            layer = newIdenticalLayers[key][0]
            if layer in layersToCombine:
                newIdenticalLayers[key] = layersToCombine
        
        self.resetIdenticalLayers(newIdenticalLayers)
        
        self.clearLayersSelection()
        if self.isTransitionEnabled:
            self.transition(newImg, self.transitionDuration, self.transitionCurve)
        else:
            self.imageMatrix = newImg.copy()
            self.drawCanvas()

    def moveLayers(self, layers, direction):
        if self.imageMatrix.size==0 or self.layerIsMoving:
            return
         
        self.layerIsMoving = True
        newImg = self.imageMatrix.copy()
        
        if direction=='down':
            layers.reverse()
        
        for layer in layers:
            idx = self.layers.index(layer)
            if direction=='up':
                if idx==0:
                    continue
                newOrder = list(range(0,len(self.layers)))
                newOrder[idx] = newOrder[idx-1]
                newOrder[idx-1] = idx
            elif direction=='down':
                if idx==len(self.layers)-1:
                    continue
                newOrder = list(range(0,len(self.layers)))
                newOrder[idx] = newOrder[idx+1]
                newOrder[idx+1] = idx
            
            newImg[:,:,:] = newImg[:,:,newOrder]
            self.layers = [self.layers[i] for i in newOrder]
        
        self.updateLayersListBox()
        if self.isTransitionEnabled:
            self.transition(newImg, self.transitionDuration, self.transitionCurve)
        else:
            self.imageMatrix = newImg.copy()
            self.drawCanvas()
        self.layerIsMoving = False
    
    def transition(self, newMat, duration, curve, anchor=False):
        for i in numpy.arange(0,1,0.01):
            try:
                val = curve(i)
                float(val)
                if val<0:
                    raise RuntimeWarning("Function does not have a positive range")
            except:
                self.imageMatrix = newMat.copy()
                self.drawCanvas()
                raise RuntimeWarning("Function is not continuous")
        
        oldMat = self.imageMatrix.copy()
        differenceMat = numpy.subtract(newMat.astype('int16'),oldMat.astype('int16'))

        #---------------------------------------------------------
        if anchor:
            absoluteDiffMat = numpy.absolute(differenceMat)
            #to prevent 0 division error when finding ratio max(absoluteDiffMat)/absoluteDiffMat,
            #since the max difference is 255 and min is 1, we want to change
            #absolute difference values in absoluteDiffMat which are smaller than than 1 to 1.
            absoluteDiffMat[absoluteDiffMat<1] = 1
            diffRatioMat = numpy.max(absoluteDiffMat)*numpy.power(absoluteDiffMat.astype('float16'),-1)
        #----------------------------------------------------------

        elapsed = 0
        startTime = time.time()
        while elapsed<duration:
            timeRatio = elapsed/duration
            transitionRatio = curve(timeRatio)
            if transitionRatio>1:
                transitionRatio=1
            
            if anchor:
                transitionRatioMat = diffRatioMat*numpy.float16(transitionRatio)
                transitionRatioMat[transitionRatioMat>1] = 1
                increment = numpy.multiply(differenceMat,transitionRatioMat)
            else:
                increment = differenceMat*numpy.float16(transitionRatio)

            self.imageMatrix = numpy.add(oldMat.astype('int16'), increment.astype('int16')).astype('uint8')
            self.drawCanvas()
        
            elapsed = time.time() - startTime

        self.imageMatrix = newMat
        self.drawCanvas()

    def drawCanvas(self):
        if self.imageMatrix.size==0:
            return
        image = PIL.Image.fromarray(self.imageMatrix.astype('uint8'))

        if image.size[0]>self.maxCanvasSize:
            image = image.resize((self.maxCanvasSize, int(self.maxCanvasSize*(image.size[1]/image.size[0]))))
        if image.size[1]>self.maxCanvasSize:
            image = image.resize((int((self.maxCanvasSize*(image.size[0]/image.size[1])), self.maxCanvasSize)))
        
        if image.size[0]<self.minCanvasSize:
            image = image.resize((self.minCanvasSize, int(self.minCanvasSize*(image.size[1]/image.size[0]))))
        if image.size[1]<self.minCanvasSize:
            image = image.resize((int((self.minCanvasSize*(image.size[0]/image.size[1])), self.minCanvasSize)))

        self.photo = PIL.ImageTk.PhotoImage(image = image)
        self.canvas.config(width=image.size[0], height=image.size[1])
        self.canvas.create_image(0,0,anchor="nw", image=self.photo)
        self.canvas.update()

    def updateLayersListBox(self):
        selectedLayers = self.getSelectedLayers()

        self.layersListBox.delete(0,'end')
        for idx in range(len(self.layers)):
            self.layersListBox.insert(idx, self.layers[idx])
            if self.layers[idx] in selectedLayers:
                self.layersListBox.selection_set(idx)
        
        #resetting listbox and window size
        self.layersListBox.config(width=0, height=0)
        self.settings.winfo_toplevel().wm_geometry("")
        
        self.resetIdenticalLayers(self.identicalLayers)

    def getSelectedLayers(self):
        selectedLayers = [self.layersListBox.get(idx) for idx in self.layersListBox.curselection()]
        return selectedLayers
    
    def clearLayersSelection(self):
        self.layersListBox.selection_clear(0, "end")

    def resetIdenticalLayers(self, identical=None):
        if identical is None:
            self.identicalLayers = {'red':['red'], 'green': ['green'], 'blue':['blue']} #they are each only identical to themselves
        else:
            self.identicalLayers = identical
        
        [self.layersListBox.itemconfig(idx, bg = "white") for idx in range(self.layersListBox.size())]

        color = '#fff6c7'
        colored = []
        for key in self.identicalLayers:
            if len(self.identicalLayers[key])==1:
                continue
            for layer in self.identicalLayers[key]:
                if layer not in colored:
                    idx = self.layers.index(layer)
                    self.layersListBox.itemconfig(idx, bg = color)
                    colored.append(layer)
        

root = tk.Tk()
myApp = App(root)
root.mainloop()
