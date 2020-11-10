#external libraries (install using pip in your system terminal)
import PIL.Image, PIL.ImageTk   #(pip install Pillow)
import numpy                    #(pip install numpy)
from scipy import ndimage       #(pip install scipy)
from cv2 import cv2             #(pip install opencv-python)

#native
import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog
import time
import copy
import os.path

class App: 
    def __init__(self, root):
        self.root = root
        self.imageMatrix = numpy.array([]).astype('uint8')
        self.imagePath = ""
        self.supportedFileTypes = ["png", "jpg"]
        self.layers = ['red','green','blue']
        self.identicalLayers = {}
        self.layerIsMoving = False
        
        self.kernels = []
        self.maxKernelSize = (5,5)
        self.maxKernelCount = 5

        self.getPresetKernels()

        self.defaultConvolvePadding = 0
        self.defaultConvolveStride = 1
        
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

        self.titleFont = tkFont.Font(family="Helvetica", size=10, weight='bold')
        self.normalFont = tkFont.Font(family='Helvetica', size=8)
        self.boldNormalFont = tkFont.Font(family='Helvetica', size=8, weight='bold')
        self.smallFont = tkFont.Font(family='Helvetica', size=7)
        self.boldSmallFont = tkFont.Font(family='Helvetica', size=7, weight='bold')

        self.numberEntrySize = 3

        self.packDisplayWidgets(300,200)
        self.packSettingsWidgets()
    
    def getPresetKernels(self):
        self.presetKernels = {
            'mean-blur': numpy.array([
                [1, 1, 1],
                [1, 1, 1],
                [1 ,1 ,1]
            ]),
            'gaussian-blur': numpy.array([
                [1, 2, 1],
                [2, 3, 2],
                [1 ,2 ,1]
            ]),
            'gaussian-blur bigger': numpy.array([
                [0.023528,	0.033969,	0.038393,	0.033969,	0.023528],
                [0.033969,	0.049045,	0.055432,	0.049045,	0.033969],
                [0.038393,	0.055432,	0.062651,	0.055432,	0.038393],
                [0.033969,	0.049045,	0.055432,	0.049045,	0.033969],
                [0.023528,	0.033969,	0.038393,	0.033969,	0.023528]
            ]),
            'sobel-X': numpy.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),
            'sobel-Y': numpy.array([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ]),
        }
    
    def getMaxConvolPadding(self):
        minKernelDim = float('inf')
        for kernel in self.kernels:
            minDim = min(kernel.shape)
            if minDim<minKernelDim:
                minKernelDim = minDim
        
        maxPadding = (minKernelDim/2)-0.5
        if maxPadding<0:
            maxPadding=0
        
        return maxPadding

    
    def packDisplayWidgets(self, defaultWidth, defaultHeight, minSize=100, maxSize=600):
        self.minCanvasSize = minSize
        self.maxCanvasSize = maxSize

        self.canvasFrame = tk.Frame(self.display)
        self.canvasFrame.grid(row=0, column=0)
        self.canvas = tk.Canvas(self.canvasFrame, width=defaultWidth, height=defaultHeight, background='white')
        self.canvas.pack(side='top', padx=self.paddingSmall, pady=(self.paddingSmall,self.paddingSmall/2))
        
        selectImageBttnFrame = tk.Frame(self.display)
        selectImageBttnFrame.grid(row=1, column=0)

        selectImageBttn = tk.Button(selectImageBttnFrame, text="Select Image", command=self.loadImage)
        selectImageBttn.grid(row=0, column=0, padx=self.paddingSmall, pady=(0,self.paddingSmall/2))
        
        restoreImageBttn = tk.Button(selectImageBttnFrame, text="Restore Image", command=lambda: self.loadImage(self.imagePath))
        restoreImageBttn.grid(row=0, column=1, padx=self.paddingSmall, pady=(0,self.paddingSmall/2))
        
        saveImageBttn = tk.Button(selectImageBttnFrame, text="Save Image", command=lambda: self.saveImage())
        saveImageBttn.grid(row=0, column=2, padx=self.paddingSmall, pady=(0,self.paddingSmall/2))
    
    def packSettingsWidgets(self):
        self.layersFrame = tk.Frame(self.settings)
        self.layersFrame.grid(row=0, column=0, padx=self.paddingSmall, pady=self.paddingSmall/2, sticky='nw')

        self.kernelFrame = tk.Frame(self.settings)
        self.kernelFrame.grid(row=1, column=0, padx=self.paddingSmall, pady=self.paddingSmall/2, sticky='nw')

        self.convolveFrame = tk.Frame(self.settings)
        self.convolveFrame.grid(row=2, column=0, padx=self.paddingSmall, pady=self.paddingSmall/2, sticky='nw')

        self.error_msg_var = tk.StringVar()
        self.error_msg_var.set("")
        self.error_msg_label = tk.Label(self.settings, textvariable=self.error_msg_var, foreground='red', wraplength=150)
        self.error_msg_label.grid(row=3, column=0, padx=self.paddingSmall, pady=self.paddingSmall/2, sticky='s')

        self.packLayersWidgets(self.layersFrame)
        self.packKernelWidgets(self.kernelFrame)
        self.packConvolveWidgets(self.convolveFrame)
    
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

    def packConvolveWidgets(self, parent):
        self.convolveTitle = tk.Frame(parent)
        self.convolveTitle.grid(row=0, column=0, sticky='nw')
        convolve_title_text = tk.Label(self.convolveTitle, text="Convolution", font=self.titleFont)
        convolve_title_text.grid(row=0, column=0, sticky='nw')

        self.convolveSettingsFrame = tk.Frame(parent)
        self.convolveSettingsFrame.grid(row=1, column=0, sticky='nw')

        convol_padding_text = tk.Label(self.convolveSettingsFrame, text="padding:", font=self.normalFont)
        convol_padding_text.grid(row=0, column=0, padx=(0,self.paddingSmall/2))

        self.convolvePadding = tk.Entry(self.convolveSettingsFrame, width=self.numberEntrySize)
        self.convolvePadding.grid(row=0, column=1)
        self.convolvePadding.delete(0,'end')
        self.convolvePadding.insert(0,self.defaultConvolvePadding)

        convol_stride_text = tk.Label(self.convolveSettingsFrame, text="stride:", font=self.normalFont)
        convol_stride_text.grid(row=0, column=2, padx=(self.paddingSmall,self.paddingSmall/2))

        self.convolveStride = tk.Entry(self.convolveSettingsFrame, validate='key', width=self.numberEntrySize)
        self.convolveStride.grid(row=0, column=3)
        self.convolveStride.delete(0,'end')
        self.convolveStride.insert(0,self.defaultConvolveStride)

        reg = self.convolveSettingsFrame.register(self.isanInteger)
        self.convolvePadding['validatecommand'] = (reg,'%P')
        self.convolveStride['validatecommand'] = (reg,'%P')

        self.convolve_bttn = tk.Button(self.convolveSettingsFrame, text="Convolve", command=lambda: self.parseConvolve(normalize=False))
        self.convolve_bttn['font'] = self.smallFont
        self.convolve_bttn.grid(row=1, column=0, columnspan=2, padx=self.paddingSmall, pady=self.paddingSmall)

        self.convolve_bttn2 = tk.Button(self.convolveSettingsFrame, text="Norm+Conv", command=lambda: self.parseConvolve())
        self.convolve_bttn2['font'] = self.smallFont
        self.convolve_bttn2.grid(row=1, column=2, columnspan=2, padx=self.paddingSmall, pady=self.paddingSmall)

    def isanInteger(self, value):
        try:
            int(value)
            return True
        except ValueError:
            if value=='' or value=='-':
                return True
            return False

    def parseConvolve(self, normalize=True):
        if self.missingKernelFieldsExist() or self.missingConvolveFieldsExist():
            self.error("An input field is missing")
            return
        try:
            padding = int(self.convolvePadding.get())
            stride = int(self.convolveStride.get())
            if padding<0 or padding>self.getMaxConvolPadding():
                self.error("Padding must be from 0 to {} (half the size of your smallest kernel)".format(int(self.getMaxConvolPadding())))
                return
            if stride<=0 or stride>self.calculateMaxStride():
                self.error("Stride must be less or equal to the smallest kernel dimension minus 1")
                return
        except ValueError:
            self.error("Kernel size must be an integer")
            return
        self.error("")

        
        if padding != 0:
            image_shape = self.imageMatrix.shape
            paddedImg = numpy.zeros((image_shape[0] + padding*2, image_shape[1] + padding*2, 3))
            for layer in range(self.imageMatrix.shape[2]):
                layerPadded = numpy.zeros((image_shape[0] + padding*2, image_shape[1] + padding*2))
                layerPadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = self.imageMatrix[:,:,layer]
                paddedImg[:,:,layer] = layerPadded
            self.imageMatrix = paddedImg
        
        for kernel in self.kernels:
            if normalize:
                kernel = self.normalizeKernel(kernel=kernel)
            newImg = self.convolve(kernel, padding, stride)

            self.transition(newImg, self.transitionDuration, self.transitionCurve)
    
    def convolve(self, kernel, padding=0, stride=1):
        newImg = self.imageMatrix.copy()

        layers = self.getSelectedLayers()
        layersIdx = [self.layers.index(i) for i in layers]
        if len(layersIdx)==0:
            self.error("Select the RGB layers to apply filter on")
        for layer in layersIdx:
            newImg[:,:,layer] = ndimage.convolve(newImg[:,:,layer].astype('float32'), kernel.astype('float32'), mode='nearest')
        
        newIdenticalLayers = copy.deepcopy(self.identicalLayers)
        #fix for some random addressing problem in the array that i have no idea why it's happening
        for key in newIdenticalLayers:
            newIdenticalLayers[key] = [newIdenticalLayers[key][idx] for idx in range(len(newIdenticalLayers[key]))]

        layersConvolved = layers
        if not (len(self.identicalLayers['red'])==3 and len(layersConvolved)==3):  #if they are both 3, then we are convolving all layers and they are already identical. skip
            for layer in layersConvolved:
                identical = list(set(self.identicalLayers[layer])-{layer})
                for l in identical:
                    if l not in layersConvolved:
                        newIdenticalLayers[l].remove(layer)
                        newIdenticalLayers[layer].remove(l)
            
            self.resetIdenticalLayers(identical=newIdenticalLayers)
        

        return newImg.astype('uint8')


    def calculateMaxStride(self):
        minDimensionSize = float('inf')
        for kernel in self.kernels:
            shape = kernel.shape
            if min(shape)<minDimensionSize:
                minDimensionSize = min(shape)
        return minDimensionSize-1
    
    def missingConvolveFieldsExist(self):
        field1 = self.convolvePadding.get()
        field2 = self.convolveStride.get()
        if field1=='' or field2=='':
            return True
        return False

    def packKernelWidgets(self, parent):
        self.kernelSizeFrame = tk.Frame(parent)
        self.kernelSizeFrame.grid(row=0, column=0, sticky='w', pady=self.paddingSmall/2)
        self.kernelGridFrame = tk.Frame(parent)
        self.kernelGridFrame.grid(row=2, column=0, pady=self.paddingSmall/2)

        self.packKernelSizeWidgts(self.kernelSizeFrame)
    
    def packKernelSizeWidgts(self, parent):
        self.kernelTitle = tk.Frame(parent)
        self.kernelTitle.grid(row=0, column=0, sticky='nw')
        kernel_text = tk.Label(self.kernelTitle, text="Kernel", font=self.titleFont)
        kernel_text.grid(row=0, column=0, sticky='nw')

        self.kernelSize = tk.Frame(parent)
        self.kernelSize.grid(row=1, column=0, sticky='nw')
        kernel_size_text = tk.Label(self.kernelSize, text="Size:", font=self.normalFont)
        kernel_size_text.grid(row=0, column=0, padx=(0,self.paddingSmall))

        self.kernel_num_rows = tk.Entry(self.kernelSize, validate='key', width=self.numberEntrySize)
        self.kernel_num_rows.grid(row=0, column=1)
        
        timesSymbol = tk.Label(self.kernelSize, text="×", font=self.boldSmallFont)
        timesSymbol.grid(row=0, column=2, padx=self.paddingSmall/4)
        
        self.kernel_num_cols = tk.Entry(self.kernelSize, validate='key', width=self.numberEntrySize)
        self.kernel_num_cols.grid(row=0, column=3)

        reg = self.kernelSize.register(self.isanInteger)
        self.kernel_num_rows['validatecommand'] = (reg,'%P')
        self.kernel_num_cols['validatecommand'] = (reg,'%P')
        
        self.add_empty_kernel_bttn = tk.Button(self.kernelSize, text="🡆", command=lambda: self.addKernel(size=(self.kernel_num_rows.get(), self.kernel_num_cols.get())))
        self.add_empty_kernel_bttn['font'] = self.normalFont
        self.add_empty_kernel_bttn.grid(row=0, column=4, padx=self.paddingSmall)

        self.presetFiltersFrame = tk.Frame(parent)
        self.presetFiltersFrame.grid(row=2, column=0, pady=self.paddingSmall/4, sticky='nw')

        presetFiltersText = tk.Label(self.presetFiltersFrame, text="Preset filters: ", font=self.normalFont)
        presetFiltersText.grid(row=0, column=0, padx=(0,self.paddingSmall))

        self.selectedPreset = tk.StringVar(self.presetFiltersFrame)
        self.presetOptions = tk.OptionMenu(self.presetFiltersFrame, self.selectedPreset, *self.presetKernels.keys(), command=lambda name: self.addKernel(kernel=self.presetKernels[name]))
        self.presetOptions.grid(row=0, column=1)

    def addKernel(self, kernel=None, size=None):
        if len(self.kernels)>=self.maxKernelCount:
            self.error("You can only have a maximum of {} kernels at once".format(self.maxKernelCount))
            return
        if size is None:
            self.kernels.append(kernel)
        else:
            numRows = size[0]
            numCols = size[1]
            try:
                numRows = int(numRows)
                numCols = int(numCols)
                if numRows<=0 or numCols<=0 or numRows>self.maxKernelSize[0] or numCols>self.maxKernelSize[1]:
                    self.error("Kernel size must be in the given range [1,{}] x [1,{}]".format(self.maxKernelSize[0], self.maxKernelSize[1]))
                    return
            except ValueError:
                self.error("Kernel size must be an integer")
                return
            self.error("")

            self.kernels.append(numpy.zeros((numRows, numCols)))
        
        self.packKernelGridWidgets(self.kernelGridFrame)


    def error(self, message=None):
        self.error_msg_var.set(message)
        if message!="":
            self.errorBlink()

    def errorExists(self):
        exists = self.error_msg_var
        return exists!=""

    def errorBlink(self):
        delay = 100
        for i in range(0,4,2):
            self.error_msg_label.after(delay*i, lambda: self.changeErrorColor('black'))
            self.error_msg_label.after(delay*(i+1), lambda: self.changeErrorColor('red'))
    
    def changeErrorColor(self, color):
        self.error_msg_label.config(fg=color)
        self.root.update()

    def packKernelGridWidgets(self, parent):
        for child in parent.winfo_children():
            child.grid_forget()
            child.destroy()
        for kernelIdx in range(len(self.kernels)):
            kernelFrame = tk.Frame(parent)
            kernelFrame.grid(row=0, column=kernelIdx, padx=self.paddingMedium)

            kernelInputGridFrame = tk.Frame(kernelFrame)
            kernelInputGridFrame.grid(row=0, column=0)
            self.packKernelInputGridWidgets(kernelInputGridFrame, kernelIdx)

            kernelButtonsFrame = tk.Frame(kernelFrame)
            kernelButtonsFrame.grid(row=1, column=0)

            remove_kernel_bttn = tk.Button(kernelButtonsFrame, text="x", command=lambda kernelIdx=kernelIdx: self.removeKernel(kernelIdx))
            remove_kernel_bttn['font'] = self.boldNormalFont
            remove_kernel_bttn.grid(row=0, column=0, padx=self.paddingSmall/4)

            norm_kernel_bttn = tk.Button(kernelButtonsFrame, text="Norm", command=lambda idx=kernelIdx: self.normalizeKernel(kernelIdx=idx))
            norm_kernel_bttn['font'] = self.boldNormalFont
            norm_kernel_bttn.grid(row=0, column=1, padx=self.paddingSmall/4)
    
    def packKernelInputGridWidgets(self, parent, kernelIdx):
        kernel = self.kernels[kernelIdx]
        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                inputField = tk.Entry(parent, width=4, validate='key')
                inputField.grid(row=row, column=col, padx=self.paddingSmall/4, pady=self.paddingSmall/4)
                
                kernVal = kernel[row,col]

                #set to kernel value
                inputField.delete(0,'end')
                if numpy.isnan(kernVal):
                    pass
                elif kernVal==int(kernVal):
                    inputField.insert(0,int(kernVal))
                else:
                    inputField.insert(0,kernVal)

                #define validation
                reg = inputField.register(lambda value, kernelIdx=kernelIdx, row=row, col=col: self.validateKernelInput(value, (row,col), kernelIdx))
                inputField['validatecommand'] = (reg,'%P')
    
    def validateKernelInput(self, value, valueIdx, kernelIdx):
        try:
            value = float(value)
            self.setKernelIndexValue(kernelIdx, valueIdx, value)
            return True
        except ValueError:
            if value=='' or value=='-':
                self.setKernelIndexValue(kernelIdx, valueIdx, numpy.nan)
                return True
            return False
    
    def missingKernelFieldsExist(self, kernelIdx=None):
        if len(self.kernels)==0:
            return True
        
        if kernelIdx is None:
            kernels = self.kernels
        else:
            kernels = [self.kernels[kernelIdx]]

        for kernel in kernels:
            if numpy.isnan(kernel).any():
                return True
        return False
    
    def setKernelIndexValue(self, kernelIdx, valueIdx, value):
        self.kernels[kernelIdx][valueIdx[0],valueIdx[1]] = value

    def removeKernel(self, kernelIdx):
        self.kernels.pop(kernelIdx)
        self.packKernelGridWidgets(self.kernelGridFrame)
    
    def normalizeKernel(self, kernelIdx=None, kernel=None):
        if kernelIdx is None and kernel is None:
            return
        elif kernelIdx is not None:
            if self.missingKernelFieldsExist(kernelIdx):
                self.error("An input field is missing")
                return
            sum = numpy.sum(numpy.absolute(self.kernels[kernelIdx]))
            if sum!=0:
                self.kernels[kernelIdx] = self.kernels[kernelIdx]/sum
                self.packKernelGridWidgets(self.kernelGridFrame)
        elif kernel is not None:
            sum = numpy.sum(numpy.absolute(kernel))
            if sum!=0:
                kernel = kernel/sum
                return kernel
        return


    def loadImage(self, filePath=None):
        if filePath is None:
            extensionsFormatted = "*."+" *.".join([ext for ext in self.supportedFileTypes])
            filePath = tk.filedialog.askopenfilename(filetypes=[("Image File",extensionsFormatted)])
            if filePath is None or filePath=='':
                return
            self.imagePath = filePath
        try:
            image = cv2.imread(filePath)
            if image is None:
                raise Exception
            self.imageMatrix = image
        except Exception:
            self.error("File does not exist")
            return
        self.imageMatrix = cv2.cvtColor(self.imageMatrix, cv2.COLOR_BGR2RGB)
        self.layers = ['red','green','blue']
        self.resetIdenticalLayers()
        self.updateLayersListBox()
        self.drawCanvas()
    
    def saveImage(self):
        if self.imageMatrix.size==0:
            self.error("No image on display")
            return
                
        imageName = os.path.basename(self.imagePath)
        imageExtension = ""
        for fileType in self.supportedFileTypes:
            if imageName.endswith(".{}".format(fileType)):
                imageName = imageName[:-(len(fileType)+1)]
                imageExtension = fileType
                break
        imageName = "{}-modified.{}".format(imageName, imageExtension)
        
        currentImgDir = os.path.dirname(os.path.abspath(self.imagePath))
        fileSaveFolder = tk.filedialog.askdirectory(initialdir=currentImgDir)
        imagePath = os.path.join(fileSaveFolder, imageName)
        cv2.imwrite(imagePath, self.imageMatrix)
        

    def flattenLayers(self,layersToCombine):
        if len(layersToCombine)<=1:
            self.clearLayersSelection()
            return
        if len(self.identicalLayers['red'])==3: #that means that all three (red, green and blue) are equal, so flattening does nothing
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
        
        self.resetIdenticalLayers(identical=newIdenticalLayers)
        
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
        
        self.resetIdenticalLayers(identical=self.identicalLayers)

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
