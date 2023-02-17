# ENCHANTED IMAGE SHRINKER V2.0 :framed_picture:

A terminal program to change image ratio with using Seam Carving. There are two different shortest path algorithms which are optimized first-level greedy and Dijkstra.

## Usage of program

- Create an RetargetImage object.
   <br>*eg:*
   > exampleObject = RetargetedImage("[IMAGE FULL DIRECTORY]")
- Determine how many pixels you want to retarget with using ImageRetarget method.
   <br>*eg:*
   > ImageRetarget(exampleObject, [PIXEL NUMBER], [RETARGET ALGORITHM])
   > <br>Retarget Algorithm:  "g" or "greedy" for Greedy, "d" or "dijkstra" for Dijkstra
- Show images you wanted. See below for the methods.
  > showRetargetedImage() = It shows the final product which is retargeted image
  <br>showOriginalImage() = It shows original image
  <br>showGrayImage() = It shows gray level image ('gray' color map will be applied)
  <br>showLaplacianImage() = It shows Laplacian transformed image (3D Array with RGB channels)
  <br>showLaplacianImage2D() = It shows Laplacian transformed image (2D Array with one channel)
  <br>showMarkedLaplacianImage() = It shows marked curves for retargeting on the Laplacian transformed image (2D)
  <br>showMarkedImage() = It shows marked curves for retargeting on the original image
- If you prefer to save the image, give "save" or "-s" parameters to these methods and give a file name as second parameters. It will be saved in "saved" folder.
   <br>*eg:*
   > showRetargetedImage("save", "Retargeted Image")
   <br>showMarkedImage("-s", "Retargeting Curves")
- If you want to skip showing part and force algorithm to save directly instead, use "fSave"
   <br>*eg:*
   > showRetargetedImage("fSave", "Retargeted Image")
