# nimslo auto-aligning gif processor

## 🎬 what it does
- processes nimslo 3d film shots into smooth animated gifs
- uses cnn-based border detection for precise alignment
- applies histogram matching for consistent exposure
- creates bounce-effect gifs with automatic quality optimization

## ✨ features
- **manual image selection** - pick and order your best 4-6 frames
- **cnn alignment** - sophisticated border detection and image subtraction
- **automatic cropping** - removes black bars from aligned images
- **quality settings** - high, medium, or optimized output
- **bounce effect** - smooth back-and-forth animation
- **streamlined workflow** - no confirmation dialogs, just results

## 🚀 quick start

### one-command setup
```bash
# from anywhere on your system
nimslo
```

that's it! the alias will:
1. activate the conda environment
2. run the processor with full gui
3. generate your nimslo gif automatically

### first-time setup
```bash
# clone and setup
git clone <repo-url>
cd nimslo_processing

# create environment and alias
conda env create -f environment.yml
echo 'alias nimslo="cd $(pwd) && ./run_nimslo.sh"' >> ~/.zshrc
source ~/.zshrc
```

## 🎯 workflow
1. **run `nimslo`** - launches the processor
2. **select folder** - choose your nimslo image directory
3. **pick images** - use the gui to select and order 4-6 frames
4. **crop area** - drag to select the region to align
5. **add reference points** - double-click to mark alignment points (optional but recommended)
6. **automatic processing** - intelligent alignment, histogram matching, gif creation
7. **output** - find your gif in `nimslo_gifs/`

## 📁 output
- **gifs** saved to `nimslo_gifs/nimslo_high_TIMESTAMP.gif`
- **previews** saved as `preview_original.png`, `preview_aligned.png`, `preview_final.png`
- **quality levels** indicated in filename

## 🔧 technical details
- **intelligent alignment** - reference point-based (photoshop-style) or cnn border detection
- **sift feature matching** - robust point correspondence detection
- **multi-transformation support** - homography (3+ points) or affine (2 points)
- **quality validation** - ssim, mse, and point error metrics for alignment assessment
- **fallback system** - graceful degradation when point matching fails
- **automatic cropping** - removes transformation artifacts
- **bounce animation** - forward + backward sequence for smooth motion

## 📦 dependencies
- tensorflow 2.16+ (for cnn alignment)
- opencv-python (computer vision + sift features)
- pillow (image processing)
- matplotlib (previews)
- tkinter (gui)
- scikit-image (quality metrics)

all managed via conda environment - just run `conda env create -f environment.yml`

---

**made with ❤️  for analog photography lovers**  
*turning nimslo film into digital magic* ✨