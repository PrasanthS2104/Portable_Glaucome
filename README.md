# Portable Glaucoma

This repository contains a portable glaucoma detection prototype demo and related materials.

## Run the glaucoma detector (Streamlit)

1. Install dependencies:

   ```bash
   cd glaucoma_detector
   pip install -r requirements.txt
   ```

2. Ensure the model weight file is present:

   * `glaucoma_detector/inception_resnet_v2_glaucoma.h5`

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. In the browser, upload an eye fundus image (`.jpg`, `.jpeg`, or `.png`) and view the prediction.

## Folder structure

```text
janesha_project/
  glaucoma_detector/
    app.py                         # Streamlit UI + model loading + inference
    requirements.txt               # Python dependencies
    bg.jpg                          # UI background asset
    inception_resnet_v2_glaucoma.h5  # Model weights (Git LFS)
  opencv/
    file.py                         # Small OpenCV webcam test script
  Glaucoma_Detection/
    glaucoma.csv                    # Dataset metadata CSV
    ACRIMA/
      Images/                       # Raw images (excluded from Git push; see note below)
    Dataset/
      Fundus_Scanes_Sorted/       # Raw train/validation image sets (excluded)
    ORIGA/
      ORIGA/
        Images/                    # Raw images (excluded)
        Semi-automatic-annotations/ # (excluded if large)
        *.mat, *.txt, *.m         # Smaller annotation files
 
```

## Large files (Git LFS)

This repo uses **Git LFS** for large binary artifacts, including:

* `*.h5`
* `*.zip`

If you clone the repo on a new machine, install Git LFS first:

```bash
git lfs install
git lfs pull
```

## Dataset note

Raw dataset image directories can be very large, so the repo excludes the image folders via `.gitignore`.
If you need to reproduce training or segmentation steps, download the datasets externally and place them into the corresponding paths under `Glaucoma_Detection/`.

