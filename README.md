### Pix2Pix Inference Demo

This repository contains a standalone demo separated from training and focused only on inference.

#### Files

- `demo_inference.py`: script to load a checkpoint and generate a prediction for one image.
- `baseline_pix2pix_cityscapes.pt`, `augmented_pix2pix_cityscapes.pt`, `label_smoothing_pix2pix_cityscapes.pt`: merge-style Pix2Pix checkpoints from this project.
- `pix2pixHD.pt`: Pix2PixHD checkpoint.
- `cityscapes/`: sample images for testing the demo.
- `results/`: outputs of the inference of the models.

#### Requirements

For this workspace it is recommended the creation of a virtual environment.

```powershell
python -m venv venv 
```

Activation:

```powershell
venv\Scripts\Activate.ps1  ## Activation in Powershell Windows
source venv/bin/activate  ## Activation in linux
```

If you need to install dependencies for the demo:

```powershell
pip install torch torchvision pillow matplotlib
python.exe -m pip install --upgrade pip ## Only if needed
```

#### Usage

The Cityscapes image in the repo has two halves. To use the right half as the conditioning input, enable `--paired-image`.

By default, outputs are saved in `results/` using a filename that includes the image name and the model identifier.

```powershell
python demo_inference.py cityscapes\image_006.jpg --checkpoint baseline_pix2pix_cityscapes.pt --paired-image
python demo_inference.py cityscapes\image_006.jpg --checkpoint augmented_pix2pix_cityscapes.pt --paired-image
python demo_inference.py cityscapes\image_006.jpg --checkpoint label_smoothing_pix2pix_cityscapes.pt --paired-image
python demo_inference.py cityscapes\image_006.jpg --checkpoint pix2pixHD.pt --paired-image
python demo_inference.py cityscapes\image_006.jpg --all --paired-image
```

By default, the script saves:

- a side-by-side comparison image with the input on the left and the output on the right
- with `--all`, one combined comparison image that includes the input and every local checkpoint output
- individual comparison images for each model under `results/`

#### Notes

- The demo uses inference only.
- The script automatically detects whether the checkpoint or Pix2Pix, or Pix2PixHD.
- To test another image, change the input path in the script.