# Image Blurring: Spatial Filtering vs. Frequency-Domain Filtering

Implements and validates the **Convolution Theorem** by blurring images two
independent ways — direct convolution in the spatial domain, and Fourier
transform multiplication in the frequency domain — and showing the results
are numerically identical.

> f(x,y) \* h(x,y)  ⟺  F(u,v) · H(u,v)
---

## What's in this repo

| File | Description |
|---|---|
| `image_blur_analysis.py` | Command-line script. Runs both blurring methods on a chosen image, prints MSE/max error between them, and saves comparison figures. |
| `app.py` | Streamlit web application — the same logic with an interactive UI (upload an image, pick a kernel, see results live). |
| `requirements.txt` | Python dependencies for both scripts. |

---

## How it works

1. **Spatial-domain blur** — `scipy.ndimage.convolve(image, kernel, mode='wrap')` slides the kernel over the image directly, with periodic (wrap-around) boundary handling.
2. **Frequency-domain blur** — the kernel is zero-padded to the image's size and circularly shifted so its center sits at index `(0,0)`, then:
   ```
   F = FFT2(image)
   H = FFT2(padded_kernel)
   result = Re(IFFT2(F * H))
   ```
3. **Validation** — the two results are compared pixel-by-pixel via Mean Squared Error (MSE) and max absolute error. Both come out on the order of `1e-27`/`1e-13` — floating-point noise only, confirming the two methods are mathematically the same operation.

The wrap-around boundary in step 1 is what makes the two methods match exactly: the Convolution Theorem holds precisely for *circular* convolution, not standard zero-padded linear convolution.

---

## Requirements

- Python 3.10+
- Packages: `numpy`, `scipy`, `matplotlib`, `pillow`, `scikit-image`, `streamlit` (only needed for the web app)

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Command-line script

```bash
# Built-in synthetic checkerboard pattern (default)
python image_blur_analysis.py --kernel gaussian --size 15 --sigma 3.0

# Built-in standard test images
python image_blur_analysis.py --sample cameraman --kernel box --size 9
python image_blur_analysis.py --sample baboon --kernel gaussian --size 21 --sigma 5

# Your own image
python image_blur_analysis.py --image path/to/photo.jpg --kernel gaussian --size 21 --sigma 5
```

**Output** (written to `./output/`): `01_original.png`, `02_blur_spatial.png`, `03_blur_frequency.png`, `04_difference_map.png`, `comparison_figure.png`, plus console printout of MSE and max error.

### 2. Web app (Streamlit)

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`. Pick a sample image or upload your own, choose a kernel type/size, and click **Run comparison** to see the original, spatial blur, frequency blur, and difference map side by side, along with the MSE/max-error readout.

---

## Theory

Full derivation of the Convolution Theorem (continuous-domain proof via Fubini's theorem and the Fourier Shift Theorem, plus the discrete/circular-convolution case) is included in the project report — see `theory_convolution_theorem.md` / the submitted PDF.

---
