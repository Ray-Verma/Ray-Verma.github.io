# Cells2Pixels portrait model

## Convert checkpoints

```bash
pip install torch
python training/convert_model_to_json.py \
  --model /path/to/model_8.pth \
  --siren /path/to/siren_8.pth \
  --output assets/nca/profile.json
```

`--siren` may be omitted when a matching `siren_8.pth`/`.pt` is beside `model_8.pth`.
The converter emits both `profile.json` and `profile.js`; the JS wrapper permits direct `file://` preview, while normal HTTP/GitHub Pages uses the JSON file.

The converter includes the important Cells2Pixels perception-order permutation needed by the SwissGL runtime.

## Display geometry

The supplied growing model uses:

- target image: 512 x 512
- padding: 128 pixels on every side
- renderer scale: 8
- full rendered field: 768 x 768
- NCA grid: 96 x 96 cells
- visible content window: central 64 x 64 cells

The web runtime evolves the entire 96 x 96 grid, including all 16 padding cells on every side, but decodes only the central 64 x 64 cells into the visible 512 x 512 portrait. This removes the training padding from the webpage without changing the NCA dynamics.

## Eraser

Erase mode is always enabled in live desktop mode. Drag the pointer directly over the portrait. The brush follows the same mapping used by the Cells2Pixels growing demo, but its view radius is fixed to exactly `512 / 768 = 2/3`, which corresponds to the unpadded target.

The default brush diameter is 10% of the visible portrait. Change `data-brush-size` in `index.html` to scale it. After an erase stroke, the runtime performs 64 additional NCA updates so the morphology can recover.

## Optional background image

Set this attribute in `index.html`:

```html
data-background="assets/nca-background.jpg"
```

The image sits under the transparent NCA canvas. Leave the attribute empty for the normal website background.
