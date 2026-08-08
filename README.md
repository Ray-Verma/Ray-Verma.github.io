# Ray Verma — GitHub Pages research site

Static GitHub Pages website with a Cells2Pixels-compatible interactive NCA portrait.

## Portrait behavior

The model uses a 512 x 512 target with 128 pixels of training padding on every side. At renderer scale 8, the full NCA arena is 96 x 96 cells and the full decoded field is 768 x 768 pixels. The site computes that entire padded arena but displays only the central 512 x 512 target region.

The live portrait also:

- preserves learned RGBA transparency;
- places `assets/nca-portrait-background.png` underneath the transparent NCA output;
- has an always-on erase brush: drag directly over the portrait to damage cells;
- resumes updates after damage so the morphology can recover;
- checks the full padded arena for a living cell before NCA updates and after erasing; if no cell has alpha above the GrowingNCA living threshold, the configured seed is injected automatically;
- uses the redo button in the bottom-right to restart from the configured seed;
- throttles only the expensive 512 x 512 SIREN visualization to every two NCA steps by default. NCA state updates themselves are unchanged;
- falls back to the static portrait on phones, reduced-motion, loading failure, or WebGL context loss.

## Local preview — use localhost, not file://

Modern browsers normally assign `file://` documents opaque origins. As a result, direct local-file testing can produce origin/CORS warnings even when files are beside each other. The reliable preview path is a local HTTP server.

### macOS

Double-click:

```text
START_LOCAL.command
```

A Terminal window opens, chooses a free localhost port, and opens the website in your default browser. Keep the Terminal window open while testing.

### Any platform with Python 3

```bash
python3 serve_local.py
```

The script prints and opens the localhost URL automatically.

Do **not** use Finder's "Open With" on `index.html` for NCA testing. GitHub Pages itself uses HTTPS, so deployment does not have this `file://` origin problem.

## Move the NCA seed

Edit these attributes on `#nca-portrait` in `index.html`:

```html
data-seed-x="0.50"
data-seed-y="0.50"
```

They are normalized coordinates inside the **visible 512 x 512 target region**, not the padded 768 x 768 field:

- `data-seed-x="0"` = left edge; `1` = right edge.
- `data-seed-y="0"` = top edge; `1` = bottom edge.
- `0.50, 0.50` = center.

For example, to move the generated foreground a little left and down:

```html
data-seed-x="0.47"
data-seed-y="0.53"
```

The NCA grid is 64 visible cells across the 512 px crop, so seed movement is quantized to roughly 8 rendered pixels per cell. Changes of about `1 / 64 = 0.0156` generally correspond to one visible NCA-cell shift.

The full 128-pixel padding is still computed around the target regardless of seed location.

## Align/crop the background image

The current background is configured with:

```html
data-background="assets/nca-portrait-background.png"
data-background-scale="1.00"
data-background-offset-x="0"
data-background-offset-y="0"
```

Use the last three values to align it with the NCA foreground:

- `data-background-scale="1.05"` zooms the background by 5% and crops the excess.
- `data-background-offset-x="-2"` moves the background left by 2% of the portrait frame; positive values move it right.
- `data-background-offset-y="1.5"` moves the background down by 1.5%; negative values move it up.

A typical tuning sequence is: first adjust the NCA seed so the generated person is positioned correctly, then adjust background scale, then fine-tune background x/y offsets.

## Rendering load

`data-decode-every="2"` controls only how often the 512 x 512 SIREN image is refreshed while the NCA is rolling out. Increase it to `3` or `4` on a machine/GPU that struggles. This does not change the NCA state transition; it only displays fewer intermediate decoded frames.

## Change the model

```bash
python training/convert_model_to_json.py \
  --model /path/to/model_8.pth \
  --siren /path/to/siren_8.pth \
  --output assets/nca/profile.json
```

See `training/README.md` for details.

## Deploy

Create a repository named `YOUR-USERNAME.github.io`, place these files at its root, push to `main`, and enable GitHub Pages under **Settings -> Pages -> Deploy from a branch -> main -> /(root)**.


## NCA display scaling and background alignment

The checkpoint geometry is fixed by training: **512x512 target + 128px padding on every side + training renderer scale 8 = 96x96 recurrent NCA cells**. The website now keeps that geometry fixed regardless of display settings.

### Safe NCA display resolution

In `index.html`:

```html
data-renderer-scale="8"
```

This setting now controls only the **decoded display texture resolution**. It does not change the NCA arena or padding crop.

- `4` -> 64 visible cells x 4 = 256x256 decoded texture
- `6.5` -> 416x416 decoded texture
- `8` -> 512x512 decoded texture (training resolution)
- `12` -> 768x768 decoded texture
- `16` -> 1024x1024 decoded texture (maximum accepted value)

The result is still CSS-scaled to fill the same square portrait region. Increasing this value increases GPU work; `8` is the best default for this checkpoint.

### Background crop / resize variables

The live NCA remains transparent and is drawn above `data-background`. These variables control that background independently:

```html
data-background-fit="cover"
data-background-zoom="1.00"
data-background-position-x="50"
data-background-position-y="50"
data-background-scale-x="1.00"
data-background-scale-y="1.00"
data-background-offset-x="0"
data-background-offset-y="0"
```

- **`data-background-fit`**: `cover`, `contain`, `fill`, `none`, or `scale-down`. `cover` is recommended.
- **`data-background-zoom`**: uniform crop/zoom. `1.10` zooms in 10%; `0.95` zooms out 5%.
- **`data-background-position-x` / `-y`**: chooses which part of the source image remains centered when it is cropped. Values are percentages from `0` to `100`. Start at `50/50`.
- **`data-background-scale-x` / `-y`**: independent horizontal/vertical resize. Keep at `1.00` unless the generated background needs slight perspective/alignment correction.
- **`data-background-offset-x` / `-y`**: final translation in percent after fit/zoom. Positive X moves right; positive Y moves down.

Example: zoom the background 6%, favor the left side of the source, and shift it slightly upward:

```html
data-background-fit="cover"
data-background-zoom="1.06"
data-background-position-x="45"
data-background-position-y="50"
data-background-scale-x="1.00"
data-background-scale-y="1.00"
data-background-offset-x="0"
data-background-offset-y="-1.5"
```

The automatic empty-arena test uses a two-stage parallel GPU reduction, so it no longer compiles a single shader loop over all 9,216 NCA cells. If the entire 96x96 arena has no alpha above `0.1`, the configured seed is injected automatically.

## Academic homepage structure

The homepage is organized as:

- About Me
- Ongoing Projects: NavNCA, GoalNCA, and Tabular Foundation Models
- Publications
- Experience
- Contact

The sidebar contains the portrait, name, NYU affiliation, and Email/CV/LinkedIn links.

### Publication images

Each publication currently contains a `.publication-image-placeholder` block in `index.html`.
When a paper figure is ready, replace that block with an image, for example:

```html
<img class="publication-image" src="assets/publications/my-paper.png" alt="Brief description of the paper figure">
```

Then add this rule to `styles.css` if you want the same geometry as the placeholder:

```css
.publication-image {
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  border-radius: .65rem;
}
```
