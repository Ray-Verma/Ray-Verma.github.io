# Third-party references

The interactive portrait follows the Cells2Pixels GrowingNCA/SIREN architecture and the interaction pattern of its public growing demo:

- https://github.com/TheDevilWillBeBee/Cells2Pixels
- https://github.com/Cells2Pixels/Cells2Pixels.github.io
- https://cells2pixels.github.io/#growing

The page loads SwissGL at runtime from the Cells2Pixels web-demo repository via jsDelivr. If it cannot load, the page uses the static portrait fallback.

The website-specific runtime preserves the training repo's 96x96 full NCA state for a 512 target + 128 padding, but fixes the visible view to the exact central 512x512 target window. The eraser follows the public demo's circular erase rule and is always enabled on the live portrait.
