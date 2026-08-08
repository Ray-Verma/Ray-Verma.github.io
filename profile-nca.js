(() => {
  'use strict';

  const root = document.getElementById('nca-portrait');
  if (!root) return;

  const canvas = document.getElementById('nca-canvas');
  const loading = document.getElementById('nca-loading');
  const staticImage = document.getElementById('nca-static');
  const backgroundImage = document.getElementById('nca-background');
  const redo = document.getElementById('nca-redo');
  const brushCursor = document.getElementById('nca-brush-cursor');
  const caption = document.getElementById('nca-caption');

  const MODEL_URL = root.dataset.model || 'assets/nca/profile.json';
  const STATIC_URL = root.dataset.static || 'assets/profile-static.png';
  const BACKGROUND_URL = root.dataset.background || '';
  const SWISSGL_URLS = [
    'https://cells2pixels.github.io/swissgl.js',
    'https://cdn.jsdelivr.net/gh/Cells2Pixels/Cells2Pixels.github.io@main/swissgl.js'
  ];
  const LOAD_TIMEOUT_MS = 10000;

  // -----------------------------------------------------------------------
  // Geometry: keep TRAINING geometry separate from DISPLAY resolution.
  // -----------------------------------------------------------------------
  // This checkpoint was trained with a 512x512 target, 128px padding on all
  // sides, and renderer scale 8. Those values define the recurrent NCA arena
  // and MUST remain fixed for this model: (512 + 2*128) / 8 = 96 cells.
  //
  // data-renderer-scale below is intentionally *display only*. You may set it
  // to 4, 6.5, 8, 10, 12, ... without changing H/W or producing non-integer
  // GLSL loop bounds. A larger value renders a higher-resolution SIREN texture;
  // CSS still scales that texture to the same portrait box.
  const TARGET_PX = 512;
  const PADDING_PX = 128;
  const TRAIN_RENDERER_SCALE = 8;
  const TOTAL_PX = TARGET_PX + 2 * PADDING_PX;
  const H = TOTAL_PX / TRAIN_RENDERER_SCALE;
  const W = H;
  const CONTENT_CELLS = TARGET_PX / TRAIN_RENDERER_SCALE;
  const PAD_CELLS = PADDING_PX / TRAIN_RENDERER_SCALE;
  const VIEW_R = TARGET_PX / TOTAL_PX; // 512 / 768 = 2/3
  const VIEW_C = [0.5, 0.5];

  const requestedDisplayScale = Number.parseFloat(root.dataset.rendererScale || '8');
  const DISPLAY_RENDERER_SCALE = Number.isFinite(requestedDisplayScale)
    ? Math.min(16, Math.max(1, requestedDisplayScale))
    : 8;
  const DISPLAY_OUTPUT_PX = Math.max(64, Math.round(CONTENT_CELLS * DISPLAY_RENDERER_SCALE));

  const CHN = 32;
  const C4 = CHN / 4;
  const D4 = 64 / 4;
  const INITIAL_STEPS = Number.parseInt(root.dataset.steps || '96', 10);
  const RECOVERY_STEPS = Number.parseInt(root.dataset.recoverySteps || '64', 10);
  const BRUSH_SIZE = Number.parseFloat(root.dataset.brushSize || '1.0');
  const DECODE_EVERY = Math.max(1, Number.parseInt(root.dataset.decodeEvery || '2', 10));

  // Seed coordinates are normalized inside the visible 512x512 target crop:
  // x=0 is left, x=1 is right; y=0 is top, y=1 is bottom. They are mapped
  // back into the full padded 96x96 NCA arena. The NCA is cell-based, so
  // positions are quantized in 8-pixel increments for the default model.
  const SEED_X = Math.min(1, Math.max(0, Number.parseFloat(root.dataset.seedX || '0.50')));
  const SEED_Y = Math.min(1, Math.max(0, Number.parseFloat(root.dataset.seedY || '0.50')));
  const SEED_CELL_X = Math.round(PAD_CELLS + SEED_X * (CONTENT_CELLS - 1));
  const SEED_CELL_Y = Math.round(PAD_CELLS + SEED_Y * (CONTENT_CELLS - 1));

  // -----------------------------------------------------------------------
  // Background crop / resize controls.
  // -----------------------------------------------------------------------
  // fit: cover | contain | fill | none | scale-down
  // zoom: uniform zoom; >1 crops more of the image, <1 reveals more.
  // scale-x/y: independent resize after zoom (use 1 unless alignment needs it).
  // position-x/y: object-position percentages selecting the crop anchor.
  // offset-x/y: final translation in percentages of the displayed image.
  const VALID_BACKGROUND_FITS = new Set(['cover', 'contain', 'fill', 'none', 'scale-down']);
  const requestedBackgroundFit = (root.dataset.backgroundFit || 'cover').toLowerCase();
  const BACKGROUND_FIT = VALID_BACKGROUND_FITS.has(requestedBackgroundFit) ? requestedBackgroundFit : 'cover';

  function finiteNumber(value, fallback) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  const BACKGROUND_ZOOM = Math.min(5, Math.max(0.1, finiteNumber(
    root.dataset.backgroundZoom ?? root.dataset.backgroundScale, 1.0
  )));
  const BACKGROUND_SCALE_X = Math.min(5, Math.max(0.1, finiteNumber(root.dataset.backgroundScaleX, 1.0)));
  const BACKGROUND_SCALE_Y = Math.min(5, Math.max(0.1, finiteNumber(root.dataset.backgroundScaleY, 1.0)));
  const BACKGROUND_POSITION_X = Math.min(100, Math.max(0, finiteNumber(root.dataset.backgroundPositionX, 50)));
  const BACKGROUND_POSITION_Y = Math.min(100, Math.max(0, finiteNumber(root.dataset.backgroundPositionY, 50)));
  const BACKGROUND_OFFSET_X = finiteNumber(root.dataset.backgroundOffsetX, 0);
  const BACKGROUND_OFFSET_Y = finiteNumber(root.dataset.backgroundOffsetY, 0);

  const TRAINING_GEOMETRY_VALID = Number.isInteger(H) && Number.isInteger(CONTENT_CELLS) && H === 96 && CONTENT_CELLS === 64;

  let cancelled = false;
  let animationId = 0;
  let glsl;
  let model;
  let ncaGrid;
  let sirenGrid;
  let frameCount = 0;
  let totalStepCount = 0;
  let recoveryStepsRemaining = 0;
  let decoderDirty = true;
  let lastDecodedStep = -1;
  let pointerDown = false;

  const IS_WEB_ORIGIN = location.protocol === 'https:' || location.protocol === 'http:';

  function resolveResourceURL(value, label, sameOrigin = false) {
    const url = new URL(value, document.baseURI);

    // A page served by GitHub Pages must never attempt to load a file:// URL.
    // If an extension or malformed configuration rewrites an asset URL, fail
    // here with a useful message instead of allowing a confusing browser-level
    // security exception.
    if (IS_WEB_ORIGIN && url.protocol !== 'https:' && url.protocol !== 'http:') {
      throw new Error(`${label} resolved to disallowed protocol ${url.protocol}`);
    }
    if (sameOrigin && IS_WEB_ORIGIN && url.origin !== location.origin) {
      throw new Error(`${label} must be served from ${location.origin}, got ${url.origin}`);
    }
    return url.href;
  }

  const isPhone = (() => {
    const uaPhone = /iPhone|iPod|Windows Phone|Android.+Mobile|Mobile.+Firefox/i.test(navigator.userAgent);
    const smallTouch = window.matchMedia('(max-width: 760px) and (pointer: coarse)').matches;
    return uaPhone || smallTouch;
  })();

  function showStatic(reason) {
    if (cancelled) return;
    cancelled = true;
    if (animationId) cancelAnimationFrame(animationId);
    loading.hidden = true;
    canvas.hidden = true;
    redo.hidden = true;
    brushCursor.hidden = true;
    caption.hidden = true;
    backgroundImage.hidden = true;
    staticImage.hidden = false;
    root.dataset.mode = `static-${reason}`;
  }

  function showLive() {
    loading.hidden = true;
    staticImage.hidden = true;
    canvas.hidden = false;
    redo.hidden = false;

    if (BACKGROUND_URL) {
      backgroundImage.src = resolveResourceURL(BACKGROUND_URL, 'NCA background', true);
      backgroundImage.style.objectFit = BACKGROUND_FIT;
      backgroundImage.style.objectPosition = `${BACKGROUND_POSITION_X}% ${BACKGROUND_POSITION_Y}%`;
      backgroundImage.style.transformOrigin = `${BACKGROUND_POSITION_X}% ${BACKGROUND_POSITION_Y}%`;
      backgroundImage.style.transform = `translate(${BACKGROUND_OFFSET_X}%, ${BACKGROUND_OFFSET_Y}%) scale(${BACKGROUND_ZOOM * BACKGROUND_SCALE_X}, ${BACKGROUND_ZOOM * BACKGROUND_SCALE_Y})`;
      backgroundImage.hidden = false;
      backgroundImage.onerror = () => { backgroundImage.hidden = true; };
    } else {
      backgroundImage.hidden = true;
    }

    caption.innerHTML = 'This portrait is grown in-browser by a <a href="https://cells2pixels.github.io/" target="_blank" rel="noreferrer">Neural Cellular Automata (NCA) model </a>. Drag to erase and watch the pixels regenerate purely using information from its neighbors; This is the inspiration for my <a href="#navnca">current swarm research</a>.';
    caption.hidden = false;
    root.dataset.mode = 'live';
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      let resolved;
      try {
        resolved = resolveResourceURL(src, 'script');
      } catch (error) {
        reject(error);
        return;
      }

      const script = document.createElement('script');
      script.src = resolved;
      script.async = true;
      script.referrerPolicy = 'no-referrer';
      script.onload = () => resolve(resolved);
      script.onerror = () => reject(new Error(`Failed to load script: ${resolved}`));
      document.head.appendChild(script);
    });
  }

  function decodePayload(src) {
    for (const key in src) {
      if (!Object.prototype.hasOwnProperty.call(src, key)) continue;
      const rec = src[key];
      if (!rec || rec.data64 === undefined) continue;
      const binary = atob(rec.data64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      rec.data = new Float32Array(bytes.buffer);
      delete rec.data64;
    }
    return src;
  }

  async function fetchModel() {
    if (!IS_WEB_ORIGIN) {
      throw new Error('Live NCA requires http:// or https://. Use a local web server instead of opening index.html with file://.');
    }

    const modelURL = resolveResourceURL(MODEL_URL, 'NCA model', true);
    let lastError;

    // Retry once. This helps with transient GitHub Pages/network failures while
    // keeping the model strictly same-origin.
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const requestURL = attempt === 0
          ? modelURL
          : `${modelURL}${modelURL.includes('?') ? '&' : '?'}retry=${Date.now()}`;
        const response = await fetch(requestURL, {
          method: 'GET',
          mode: 'same-origin',
          credentials: 'same-origin',
          cache: attempt === 0 ? 'default' : 'reload'
        });
        if (!response.ok) throw new Error(`Portrait model request failed (${response.status})`);
        return decodePayload(await response.json());
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error('Portrait model request failed');
  }

  async function ensureSwissGL() {
    if (window.SwissGL) return;

    let lastError;
    for (const src of SWISSGL_URLS) {
      try {
        await loadScript(src);
        if (window.SwissGL) return;
        lastError = new Error(`SwissGL loaded from ${src} but did not initialize`);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error('SwissGL did not initialize');
  }

  function buildModel(src) {
    const required = [
      'nca.w1.weight', 'nca.w1.bias', 'nca.w2.weight.T',
      'lppn.net.0.linear.weight', 'lppn.net.0.linear.bias',
      'lppn.net.1.linear.weight', 'lppn.net.1.linear.bias',
      'lppn.net.2.linear.weight', 'lppn.net.2.linear.bias',
      'lppn.net.3.weight', 'lppn.net.3.bias'
    ];
    for (const key of required) if (!src[key]) throw new Error(`Portrait model is missing ${key}`);

    const [hiddenChannels, inputChannels] = src['nca.w1.weight'].shape;
    const outputChannels = src['nca.w2.weight.T'].shape[1];
    if (outputChannels !== CHN) throw new Error(`Expected ${CHN} NCA channels, found ${outputChannels}`);

    const nca = {
      w1: glsl({}, { size: [inputChannels / 4, hiddenChannels], format: 'rgba32f', data: src['nca.w1.weight'].data, tag: 'profile_nca_w1' }),
      b1: glsl({}, { size: [1, hiddenChannels], format: 'r32f', data: src['nca.w1.bias'].data, tag: 'profile_nca_b1' }),
      w2t: glsl({}, { size: [outputChannels / 4, hiddenChannels], format: 'rgba32f', data: src['nca.w2.weight.T'].data, tag: 'profile_nca_w2t' })
    };

    const siren = {};
    const includes = [`const int C4=${C4}; const int D4=${D4};`];
    for (let i = 0; i < 4; i += 1) {
      const last = i === 3;
      const prefix = `lppn.net.${i}${last ? '' : '.linear'}`;
      const weight = src[`${prefix}.weight`];
      const bias = src[`${prefix}.bias`];
      const [no, ni] = weight.shape;
      siren[`w${i}s`] = glsl({}, { size: [ni / 4, no], format: 'rgba32f', data: weight.data, tag: `profile_lppn_w${i}` });
      siren[`b${i}s`] = glsl({}, { size: [1, no], format: 'r32f', data: bias.data, tag: `profile_lppn_b${i}` });
      includes.push(`
        void run_layer${i}(in vec4 src[D4], out vec4 dst[D4]) {
          const int no=${no}, ni=${ni};
          for (int j=0; j<no; ++j) {
            float a = b${i}s(ivec2(0, j)).x;
            #pragma unroll
            for (int k=0; k<ni/4; ++k) a += dot(src[k], w${i}s(ivec2(k, j)));
            dst[j/4][j%4] = ${last ? 'a' : 'sin(a*10.0)'};
          }
        }`);
    }
    siren.Inc = includes.join('\n');
    return { nca, siren };
  }

  function reset() {
    frameCount = 0;
    totalStepCount = 0;
    recoveryStepsRemaining = 0;
    sirenGrid = null;
    decoderDirty = true;
    lastDecodedStep = -1;
    ncaGrid = glsl({
      seed: 42,
      FP: `
        FOut = vec4(0.0);
        if (I.x == ${SEED_CELL_X} && I.y == ${SEED_CELL_Y}) {
          // GrowingNCA seed: visible RGB are zero, alpha=1, latent channels=1.
          FOut.w = 1.0;
          FOut1 = FOut2 = FOut3 = vec4(1.0);
          FOut4 = FOut5 = FOut6 = FOut7 = vec4(1.0);
        } else {
          FOut1 = FOut2 = FOut3 = vec4(0.0);
          FOut4 = FOut5 = FOut6 = FOut7 = vec4(0.0);
        }
      `
    }, {
      size: [H, W],
      layern: C4,
      format: 'rgba32f',
      story: 3,
      tag: 'profile_grid'
    });
  }

  function ensureArenaHasLivingCell() {
    if (!ncaGrid || cancelled) return;

    // Robust two-stage GPU reduction. The old implementation compiled one
    // fragment containing a loop over all 96*96 cells. That became fragile when
    // display scaling accidentally changed H/W and is also unnecessarily hard
    // on some GPU drivers. Instead, check 8x8 blocks in parallel (12x12 output),
    // then reduce those 144 block values in a tiny 1x1 pass.
    const activityBlocks = glsl({
      arena: ncaGrid[0],
      FP: `
        const int BLOCK = 8;
        float max_alpha = 0.0;
        ivec2 base = I * BLOCK;
        for (int by=0; by<BLOCK; ++by) {
          for (int bx=0; bx<BLOCK; ++bx) {
            max_alpha = max(max_alpha, arena(base + ivec2(bx,by),0).w);
          }
        }
        FOut = vec4(max_alpha, 0.0, 0.0, 1.0);
      `
    }, {
      size: [12, 12],
      layern: 1,
      format: 'rgba32f',
      tag: 'profile_activity_blocks'
    });

    const activityMap = glsl({
      activity_blocks: activityBlocks,
      FP: `
        float max_alpha = 0.0;
        const int BLOCK_GRID = 12;
        for (int y=0; y<BLOCK_GRID; ++y) {
          for (int x=0; x<BLOCK_GRID; ++x) {
            max_alpha = max(max_alpha, activity_blocks(ivec2(x,y),0).x);
          }
        }
        FOut = vec4(max_alpha, 0.0, 0.0, 1.0);
      `
    }, {
      size: [1, 1],
      layern: 1,
      format: 'rgba32f',
      tag: 'profile_activity_check'
    });

    // If no living cell remains, inject the standard seed at the configured
    // location. If the arena is alive every fragment discards, so state is untouched.
    glsl({
      activity_map: activityMap,
      FP: `
        bool arena_dead = activity_map(ivec2(0,0),0).x <= 0.1;
        if (arena_dead && I.x == ${SEED_CELL_X} && I.y == ${SEED_CELL_Y}) {
          FOut = vec4(0.0, 0.0, 0.0, 1.0);
          FOut1 = FOut2 = FOut3 = vec4(1.0);
          FOut4 = FOut5 = FOut6 = FOut7 = vec4(1.0);
        } else {
          discard;
        }
      `
    }, ncaGrid[0]);
  }

  function step() {
    // This also makes a completely erased/dead rollout self-healing. The full
    // padded arena is checked, not merely the visible 512x512 crop.
    ensureArenaHasLivingCell();
    const { nca } = model;

    // Raw Euler update with the same circular perception kernels and 0.5
    // stochastic cell update used by GrowingNCA.step_euler().
    glsl({
      ...nca,
      seed: Math.random() * 26321,
      FP: `
        const int C4 = ${C4};
        const mat3 Kx = mat3(-1,-2,-1, 0,0,0, 1,2,1);
        const mat3 Ky = mat3(-1,0,1, -2,0,2, -1,0,1);
        const mat3 Klap = mat3(1,2,1, 2,-12,2, 1,2,1);
        vec4 perc[C4*4], upd[C4];

        void neib(int x, int y) {
          ivec2 p = (ivec2(I.x+x-1, I.y+y-1)+ViewSize)%ViewSize;
          for (int i=0; i<C4; ++i) {
            vec4 v = Src(p,i);
            perc[C4+i]   += Kx[x][y]*v;
            perc[C4*2+i] += Ky[x][y]*v;
            perc[C4*3+i] += Klap[x][y]*v;
          }
        }
        void write_upd() {
          FOut = upd[0]; FOut1 = upd[1]; FOut2 = upd[2]; FOut3 = upd[3];
          FOut4 = upd[4]; FOut5 = upd[5]; FOut6 = upd[6]; FOut7 = upd[7];
        }
        void fragment() {
          for (int i=0; i<C4; ++i) {
            upd[i] = perc[i] = Src(I,i);
            perc[i+C4*3] = Klap[1][1]*upd[i];
          }
          neib(0,0); neib(0,1); neib(0,2);
          neib(1,0);            neib(1,2);
          neib(2,0); neib(2,1); neib(2,2);

          if (hash(ivec3(I,seed)).x >= 0.5) {
            int ci = w1_size().x, ch = w1_size().y;
            for (int h=0; h<ch; ++h) {
              float y = b1(ivec2(0,h)).x;
              for (int i=0; i<ci; ++i) y += dot(perc[i], w1(ivec2(i,h)));
              y = max(y, 0.0);
              if (y == 0.0) continue;
              for (int i=0; i<C4; ++i) upd[i] += y*w2t(ivec2(i,h));
            }
          }
          write_upd();
        }
      `
    }, ncaGrid);

    const preUpdateState = ncaGrid[1];

    // GrowingNCA keeps a cell only when it is alive both before and after the
    // raw update. This second pass reproduces that pre/post life masking.
    glsl({
      old_state: preUpdateState,
      FP: `
        bool inside(ivec2 p) {
          return p.x >= 0 && p.y >= 0 && p.x < ViewSize.x && p.y < ViewSize.y;
        }
        float old_max_alpha() {
          float a = 0.0;
          for (int dx=-1; dx<=1; ++dx) for (int dy=-1; dy<=1; ++dy) {
            ivec2 p = I + ivec2(dx,dy);
            if (inside(p)) a = max(a, old_state(p,0).w);
          }
          return a;
        }
        float new_max_alpha() {
          float a = 0.0;
          for (int dx=-1; dx<=1; ++dx) for (int dy=-1; dy<=1; ++dy) {
            ivec2 p = I + ivec2(dx,dy);
            if (inside(p)) a = max(a, Src(p,0).w);
          }
          return a;
        }
        void zero_all() {
          FOut = FOut1 = FOut2 = FOut3 = vec4(0.0);
          FOut4 = FOut5 = FOut6 = FOut7 = vec4(0.0);
        }
        void copy_all() {
          FOut = Src(I,0); FOut1 = Src(I,1); FOut2 = Src(I,2); FOut3 = Src(I,3);
          FOut4 = Src(I,4); FOut5 = Src(I,5); FOut6 = Src(I,6); FOut7 = Src(I,7);
        }
        void fragment() {
          if (old_max_alpha() > 0.1 && new_max_alpha() > 0.1) copy_all();
          else zero_all();
        }
      `
    }, ncaGrid);

    totalStepCount += 1;
    decoderDirty = true;
  }

  function erase(adjustedX, adjustedY) {
    if (!model || !ncaGrid || cancelled) return;

    // Same erase rule as the Cells2Pixels growing demo. VIEW_R is not an
    // arbitrary zoom here: it is exactly the unpadded 512/768 training window.
    glsl({
      x_pos: adjustedX,
      y_pos: adjustedY,
      viewR: VIEW_R,
      viewC: VIEW_C,
      brush_size: BRUSH_SIZE,
      FP: `
        vec2 click_pos = vec2(x_pos, y_pos) * viewR * 0.5 + viewC;
        float dist = length(UV.xy - click_pos);
        if (dist < 0.05 * brush_size * viewR) {
          FOut = FOut1 = FOut2 = FOut3 = vec4(0.0);
          FOut4 = FOut5 = FOut6 = FOut7 = vec4(0.0);
        } else {
          discard;
        }
      `
    }, ncaGrid[0]);

    // If the brush removed the final living cells, restore the configured seed
    // immediately; the recovery rollout below will then regrow it.
    ensureArenaHasLivingCell();

    // Guarantee at least RECOVERY_STEPS future updates after damage without
    // needlessly extending an early initial rollout.
    const initialStepsRemaining = Math.max(0, INITIAL_STEPS - totalStepCount);
    const extraRecoveryNeeded = Math.max(0, RECOVERY_STEPS - initialStepsRemaining);
    recoveryStepsRemaining = Math.max(recoveryStepsRemaining, extraRecoveryNeeded);
    decoderDirty = true;
  }

  function updateSiren() {
    const { siren } = model;

    // Decode only the central 64x64-cell content window to a 512x512 texture.
    // The outer 16 cells on every side still exist and evolve in ncaGrid, but
    // they are intentionally outside the visible SIREN sampling window.
    sirenGrid = glsl({
      nca_grid: ncaGrid[0].linear.repeat,
      mask_grid: ncaGrid[0].linear.edge,
      ...siren,
      viewR: VIEW_R,
      viewC: VIEW_C,
      FP: `
        float max_pool_alive(vec2 xy, vec2 sz) {
          float max_alive = 0.0;
          vec2 out_sz = vec2(ViewSize);
          for (int dx=-1; dx<2; ++dx) for (int dy=-1; dy<2; ++dy) {
            vec2 xyp = xy + vec2(dx,dy) / out_sz;
            if (xyp.x < 0.0 || xyp.x > 1.0 || xyp.y < 0.0 || xyp.y > 1.0) continue;
            max_alive = max(max_alive, mask_grid(xyp,0).w);
          }
          return max_alive;
        }
        void fragment() {
          vec4 A[D4], B[D4];
          vec2 sz = vec2(nca_grid_size().xy);
          vec2 fetch_uv = XY * viewR * 0.5 + viewC;

          if (max_pool_alive(fetch_uv, sz) <= 0.1) {
            FOut = vec4(0.0);
            return;
          }

          vec2 patch_coord = (fract(fetch_uv*sz)-0.5)*2.0;
          A[0].yx = sin(PI * patch_coord);
          A[0].wz = cos(PI * patch_coord);
          for (int i=0; i<C4; ++i) A[i+1] = nca_grid(fetch_uv,i);
          run_layer0(A, B);
          run_layer1(B, A);
          run_layer2(A, B);
          run_layer3(B, A);
          FOut = A[0];
        }
      `
    }, {
      size: [DISPLAY_OUTPUT_PX, DISPLAY_OUTPUT_PX],
      format: 'rgba32f',
      layern: 1,
      tag: 'profile_siren'
    });

    decoderDirty = false;
    lastDecodedStep = totalStepCount;
  }

  function renderFrame() {
    if (cancelled) return;
    try {
      glsl.adjustCanvas();

      const initialRollout = totalStepCount < INITIAL_STEPS;
      const recovering = recoveryStepsRemaining > 0;
      if ((initialRollout || recovering) && (frameCount % 2) === 0) {
        step();
        if (!initialRollout && recoveryStepsRemaining > 0) recoveryStepsRemaining -= 1;
      }
      frameCount += 1;

      const rolloutFinished = totalStepCount >= INITIAL_STEPS && recoveryStepsRemaining <= 0;
      const decodeDue = !sirenGrid || (decoderDirty && (
        totalStepCount - lastDecodedStep >= DECODE_EVERY || rolloutFinished
      ));
      if (decodeDue) updateSiren();

      glsl({
        siren_grid: sirenGrid.linear,
        Aspect: 'fit',
        FP: `
          void fragment() {
            // siren_grid is already the exact 512x512 unpadded crop.
            vec2 uv = vec2(UV.x, 1.0 - UV.y);
            FOut = clamp(siren_grid(uv, 0), 0.0, 1.0);
          }
        `
      });

      animationId = requestAnimationFrame(renderFrame);
    } catch (error) {
      console.warn('Portrait NCA runtime failed; using static portrait.', error);
      showStatic('runtime-error');
    }
  }

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const y = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
    return { x, y, adjustedX: x * 2 - 1, adjustedY: y * 2 - 1 };
  }

  function updateBrushCursor(event) {
    if (!brushCursor || root.dataset.mode !== 'live') return;
    const p = pointerPosition(event);
    brushCursor.style.left = `${p.x * 100}%`;
    brushCursor.style.top = `${p.y * 100}%`;
    brushCursor.style.width = `${10 * BRUSH_SIZE}%`;
    brushCursor.style.height = `${10 * BRUSH_SIZE}%`;
    brushCursor.hidden = false;
  }

  function installBrush() {
    canvas.addEventListener('pointerenter', (event) => updateBrushCursor(event));
    canvas.addEventListener('pointerleave', () => {
      if (!pointerDown) brushCursor.hidden = true;
    });
    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      pointerDown = true;
      canvas.setPointerCapture?.(event.pointerId);
      updateBrushCursor(event);
      const p = pointerPosition(event);
      erase(p.adjustedX, p.adjustedY);
    });
    canvas.addEventListener('pointermove', (event) => {
      updateBrushCursor(event);
      if (!pointerDown) return;
      event.preventDefault();
      const p = pointerPosition(event);
      erase(p.adjustedX, p.adjustedY);
    });
    const end = (event) => {
      pointerDown = false;
      try { canvas.releasePointerCapture?.(event.pointerId); } catch (_) {}
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }

  async function start() {
    if (!IS_WEB_ORIGIN) {
      staticImage.src = STATIC_URL;
      console.warn('[Portrait NCA] Live inference is disabled for file:// pages. Start a local HTTP server (python3 -m http.server) to test the NCA.');
      showStatic('unsupported-protocol');
      return;
    }

    staticImage.src = resolveResourceURL(STATIC_URL, 'static portrait', true);

    if (isPhone) {
      showStatic('phone');
      return;
    }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      showStatic('reduced-motion');
      return;
    }

    const timeoutId = window.setTimeout(() => showStatic('timeout'), LOAD_TIMEOUT_MS);
    try {
      if (!TRAINING_GEOMETRY_VALID) {
        throw new Error('Invalid fixed training geometry; expected a 96x96 arena for the 512px target + 128px padding checkpoint.');
      }
      const [, modelData] = await Promise.all([ensureSwissGL(), fetchModel()]);
      if (cancelled) return;

      const webgl = canvas.getContext('webgl2', {
        alpha: true,
        premultipliedAlpha: true,
        antialias: false,
        preserveDrawingBuffer: false,
        powerPreference: 'high-performance'
      });
      if (!webgl || !window.SwissGL) throw new Error('WebGL2 is unavailable');

      canvas.addEventListener('webglcontextlost', (event) => {
        event.preventDefault();
        console.warn('[Portrait NCA] WebGL context lost; switching to static portrait.');
        showStatic('webgl-context-lost');
      }, { once: true });

      glsl = window.SwissGL(canvas);
      model = buildModel(modelData);
      reset();
      installBrush();
      showLive();
      window.clearTimeout(timeoutId);

      console.info('[Portrait NCA] training-aligned crop + erase enabled', {
        fullGrid: `${W}x${H}`,
        visibleCells: `${CONTENT_CELLS}x${CONTENT_CELLS}`,
        paddingCells: PAD_CELLS,
        trainingRendererScale: TRAIN_RENDERER_SCALE,
        displayRendererScale: DISPLAY_RENDERER_SCALE,
        displayTexturePixels: `${DISPLAY_OUTPUT_PX}x${DISPLAY_OUTPUT_PX}`,
        fullTrainingRenderedPixels: `${TOTAL_PX}x${TOTAL_PX}`,
        trainedVisiblePixels: `${TARGET_PX}x${TARGET_PX}`,
        viewRange: [PADDING_PX / TOTAL_PX, (PADDING_PX + TARGET_PX) / TOTAL_PX],
        seedVisible: [SEED_X, SEED_Y],
        seedCell: [SEED_CELL_X, SEED_CELL_Y],
        background: {
          fit: BACKGROUND_FIT,
          zoom: BACKGROUND_ZOOM,
          scaleXY: [BACKGROUND_SCALE_X, BACKGROUND_SCALE_Y],
          positionPercent: [BACKGROUND_POSITION_X, BACKGROUND_POSITION_Y],
          offsetPercent: [BACKGROUND_OFFSET_X, BACKGROUND_OFFSET_Y]
        },
        decodeEverySteps: DECODE_EVERY,
        autoReseed: true,
        eraser: 'always on'
      });

      redo.addEventListener('click', () => {
        if (!model || cancelled) return;
        reset();
      });

      renderFrame();
    } catch (error) {
      window.clearTimeout(timeoutId);
      console.warn('[Portrait NCA] unavailable; using static portrait.', {
        error,
        page: location.href,
        model: (() => { try { return resolveResourceURL(MODEL_URL, 'NCA model', true); } catch (_) { return MODEL_URL; } })(),
        protocol: location.protocol
      });
      showStatic('error');
    }
  }

  start();
})();
