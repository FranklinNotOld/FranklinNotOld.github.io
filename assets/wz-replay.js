// Cover for the Work Zone Risk Prediction page: a real-time 3D replay of real
// radar tracks (assets/projects/workzone/replay-data.js) at a lane closure,
// drawn at true scale with three.js.
//
// Road frame (feet, origin at the radar): x across the road, y along it,
// traffic drives towards -y. World frame: x = x, z = -y, up = +y, so traffic
// drives towards +z and nothing is mirrored.
//
// Cars in the closing lane carry their 4-second reach; when it crosses the
// closure front the car is flagged for Stage 2 — the Stage 1 time-to-collision
// gate, computed live from the tracks. If three.js or the data are missing the
// poster image under the canvas stays visible.
(function () {
  const host = document.getElementById('wz-replay');
  if (!host || !window.THREE || !window.WZ_REPLAY || !THREE.EffectComposer) return;
  const D = window.WZ_REPLAY;
  const T = THREE;

  // ---- site geometry (road frame) -------------------------------------------
  const LANE_LINES = [-37.5, -25.65, -13.9];           // between the four lanes
  const EDGE_R = -49.5, EDGE_L = -2.0;                  // driver's right / left (median) edges
  const DEV_X = D.device_x;                             // drums along the closed lane, 1.5 ft inside the lane line
  const CLOSED = [DEV_X, EDGE_L];                       // closed lane behind the drums
  const IN_LANE = -12.4, LANE_LINE = -13.9;             // centre this far in counts as "in the closing lane"
  const FENCE = D.fence_y, TAPER = D.taper_start, GATE = D.gate_s;
  const W = (x, y, h = 0) => new T.Vector3(x, h, -y);    // road frame -> world

  const COL = {
    open: new T.Color(0.62, 0.70, 0.84), wz: new T.Color(0.30, 0.88, 1.0), hot: new T.Color(1.0, 0.22, 0.32),
  };

  // ---- renderer, scene, camera ----------------------------------------------
  const renderer = new T.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFSoftShadowMap;
  host.appendChild(renderer.domElement);

  const scene = new T.Scene();
  scene.background = new T.Color(0x04070d);
  scene.fog = new T.FogExp2(0x04070d, 0.00045);
  const camera = new T.PerspectiveCamera(30, 2.35, 10, 6000);

  scene.add(new T.HemisphereLight(0x8aa6d6, 0x05070b, 0.55));
  const moon = new T.DirectionalLight(0xc6d6ff, 0.75);
  moon.position.copy(W(-260, 520, 420));
  moon.target.position.copy(W(-25, 200, 0));
  moon.castShadow = true;
  Object.assign(moon.shadow.camera, { left: -500, right: 500, top: 500, bottom: -500, near: 10, far: 1600 });
  moon.shadow.mapSize.set(2048, 2048);
  moon.shadow.bias = -0.0006;
  scene.add(moon, moon.target);
  // warm work lights over the activity area
  const workLight = new T.PointLight(0xffb86b, 1.4, 260, 1.6);
  workLight.position.copy(W(-6, -110, 26));
  scene.add(workLight);

  // ---- shaders ----------------------------------------------------------------
  // ground ribbons: trails and the 4 s reach (additive, soft edges, optional chevrons)
  const ribbonMat = (chevron) => new T.ShaderMaterial({
    uniforms: { uColor: { value: new T.Color() }, uOpacity: { value: 1 }, uTime: { value: 0 }, uChevron: { value: chevron ? 1 : 0 } },
    vertexShader: `attribute float aA; attribute float aS; attribute float aV; varying float vA; varying float vS; varying float vV;
      void main(){ vA=aA; vS=aS; vV=aV; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; uniform float uTime; uniform float uChevron;
      varying float vA; varying float vS; varying float vV;
      void main(){
        float edge = 1. - abs(vV*2. - 1.);
        float a = vA * smoothstep(0., .7, edge);
        if (uChevron > .5) {
          float c = fract(vS/9. - uTime*1.4 + abs(vV - .5)*.8);
          a *= .25 + .75*smoothstep(.45, .6, c)*(1. - smoothstep(.8, .92, c));
        }
        gl_FragColor = vec4(uColor*(1. + .8*a), a*uOpacity);
      }`,
    transparent: true, depthWrite: false, blending: T.AdditiveBlending, side: T.DoubleSide,
  });

  // soft glow under a vehicle or a ring that grows when the radar acquires it
  const glowMat = () => new T.ShaderMaterial({
    uniforms: { uColor: { value: new T.Color() }, uOpacity: { value: 0 }, uRing: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; uniform float uRing; varying vec2 vUv;
      void main(){
        vec2 p = (vUv - .5)*2.; float r = length(p);
        float a = uRing > 0. ? exp(-pow((r - uRing)*14., 2.)) : pow(max(0., 1. - r), 1.6);
        gl_FragColor = vec4(uColor*1.4, a*uOpacity);
      }`,
    transparent: true, depthWrite: false, blending: T.AdditiveBlending,
  });

  // ---- static scene -------------------------------------------------------------
  const Y0 = -420, Y1 = 1500;                           // modelled stretch of road
  const plane = (w, l, mat, x, y, h = 0) => {
    const m = new T.Mesh(new T.PlaneGeometry(w, l), mat);
    m.rotation.x = -Math.PI / 2; m.position.copy(W(x, y, h)); m.receiveShadow = true; scene.add(m); return m;
  };
  // asphalt with a little grain
  const grain = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 256;
    const g = c.getContext('2d'); g.fillStyle = '#80868f'; g.fillRect(0, 0, 256, 256);
    for (let i = 0; i < 9000; i++) { const v = 100 + Math.random() * 70 | 0; g.fillStyle = `rgb(${v},${v},${v + 4})`; g.fillRect(Math.random() * 256, Math.random() * 256, 1.4, 1.4); }
    const t = new T.CanvasTexture(c); t.wrapS = t.wrapT = T.RepeatWrapping; t.repeat.set(6, 200); return t;
  })();
  // ground surfaces are pushed back in depth so markings and glows on them never z-fight
  const surface = (opts) => new T.MeshStandardMaterial({ roughness: 1, polygonOffset: true, polygonOffsetFactor: 4, polygonOffsetUnits: 8, ...opts });
  plane(3200, 3200, surface({ color: 0x090c12 }), -20, 500, -0.05);
  plane(EDGE_L - EDGE_R + 12, Y1 - Y0, surface({ color: 0x3a3f47, map: grain, roughness: 0.95 }), (EDGE_L + EDGE_R) / 2 - 2, (Y0 + Y1) / 2, 0);
  plane(26, Y1 - Y0, surface({ color: 0x0f1512 }), 11, (Y0 + Y1) / 2, 0.01);     // median
  // concrete barrier on the far side of the median
  const barrier = new T.Mesh(new T.BoxGeometry(2, 3.2, Y1 - Y0), new T.MeshStandardMaterial({ color: 0x5b616b, roughness: 0.9 }));
  barrier.position.copy(W(24, (Y0 + Y1) / 2, 1.6)); barrier.castShadow = barrier.receiveShadow = true; scene.add(barrier);

  // lane markings: 10 ft dashes every 40 ft, solid edge lines
  const dashGeo = new T.PlaneGeometry(0.5, 10); dashGeo.rotateX(-Math.PI / 2);
  const dashes = new T.InstancedMesh(dashGeo, new T.MeshBasicMaterial({ color: 0x8b93a1 }), LANE_LINES.length * 60);
  let k = 0; const mtx = new T.Matrix4();
  for (const lx of LANE_LINES) for (let y = Y0; y < Y1; y += 40) { mtx.setPosition(W(lx, y, 0.08)); dashes.setMatrixAt(k++, mtx); }
  dashes.count = k; scene.add(dashes);
  plane(0.5, Y1 - Y0, new T.MeshBasicMaterial({ color: 0x9aa2af }), EDGE_R, (Y0 + Y1) / 2, 0.08);
  plane(0.5, Y1 - Y0, new T.MeshBasicMaterial({ color: 0xb38a2e }), EDGE_L, (Y0 + Y1) / 2, 0.08);

  // drums: orange with two retroreflective bands
  const drumParts = [
    [new T.CylinderGeometry(0.95, 1.05, 3.1, 18), new T.MeshStandardMaterial({ color: 0xff6a1a, emissive: 0x5a1800, roughness: 0.6 }), 1.55],
    [new T.CylinderGeometry(1.0, 1.02, 0.36, 18), new T.MeshBasicMaterial({ color: new T.Color(2.2, 2.2, 2.1) }), 2.45],
    [new T.CylinderGeometry(1.03, 1.05, 0.36, 18), new T.MeshBasicMaterial({ color: new T.Color(2.0, 2.0, 1.9) }), 1.55],
  ];
  const drum = (x, y) => drumParts.forEach(([g, m, h]) => { const d = new T.Mesh(g, m); d.position.copy(W(x, y, h)); d.castShadow = true; scene.add(d); });
  const N_TAPER = 10;                                    // ~30 ft apart along the taper
  for (let i = 0; i <= N_TAPER; i++) { const u = i / N_TAPER; drum(-2.8 + (DEV_X + 2.8) * u, FENCE + (TAPER - FENCE) * (1 - u)); }
  for (let y = FENCE - 40; y > Y0 + 40; y -= 40) drum(DEV_X, y);

  // closed area: faint hatched band beyond the taper
  const hatch = (() => {
    const c = document.createElement('canvas'); c.width = 64; c.height = 64; const g = c.getContext('2d');
    g.strokeStyle = 'rgba(255,140,60,.9)'; g.lineWidth = 10;
    for (let i = -64; i < 128; i += 32) { g.beginPath(); g.moveTo(i, 64); g.lineTo(i + 64, 0); g.stroke(); }
    const t = new T.CanvasTexture(c); t.wrapS = t.wrapT = T.RepeatWrapping; t.repeat.set(1, 30); return t;
  })();
  const closed = plane(CLOSED[1] - CLOSED[0] - 1, FENCE - Y0, new T.MeshBasicMaterial({ map: hatch, transparent: true, opacity: 0.14, depthWrite: false }),
    (CLOSED[0] + CLOSED[1]) / 2 + 0.5, (FENCE + Y0) / 2, 0.12);
  closed.receiveShadow = false;

  // closure front: a translucent red gate across the closed lane
  const gateMat = new T.ShaderMaterial({
    uniforms: { uPulse: { value: 0 } },
    vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform float uPulse; varying vec2 vUv;
      void main(){ float a = (1. - vUv.y)*.5 + exp(-pow((vUv.y - .02)*30., 2.))*.6; a *= .55 + .45*uPulse;
        gl_FragColor = vec4(vec3(1., .2, .3)*1.6, a); }`,
    transparent: true, depthWrite: false, blending: T.AdditiveBlending, side: T.DoubleSide,
  });
  const gate = new T.Mesh(new T.PlaneGeometry(CLOSED[1] - CLOSED[0], 7), gateMat);
  gate.position.copy(W((CLOSED[0] + CLOSED[1]) / 2, FENCE, 3.5)); scene.add(gate);

  // distance to the closure: ticks on the median shoulder every 100 ft (labels are HTML, see below)
  for (let d = 100; d <= 600; d += 100) {
    const tick = new T.Mesh(new T.PlaneGeometry(4, 0.4), new T.MeshBasicMaterial({ color: new T.Color(0.4, 1.4, 1.7) }));
    tick.rotation.x = -Math.PI / 2; tick.position.copy(W(19, FENCE + d, 0.15)); scene.add(tick);
  }

  // arrow board on the median shoulder at the start of the taper, facing traffic
  const arrowTex = (() => {
    const c = document.createElement('canvas'); c.width = 256; c.height = 128; const g = c.getContext('2d');
    g.fillStyle = '#05060a'; g.fillRect(0, 0, 256, 128);
    const lamp = (x, y) => { g.fillStyle = '#ffb02e'; g.beginPath(); g.arc(x, y, 6, 0, 7); g.fill(); };
    for (let x = 30; x <= 186; x += 17) lamp(x, 64);                               // shaft
    for (let i = 1; i <= 4; i++) { lamp(186 + 0 - i * 0 + i * 9, 64 - i * 11); lamp(186 + i * 9, 64 + i * 11); } // head
    lamp(232, 64);
    return new T.CanvasTexture(c);
  })();
  const boardMat = new T.MeshBasicMaterial({ map: arrowTex, color: new T.Color(1.8, 1.8, 1.8) });
  const board = new T.Mesh(new T.PlaneGeometry(9, 4.5), boardMat);
  board.position.copy(W(3.5, TAPER + 25, 9)); board.rotation.y = Math.PI; scene.add(board);
  const boardPost = new T.Mesh(new T.BoxGeometry(0.5, 7, 0.5), new T.MeshStandardMaterial({ color: 0x30343b }));
  boardPost.position.copy(W(3.5, TAPER + 25, 3.5)); scene.add(boardPost);

  // roadside radar on a trailer in the median, looking upstream
  const radar = new T.Group();
  const dark = new T.MeshStandardMaterial({ color: 0x2a2f37, roughness: 0.7, metalness: 0.3 });
  const trailer = new T.Mesh(new T.BoxGeometry(6, 2.4, 10), dark); trailer.position.y = 2.4; trailer.castShadow = true;
  const mast = new T.Mesh(new T.CylinderGeometry(0.25, 0.3, 14, 10), dark); mast.position.y = 10.5;
  const head = new T.Mesh(new T.BoxGeometry(2.2, 1.6, 0.8), dark); head.position.set(0, 17.6, 0);
  const face = new T.Mesh(new T.PlaneGeometry(1.8, 1.2), new T.MeshBasicMaterial({ color: new T.Color(0.6, 2.4, 3.0) }));
  face.position.set(0, 17.6, -0.42); face.rotation.y = Math.PI;
  radar.add(trailer, mast, head, face); radar.position.copy(W(7, 4, 0)); scene.add(radar);

  // radar field of view on the ground, with wavefronts travelling upstream
  const fanMat = new T.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: `attribute vec2 aP; varying vec2 vP; void main(){ vP=aP; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `uniform float uTime; varying vec2 vP;
      void main(){ float r = vP.x, s = vP.y;
        float edge = smoothstep(0., .08, s)*smoothstep(1., .9, s);
        float wave = exp(-pow((fract(r*2.2 - uTime*.45) - .5)*16., 2.));
        float a = (.12 + .45*wave)*pow(1. - r, 1.1)*edge*smoothstep(0., .03, r);
        gl_FragColor = vec4(vec3(.25, .85, 1.)*1.3, a); }`,
    transparent: true, depthWrite: false, blending: T.AdditiveBlending,
  });
  (() => {
    const R = 760, A0 = -0.56, A1 = 0.02, S = 60, Rn = 24, pos = [], aP = [], idx = [];
    for (let i = 0; i <= Rn; i++) for (let j = 0; j <= S; j++) {
      const r = i / Rn, a = A0 + (A1 - A0) * j / S;
      pos.push(7 + Math.sin(a) * R * r, 0.06, -4 - Math.cos(a) * R * r); aP.push(r, j / S);
    }
    for (let i = 0; i < Rn; i++) for (let j = 0; j < S; j++) { const a = i * (S + 1) + j, b = a + S + 1; idx.push(a, b, a + 1, b, b + 1, a + 1); }
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.Float32BufferAttribute(pos, 3)); g.setAttribute('aP', new T.Float32BufferAttribute(aP, 2)); g.setIndex(idx);
    scene.add(new T.Mesh(g, fanMat));
  })();

  // work activity area: a work truck with an attenuator, workers in vests
  const box = (w, h, l, mat, x, y, z) => { const m = new T.Mesh(new T.BoxGeometry(w, h, l), mat); m.position.set(x, y, z); m.castShadow = true; return m; };
  const truck = new T.Group();
  const white = new T.MeshStandardMaterial({ color: 0xd9dce1, roughness: 0.6 });
  truck.add(box(7.6, 6.5, 8, white, 0, 4.4, 8), box(7.8, 4, 14, white, 0, 3.4, -3));
  const chevTex = (() => {
    const c = document.createElement('canvas'); c.width = 128; c.height = 64; const g = c.getContext('2d');
    g.fillStyle = '#111'; g.fillRect(0, 0, 128, 64); g.fillStyle = '#ff9a1f';
    for (let i = -2; i < 6; i++) { g.beginPath(); g.moveTo(i * 32, 64); g.lineTo(i * 32 + 16, 64); g.lineTo(i * 32 + 48, 0); g.lineTo(i * 32 + 32, 0); g.fill(); }
    return new T.CanvasTexture(c);
  })();
  const tma = new T.Mesh(new T.PlaneGeometry(7.6, 3.6), new T.MeshBasicMaterial({ map: chevTex, color: new T.Color(1.4, 1.4, 1.4) }));
  tma.position.set(0, 3.2, -14.2); tma.rotation.y = Math.PI;
  const beacon = new T.Mesh(new T.SphereGeometry(0.45, 12, 8), new T.MeshBasicMaterial({ color: new T.Color(3, 1.6, 0.2) }));
  beacon.position.set(0, 8.1, 9);
  truck.add(tma, beacon); truck.position.copy(W(-7.6, -75, 0)); scene.add(truck);
  const vest = new T.MeshStandardMaterial({ color: 0xd8ff3a, emissive: 0x3a4a00 });
  const hat = new T.MeshStandardMaterial({ color: 0xf2f2f2 });
  [[-5.5, -128], [-9.5, -142], [-4.6, -160]].forEach(([x, y]) => {
    const p = new T.Group();
    p.add(box(1.4, 3.0, 0.9, new T.MeshStandardMaterial({ color: 0x1d2330 }), 0, 1.5, 0), box(1.6, 2.2, 1.0, vest, 0, 4.0, 0));
    const h = new T.Mesh(new T.SphereGeometry(0.55, 12, 8), hat); h.position.y = 5.6; p.add(h);
    p.position.copy(W(x, y, 0)); p.rotation.y = Math.random() * 6; scene.add(p);
  });

  // ---- vehicles -----------------------------------------------------------------
  const PAINT = [0x2a2f38, 0xb9bec7, 0xe3e5e9, 0x1c2a44, 0x5b1d26, 0x3d434c, 0x101318, 0x7e848e, 0x26394f];
  const glass = new T.MeshStandardMaterial({ color: 0x0b1018, roughness: 0.15, metalness: 0.6 });
  const tyre = new T.MeshStandardMaterial({ color: 0x0a0b0d, roughness: 0.9 });
  const headMat = new T.MeshBasicMaterial({ color: new T.Color(3.2, 3.1, 2.8) });
  const tailMat = new T.MeshBasicMaterial({ color: new T.Color(2.6, 0.12, 0.18) });

  // side profile (u along the car from the rear, v up), extruded across the width
  const extrude = (pts, width, mat) => {
    const s = new T.Shape(pts.map(([u, v]) => new T.Vector2(u, v)));
    const g = new T.ExtrudeGeometry(s, { depth: width, bevelEnabled: true, bevelThickness: 0.12, bevelSize: 0.12, bevelSegments: 2 });
    g.rotateY(-Math.PI / 2); g.translate(width / 2, 0, 0);
    const m = new T.Mesh(g, mat); m.castShadow = true; return m;
  };
  const wheel = (x, z, r = 1.05) => { const w = new T.Mesh(new T.CylinderGeometry(r, r, 0.8, 16), tyre); w.rotation.z = Math.PI / 2; w.position.set(x, r, z); return w; };

  function makeCar(paint) {
    const L = 15.6, Wd = 6.2, g = new T.Group(), body = new T.MeshStandardMaterial({ color: paint, roughness: 0.35, metalness: 0.55 });
    const lower = extrude([[0.1, 0.9], [15.4, 0.9], [15.6, 1.6], [15.5, 2.6], [15.2, 2.95], [11.4, 3.35], [4.2, 3.35], [0.5, 3.2], [0, 2.6], [0, 1.4]], Wd, body);
    const cabin = extrude([[4.0, 3.3], [11.6, 3.3], [9.6, 4.7], [5.3, 4.7]], Wd - 0.7, glass);
    const roof = extrude([[5.25, 4.68], [9.65, 4.68], [9.45, 4.92], [5.45, 4.92]], Wd - 0.75, body);
    [lower, cabin, roof].forEach(m => m.position.z = -L / 2);
    g.add(lower, cabin, roof, wheel(2.8, -L / 2 + 2.9), wheel(-2.8, -L / 2 + 2.9), wheel(2.8, L / 2 - 3.1), wheel(-2.8, L / 2 - 3.1));
    const hl = box(1.3, 0.35, 0.15, headMat, 2.2, 2.55, L / 2 + 0.02); const hr = hl.clone(); hr.position.x = -2.2;
    const tl = box(5.2, 0.3, 0.15, tailMat, 0, 2.75, -L / 2 - 0.02);
    g.add(hl, hr, tl);
    return { g, L, Wd, H: 5.0 };
  }
  function makeTruck(semi) {
    const g = new T.Group(), cabMat = new T.MeshStandardMaterial({ color: semi ? 0x8f1f24 : 0xd5d8dd, roughness: 0.4, metalness: 0.4 });
    const boxMat = new T.MeshStandardMaterial({ color: 0xcfd3d9, roughness: 0.7 });
    let L;
    if (semi) {
      L = 70;
      g.add(box(8.2, 7.5, 8, cabMat, 0, 6.2, 27), box(7.6, 3.6, 6, cabMat, 0, 4.3, 32.6), box(8.4, 9.5, 48, boxMat, 0, 9.0, -9));
      g.add(box(7.4, 0.35, 0.15, headMat, 0, 4.0, 35.7));
      [33, 25, 21, -24, -28].forEach(z => { g.add(wheel(3.6, z, 1.6), wheel(-3.6, z, 1.6)); });
    } else {
      L = 26;
      g.add(box(7.6, 6.6, 7, cabMat, 0, 4.6, 9.5), box(8.0, 9.0, 18, boxMat, 0, 6.2, -3.5));
      g.add(box(6.6, 0.35, 0.15, headMat, 0, 3.4, 13.05));
      [10, -8, -10.5].forEach(z => { g.add(wheel(3.3, z, 1.4), wheel(-3.3, z, 1.4)); });
    }
    g.add(box(7.0, 0.35, 0.15, tailMat, 0, 2.6, -L / 2 - 0.05));
    return { g, L, Wd: 8.4, H: semi ? 13.8 : 10.7 };
  }

  const beamGeo = (() => {                       // headlight pool on the road ahead of a car
    const g = new T.PlaneGeometry(18, 46); g.rotateX(-Math.PI / 2); g.translate(0, 0.05, 23 + 7.8); return g;
  })();
  const beamMat = new T.ShaderMaterial({
    vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `varying vec2 vUv;
      void main(){ float along = vUv.y; float spread = mix(.12, .5, along);
        float a = smoothstep(spread, spread*.4, abs(vUv.x - .5))*(1. - along)*smoothstep(0., .08, along)*.16;
        gl_FragColor = vec4(vec3(.9, .95, 1.), a); }`,
    transparent: true, depthWrite: false, blending: T.AdditiveBlending,
  });

  class Ribbon {
    constructor(n, width, chevron) {
      this.n = n; this.w = width; this.mat = ribbonMat(chevron);
      const g = new T.BufferGeometry();
      this.p = new Float32Array(n * 6); this.a = new Float32Array(n * 2); this.s = new Float32Array(n * 2);
      const v = new Float32Array(n * 2); for (let i = 0; i < n; i++) { v[i * 2] = 0; v[i * 2 + 1] = 1; }
      g.setAttribute('position', new T.BufferAttribute(this.p, 3)); g.setAttribute('aA', new T.BufferAttribute(this.a, 1));
      g.setAttribute('aS', new T.BufferAttribute(this.s, 1)); g.setAttribute('aV', new T.BufferAttribute(v, 1));
      const idx = []; for (let i = 0; i < n - 1; i++) { const a = i * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
      g.setIndex(idx); this.g = g;
      this.mesh = new T.Mesh(g, this.mat); this.mesh.frustumCulled = false; scene.add(this.mesh);
    }
    // pts: [[worldX, worldZ, alpha], ...]
    set(pts, h = 0.3) {
      const m = Math.min(pts.length, this.n); let s = 0;
      for (let i = 0; i < m; i++) {
        const [x, z, a] = pts[i], j = Math.min(i + 1, m - 1), q = Math.max(i - 1, 0);
        let dx = pts[j][0] - pts[q][0], dz = pts[j][1] - pts[q][1]; const L = Math.hypot(dx, dz) || 1; dx /= L; dz /= L;
        const nx = -dz * this.w / 2, nz = dx * this.w / 2;
        this.p.set([x + nx, h, z + nz, x - nx, h, z - nz], i * 6);
        if (i > 0) s += Math.hypot(x - pts[i - 1][0], z - pts[i - 1][1]);
        this.a[i * 2] = this.a[i * 2 + 1] = a; this.s[i * 2] = this.s[i * 2 + 1] = s;
      }
      this.g.attributes.position.needsUpdate = this.g.attributes.aA.needsUpdate = this.g.attributes.aS.needsUpdate = true;
      this.g.setDrawRange(0, Math.max(0, m - 1) * 6);
      this.mesh.visible = m > 1;
    }
  }

  const labels = document.createElement('div'); labels.className = 'wz-labels'; host.appendChild(labels);
  const vehicles = D.vehicles.map((v, i) => {
    const model = v.k === 0 ? makeCar(PAINT[(i * 7) % PAINT.length]) : makeTruck(v.k === 2);
    const g = model.g; g.visible = false; scene.add(g);
    const outline = new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(model.Wd + 1.2, model.H + 0.8, model.L + 1.4)),
      new T.LineBasicMaterial({ color: COL.open, transparent: true, opacity: 0.35 }));
    outline.position.y = (model.H + 0.8) / 2; g.add(outline);
    const beam = new T.Mesh(beamGeo, beamMat); beam.position.z = model.L / 2 - 7.8; g.add(beam);
    const glow = new T.Mesh(new T.PlaneGeometry(1, 1), glowMat()); glow.rotation.x = -Math.PI / 2; glow.visible = false; scene.add(glow);
    const ping = new T.Mesh(new T.PlaneGeometry(1, 1), glowMat()); ping.rotation.x = -Math.PI / 2; ping.visible = false; scene.add(ping);
    const tag = document.createElement('div'); tag.className = 'wz-tag'; labels.appendChild(tag);
    return { v, model, g, outline, glow, ping, tag, trail: new Ribbon(32, 1.3, false), reach: new Ribbon(24, 2.4, true), id: 1000 + i * 37 % 900 };
  });

  // ---- sampling -----------------------------------------------------------------
  const clampX = x => Math.min(EDGE_L - 3.5, Math.max(EDGE_R + 3.5, x));
  const COAST = 10;                                      // frames a track coasts on after its last fix
  function velocity(v, a) {
    const n = v.y.length, p0 = Math.max(0, a - 4), p1 = Math.min(n - 1, a + 4), dt = (p1 - p0) / D.hz || 0.1;
    return [(v.x[p1] - v.x[p0]) / dt, (v.y[p1] - v.y[p0]) / dt];
  }
  function sample(v, f) {
    const i = f - v.s, n = v.y.length;
    if (i < 0 || i > n - 1 + COAST) return null;
    if (i > n - 1) {                                     // past the last fix: straight on at the last speed, fading
      const [, vy] = velocity(v, n - 1), dt = (i - (n - 1)) / D.hz;
      return { x: clampX(v.x[n - 1]), y: v.y[n - 1] + vy * dt, vx: 0, vy, i, n, fade: 1 - (i - (n - 1)) / COAST };
    }
    const a = Math.floor(i), b = Math.min(a + 1, n - 1), u = i - a;
    const x = v.x[a] + (v.x[b] - v.x[a]) * u, y = v.y[a] + (v.y[b] - v.y[a]) * u;
    const [vx, vy] = velocity(v, a);
    return { x: clampX(x), y, vx, vy, i, n, fade: Math.min(1, i / 6) };
  }
  // Stage 1, replayed: in the closing lane and TTC <= 4 s. Once flagged a car stays
  // flagged until it has left the lane, so the state does not flicker.
  function flagged(v, f) {
    let on = false;
    for (let k = v.s; k <= Math.min(f, v.s + v.y.length - 1); k++) {
      const a = k - v.s, x = v.x[a], y = v.y[a], [, vy] = velocity(v, a);
      const ttc = (y - FENCE) / Math.max(1, -vy);
      if (x > IN_LANE && y > FENCE && ttc <= GATE) on = true;
      else if (x < LANE_LINE) on = false;
    }
    return on;
  }

  // ---- labels and HUD (HTML over the canvas) ------------------------------------
  const hud = document.createElement('div'); hud.className = 'wz-hud';
  hud.innerHTML = `<div class="tl"><span><b>REPLAY</b> · REAL RADAR TRACKS · INDIANA WORK ZONE<span class="t"></span></span></div>
    <div class="tr"><span>VEHICLES <b class="n0"></b></span><span>CLOSING LANE <b class="n1 c"></b></span><span>FLAGGED <b class="n2 r"></b></span></div>
    <div class="bl"><span><i class="o"></i>tracked vehicle</span><span><i class="c"></i>closing lane · 4 s reach</span><span><i class="r"></i>reach crosses the closure → Stage 2</span></div>`;
  host.appendChild(hud);
  const seam = document.createElement('div'); seam.className = 'wz-seam'; host.appendChild(seam);
  // fixed labels: [text, anchor, colour, which side of the anchor]
  const distTags = [100, 200, 300, 400, 500].map(d => [`${d} ft`, W(21.5, FENCE + d, 0), 'd', 'right']);
  const fixedTags = [['CLOSURE FRONT', W(-8, FENCE, 8), 'r', 'left'], ['RADAR', W(7, 4, 21), 'c', 'right'], ...distTags].map(([txt, p, c, side]) => {
    const e = document.createElement('div'); e.className = 'wz-tag fixed ' + c; e.textContent = txt; labels.appendChild(e); return { e, p, side };
  });
  const tmp = new T.Vector3();
  const ANCHOR = { center: 'translate(-50%, -100%)', left: 'translate(calc(-100% - 6px), -50%)', right: 'translate(6px, -50%)' };
  function place(el, p, side = 'center') {
    tmp.copy(p).project(camera);
    const vis = tmp.z < 1 && Math.abs(tmp.x) < 1.05 && Math.abs(tmp.y) < 1.05;
    el.style.display = vis ? '' : 'none';
    if (vis) el.style.transform = `translate(${(tmp.x + 1) / 2 * host.clientWidth}px, ${(1 - tmp.y) / 2 * host.clientHeight}px) ${ANCHOR[side]}`;
  }

  // ---- composer -----------------------------------------------------------------
  const composer = new T.EffectComposer(renderer);
  composer.addPass(new T.RenderPass(scene, camera));
  const bloom = new T.UnrealBloomPass(new T.Vector2(512, 256), 1.05, 0.6, 0.42);
  composer.addPass(bloom);

  // ---- camera -------------------------------------------------------------------
  // a drone view from beside the open lanes, behind the traffic, looking across at the closing lane
  const VIEWS = {
    wide: { pos: W(-150, 370, 82), at: W(-14, 150, 0), fov: 32 },
    square: { pos: W(-150, 420, 175), at: W(-12, 170, 0), fov: 44 },
  };
  // #cam=x,y,h,atX,atY,fov (road frame) overrides the view, for framing tests
  const camHash = /cam=([-\d.,]+)/.exec(location.hash);
  if (camHash) { const c = camHash[1].split(',').map(Number); VIEWS.wide = VIEWS.square = { pos: W(c[0], c[1], c[2]), at: W(c[3], c[4], 0), fov: c[5] }; }
  let view = VIEWS.wide;
  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); composer.setSize(w, h);
    camera.aspect = w / h; view = camera.aspect < 1.5 ? VIEWS.square : VIEWS.wide;
    camera.fov = view.fov; camera.updateProjectionMatrix();
  }

  // ---- frame --------------------------------------------------------------------
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const n0 = hud.querySelector('.n0'), n1 = hud.querySelector('.n1'), n2 = hud.querySelector('.n2'), tEl = hud.querySelector('.t');
  // the loop: 24 s of the recording with steady traffic and three flagged cars
  const LOOP0 = 21, LOOP1 = D.frames / D.hz, DUR = LOOP1 - LOOP0;

  function render(t, clock) {
    const f = t * D.hz;
    // slow drift of the camera around its target
    const sway = still ? 0 : Math.sin(clock * 0.07) * 0.05;
    const off = view.pos.clone().sub(view.at);
    off.applyAxisAngle(new T.Vector3(0, 1, 0), sway);
    camera.position.copy(view.at).add(off); camera.lookAt(view.at);

    let count = 0, inWz = 0, nFlag = 0;
    for (const o of vehicles) {
      const p = sample(o.v, f);
      if (!p) { o.g.visible = o.glow.visible = o.ping.visible = o.trail.mesh.visible = o.reach.mesh.visible = false; o.tag.style.display = 'none'; continue; }
      count++;
      const speed = Math.max(1, -p.vy);
      const flag = flagged(o.v, Math.floor(f)) && p.y > FENCE - 60;
      const wz = flag || (p.x > IN_LANE && p.y > FENCE);
      const ttc = (p.y - FENCE) / speed;
      if (wz) inWz++;
      if (flag) nFlag++;
      const col = flag ? COL.hot : wz ? COL.wz : COL.open;
      const fade = p.fade;
      const pos = W(p.x, p.y, 0);

      o.g.visible = true; o.g.position.copy(pos);
      o.g.rotation.y = Math.atan2(p.vx, -p.vy);          // world velocity (vx, -vy)
      o.outline.material.color.copy(col);
      o.outline.material.opacity = (flag ? 0.95 + 0.05 * Math.sin(clock * 12) : wz ? 0.9 : 0.45) * fade;
      // the radar has just acquired this vehicle
      const age = p.i / D.hz;
      if (age < 1.2 && p.i < p.n - 1) {
        o.ping.visible = true; o.ping.position.copy(W(p.x, p.y, 0.25)); o.ping.scale.setScalar(o.model.L * 2.4);
        o.ping.material.uniforms.uColor.value.copy(COL.wz); o.ping.material.uniforms.uRing.value = 0.15 + 0.8 * age / 1.2;
        o.ping.material.uniforms.uOpacity.value = 1 - age / 1.2;
        o.outline.material.opacity = Math.max(o.outline.material.opacity, (1 - age / 1.2) * (0.5 + 0.5 * Math.sin(clock * 40)));
      } else o.ping.visible = false;

      o.glow.visible = wz;
      if (wz) {
        o.glow.position.copy(W(p.x, p.y, 0.2)); o.glow.scale.set(o.model.Wd * 2.6, o.model.L * 1.5, 1);
        o.glow.rotation.z = o.g.rotation.y;
        o.glow.material.uniforms.uColor.value.copy(col); o.glow.material.uniforms.uOpacity.value = (flag ? 0.75 : 0.45) * fade;
      }

      // trail: the last 3 s of the real track
      const trail = [];
      for (let k2 = Math.max(0, Math.floor(p.i) - 30); k2 <= Math.min(Math.floor(p.i), p.n - 1); k2++) {
        const q = W(clampX(o.v.x[k2]), o.v.y[k2]); trail.push([q.x, q.z, 0]);
      }
      trail.push([pos.x, pos.z, 0]);
      trail.forEach((q, j) => { q[2] = (j / (trail.length - 1)) * (wz ? 0.75 : 0.32) * fade; });
      o.trail.mat.uniforms.uColor.value.copy(col); o.trail.set(trail);

      // reach: where the car will be in 4 s at its current speed and heading
      if (wz) {
        const hx = p.vx / speed, len = speed * GATE, pts = [];
        for (let s = 0; s < 24; s++) {
          const d = o.model.L / 2 + 2 + (len - o.model.L / 2) * s / 23;
          const q = W(clampX(p.x + hx * d), p.y - d);
          pts.push([q.x, q.z, (flag ? 0.95 : 0.8) * fade * (1 - 0.6 * s / 23)]);
        }
        o.reach.mat.uniforms.uColor.value.copy(col); o.reach.mat.uniforms.uTime.value = clock; o.reach.set(pts, 0.35);
        o.tag.style.display = '';
        o.tag.className = 'wz-tag ' + (flag ? 'r' : 'c');
        o.tag.textContent = flag ? `TTC ${ttc.toFixed(1)} s → STAGE 2` : `#${o.id} · ${(speed / 1.4667).toFixed(0)} mph · TTC ${ttc.toFixed(1)} s`;
        place(o.tag, W(p.x, p.y, o.model.H + 4));
        o.tag.style.opacity = fade;
      } else { o.reach.mesh.visible = false; o.tag.style.display = 'none'; }
    }

    gateMat.uniforms.uPulse.value = nFlag ? 0.5 + 0.5 * Math.sin(clock * 10) : 0.15;
    fanMat.uniforms.uTime.value = clock;
    const blink = Math.floor(clock / 0.6) % 2 === 0;
    boardMat.color.setScalar(blink ? 1.9 : 0.25);
    beacon.material.color.setRGB(blink ? 3 : 0.4, blink ? 1.6 : 0.2, 0.1);
    fixedTags.forEach(({ e, p, side }) => place(e, p, side));

    n0.textContent = count; n1.textContent = inWz; n2.textContent = nFlag;
    tEl.textContent = ` · t ${(t - LOOP0).toFixed(1)} s / ${DUR} s`;
    const edge = Math.min(t - LOOP0, LOOP1 - t);
    seam.style.opacity = edge < 0.6 ? 1 - edge / 0.6 : 0;
    composer.render();
  }

  // ---- loop -----------------------------------------------------------------------
  let running = false, visible = true, start = null;
  function loop(now) {
    if (!running) return;
    if (start === null) start = now;
    const clock = (now - start) / 1000;
    render(LOOP0 + clock % DUR, clock);
    requestAnimationFrame(loop);
  }
  const play = () => { if (!running && visible && !document.hidden) { running = true; start = null; requestAnimationFrame(loop); } };
  const pause = () => { running = false; };

  resize();
  host.classList.add('live');
  
  const fixed = /replay-t=([\d.]+)/.exec(location.hash);
  // first moment a car is flagged, for still renders
  const firstFlag = 35.0;
  if (fixed || still) { render(fixed ? parseFloat(fixed[1]) : firstFlag, fixed ? parseFloat(fixed[1]) : 0); addEventListener('resize', () => { resize(); render(fixed ? parseFloat(fixed[1]) : firstFlag, 0); }); return; }
  render(LOOP0 + 0.7, 0);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; visible ? play() : pause(); }).observe(host);
  document.addEventListener('visibilitychange', () => document.hidden ? pause() : play());
  addEventListener('resize', resize);
  play();
})();
