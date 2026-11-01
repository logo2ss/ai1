import * as THREE from "three";

export interface SceneHeader {
  version: number;
  grid_w: number;
  grid_h: number;
  frame: { x0: number; y0: number; x1: number; y1: number };
  extent_x: number;
  extent_y: number;
  aspect: number;
  sky_disparity: number;
  ref_zfar: number;
  default_fov: number;
  default_zfar: number;
  safe_radius: number;
  back_w?: number;
  back_h?: number;
  buffers: [number, number][];
  buffer_names: string[];
}

export interface ParsedScene {
  header: SceneHeader;
  disp: Float32Array;
  keep: Uint8Array; // unpacked 0/1 per quad
  bgDisp: Float32Array;
  keepBg: Uint8Array;
  backDisp: Float32Array | null;
}

function unpackBits(bytes: Uint8Array, n: number): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

/** Parse the engine's 'LSC1' scene container. */
export function parseScene(buf: ArrayBuffer): ParsedScene {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "LSC1") throw new Error("Not a Lumen scene file");
  const hlen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen))) as SceneHeader;
  const base = 8 + hlen;
  const [gw, gh] = [header.grid_w, header.grid_h];
  const b = (i: number) => header.buffers[i];
  const disp = new Float32Array(buf.slice(base + b(0)[0], base + b(0)[0] + b(0)[1]));
  const keep = unpackBits(new Uint8Array(buf, base + b(1)[0], b(1)[1]), (gw - 1) * (gh - 1));
  const bgDisp = new Float32Array(buf.slice(base + b(2)[0], base + b(2)[0] + b(2)[1]));
  const keepBg = unpackBits(new Uint8Array(buf, base + b(3)[0], b(3)[1]), (gw - 1) * (gh - 1));
  const backDisp = header.buffers.length > 4 ? new Float32Array(buf.slice(base + b(4)[0], base + b(4)[0] + b(4)[1])) : null;
  return { header, disp, keep, bgDisp, keepBg, backDisp };
}

/** Grid geometry with a per-vertex (u, v, disparity) attribute; only kept quads are indexed. */
export function buildGeometry(gw: number, gh: number, disp: Float32Array, keep: Uint8Array, onlyUsed = false): THREE.BufferGeometry {
  const idx: number[] = [];
  const used = new Uint8Array(gw * gh);
  for (let y = 0; y < gh - 1; y++) {
    for (let x = 0; x < gw - 1; x++) {
      if (!keep[y * (gw - 1) + x]) continue;
      const a = y * gw + x;
      const b = a + 1;
      const c = a + gw;
      const d = c + 1;
      idx.push(a, c, b, b, c, d);
      used[a] = used[b] = used[c] = used[d] = 1;
    }
  }
  let uvd: Float32Array;
  let index: Uint32Array;
  if (onlyUsed) {
    // compact vertex buffer (background layer is sparse)
    const remap = new Int32Array(gw * gh).fill(-1);
    let n = 0;
    for (let i = 0; i < used.length; i++) if (used[i]) remap[i] = n++;
    uvd = new Float32Array(n * 3);
    for (let i = 0; i < used.length; i++) {
      const j = remap[i];
      if (j < 0) continue;
      uvd[j * 3] = (i % gw) / (gw - 1);
      uvd[j * 3 + 1] = Math.floor(i / gw) / (gh - 1);
      uvd[j * 3 + 2] = disp[i];
    }
    index = new Uint32Array(idx.length);
    for (let i = 0; i < idx.length; i++) index[i] = remap[idx[i]];
  } else {
    uvd = new Float32Array(gw * gh * 3);
    for (let i = 0; i < gw * gh; i++) {
      uvd[i * 3] = (i % gw) / (gw - 1);
      uvd[i * 3 + 1] = Math.floor(i / gw) / (gh - 1);
      uvd[i * 3 + 2] = disp[i];
    }
    index = new Uint32Array(idx);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("uvd", new THREE.BufferAttribute(uvd, 3));
  // dummy position attribute so three.js computes draw ranges; real positions come from the shader
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(uvd.length), 3));
  g.setIndex(new THREE.BufferAttribute(index, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -5), 1e5);
  return g;
}

export const vertexShader = /* glsl */ `
  attribute vec3 uvd;
  uniform float tanHalfX;
  uniform float aspect;
  uniform vec2 extent;
  uniform float zfar;
  uniform float pointSize;
  varying vec2 vUv;
  varying float vMargin;
  varying float vDisp;
  varying vec3 vWorld;
  vec3 unproject(vec3 a) {
    float denom = a.z * (1.0 - 1.0 / zfar) + 1.0 / zfar;
    float z = 1.0 / max(denom, 1.0 / (3.0 * zfar));
    float xn = (a.x * 2.0 - 1.0) * extent.x;
    float yn = (a.y * 2.0 - 1.0) * extent.y;
    return vec3(xn * z * tanHalfX, -yn * z * tanHalfX / aspect, -z);
  }
  void main() {
    vec3 p = unproject(uvd);
    vUv = vec2(uvd.x, 1.0 - uvd.y);
    float xn = abs((uvd.x * 2.0 - 1.0) * extent.x);
    float yn = abs((uvd.y * 2.0 - 1.0) * extent.y);
    vMargin = max(max(xn - 1.0, 0.0) / max(extent.x - 1.0, 1e-4), max(yn - 1.0, 0.0) / max(extent.y - 1.0, 1e-4));
    vDisp = uvd.z;
    vWorld = (modelMatrix * vec4(p, 1.0)).xyz;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = pointSize / max(-mv.z, 0.1);
  }
`;

// Far environment (equirectangular panorama at infinity, see engine recon/environment.py): looked up by
// view direction, u = 0.5 + atan(x, -z) / 2pi, v = 0.5 - asin(y) / pi (row 0 = up; textures are flipped
// on upload, hence 1 - v). Identical to the CPU renderer's _env_color.
const envGlsl = /* glsl */ `
  uniform sampler2D envMap;
  uniform float hasEnv;
  vec3 envColor(vec3 dir) {
    dir = normalize(dir);
    float u = 0.5 + atan(dir.x, -dir.z) / 6.28318530718;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / 3.14159265359;
    return texture2D(envMap, vec2(u, 1.0 - v)).rgb;
  }
`;

export const fragmentShader = /* glsl */ `
  uniform sampler2D map;
  uniform int mode; // 0 colour, 1 depth, 2 layer tint
  uniform float layerTint;
  uniform float marginFade;
  uniform float dim;
  uniform float stretchCos; // > 0: drop triangles seen nearly edge-on from the photo's viewpoint (render.py _drop_stretched)
  varying vec2 vUv;
  varying float vMargin;
  varying float vDisp;
  varying vec3 vWorld;
  ${envGlsl}
  vec3 inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 a = vec3(0.0015, 0.0005, 0.014), b = vec3(0.87, 0.24, 0.32), c = vec3(0.99, 0.99, 0.64);
    return t < 0.6 ? mix(a, b, t / 0.6) : mix(b, c, (t - 0.6) / 0.4);
  }
  void main() {
    if (stretchCos > 0.0) {
      // the scene lives in the reference camera's frame: the photo was taken from the origin
      vec3 n = cross(dFdx(vWorld), dFdy(vWorld));
      float nn = length(n) * length(vWorld);
      if (nn > 0.0 && abs(dot(n, vWorld)) < stretchCos * nn) discard;
    }
    vec3 col = texture2D(map, vUv).rgb;
    if (mode == 1) col = inferno(max(vDisp, 0.0));
    if (mode == 2) col = mix(col, vec3(1.0, 0.25, 0.6), layerTint);
    col *= dim;
    // low-confidence periphery: fade towards the far environment (or the background colour); with an
    // environment the outer part of the margin always dissolves into it (no hard edge)
    float m = clamp(vMargin, 0.0, 1.0);
    float f = m * marginFade;
    bool env = hasEnv > 0.5 && mode != 1;
    if (env) f = max(f, smoothstep(0.72, 1.0, m));
    vec3 target = vec3(0.04, 0.045, 0.06);
    if (env && f > 0.0) {
      target = envColor(vWorld - cameraPosition);
      if (mode == 2) target = mix(target, vec3(1.0, 0.25, 0.6), 0.85);
    }
    col = mix(col, target, f);
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }
`;

/** Environment sphere around the camera: every direction shows the far environment. */
export const envVertexShader = /* glsl */ `
  varying vec3 vDir;
  void main() {
    vDir = (modelMatrix * vec4(position, 1.0)).xyz - cameraPosition;
    vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_Position = p.xyww; // at the far plane
  }
`;

export const envFragmentShader = /* glsl */ `
  uniform int mode;
  varying vec3 vDir;
  ${envGlsl}
  void main() {
    vec3 col = mode == 1 ? vec3(0.04, 0.045, 0.06) : envColor(vDir);
    if (mode == 2) col = mix(col, vec3(1.0, 0.25, 0.6), 0.85);
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }
`;

/** Same threshold as the CPU renderer (render.py STRETCH_MIN_COS). */
export const STRETCH_MIN_COS = 0.06;

export function bakePositions(uvd: Float32Array, u: { tanHalfX: number; aspect: number; extent: [number, number]; zfar: number }): Float32Array {
  const out = new Float32Array(uvd.length);
  for (let i = 0; i < uvd.length; i += 3) {
    const denom = uvd[i + 2] * (1 - 1 / u.zfar) + 1 / u.zfar;
    const z = 1 / Math.max(denom, 1 / (3 * u.zfar));
    const xn = (uvd[i] * 2 - 1) * u.extent[0];
    const yn = (uvd[i + 1] * 2 - 1) * u.extent[1];
    out[i] = xn * z * u.tanHalfX;
    out[i + 1] = (-yn * z * u.tanHalfX) / u.aspect;
    out[i + 2] = -z;
  }
  return out;
}
