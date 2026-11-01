// The engine's dense 3D body model ('LBD1' container) and its mesh, shared by the GPU and CPU viewers.
import * as THREE from "three";

export interface BodyHeader {
  cols: number;
  rows: number;
  aspect: number;
  depth_scale: number;
  vertices: number;
  bbox: number[];
}
export interface BodyData {
  header: BodyHeader;
  z: Float32Array;
  keep: Uint8Array;
}
export interface BodyViewOptions {
  style: "wireframe" | "scanlines" | "points" | "solid";
  colors: "depth" | "photo" | "hologram";
  lineStep: number; // draw a line every N grid rows/columns
  hideHidden: boolean;
  relief: number; // depth exaggeration
  autoRotate: boolean;
}
export interface BodyViewerHandle {
  reset: () => void;
  exportPLY: () => Blob;
  exportGLB: () => Promise<Blob>;
  screenshot: () => string;
}

export function parseBody(buf: ArrayBuffer): BodyData {
  const dv = new DataView(buf);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== "LBD1") throw new Error("Not a Lumen body model");
  const hl = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl))) as BodyHeader;
  const n = header.cols * header.rows;
  const off = 8 + hl;
  const z = new Float32Array(buf.slice(off, off + n * 4));
  const keep = new Uint8Array(buf.slice(off + n * 4, off + n * 5));
  return { header, z, keep };
}

// perceptual depth palette (blue -> cyan -> green -> yellow -> red)
export function turbo(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t));
  const r = Math.max(0, Math.min(1, 1.6 * t - 0.4 + 0.6 * Math.sin(Math.PI * t) * (t > 0.5 ? 1 : 0)));
  const g = Math.max(0, Math.min(1, Math.sin(Math.PI * Math.min(1, t * 1.05))));
  const b = Math.max(0, Math.min(1, 1.0 - 1.8 * Math.max(0, t - 0.15) + 0.4 * Math.sin(Math.PI * t * 0.5)));
  return [r, g, b];
}

export interface BodyMesh {
  geo: THREE.BufferGeometry; // positions, uvs, colours, triangles of fully kept quads, normals
  pos: Float32Array;
  col: Float32Array;
  n: number;
}

/** Surface mesh of the body model: vertex (x, y) -> ((u-0.5)*aspect, 0.5-v, (z-0.5)*depth_scale*relief). */
export function buildBodyGeometry(data: BodyData, colors: BodyViewOptions["colors"], relief: number): BodyMesh {
  const { cols, rows, aspect, depth_scale } = data.header;
  const n = cols * rows;
  const pos = new Float32Array(n * 3);
  const uv = new Float32Array(n * 2);
  const col = new Float32Array(n * 3);
  const zs = depth_scale * relief;
  for (let i = 0; i < n; i++) {
    const x = i % cols;
    const y = Math.floor(i / cols);
    const u = x / (cols - 1);
    const v = y / (rows - 1);
    const z = data.z[i];
    pos[i * 3] = (u - 0.5) * aspect;
    pos[i * 3 + 1] = 0.5 - v;
    pos[i * 3 + 2] = (z - 0.5) * zs;
    uv[i * 2] = u;
    uv[i * 2 + 1] = 1 - v;
    let c: [number, number, number];
    if (colors === "hologram") {
      const t = Math.min(1, Math.max(0, z));
      c = [0.25 + 0.75 * t * 0.6, 0.75 + 0.25 * t, 1.0];
    } else c = turbo(z);
    col[i * 3] = c[0];
    col[i * 3 + 1] = c[1];
    col[i * 3 + 2] = c[2];
  }
  const k = data.keep;
  const tri: number[] = [];
  for (let y = 0; y < rows - 1; y++)
    for (let x = 0; x < cols - 1; x++) {
      const a = y * cols + x;
      if (k[a] && k[a + 1] && k[a + cols] && k[a + cols + 1]) tri.push(a, a + cols, a + 1, a + 1, a + cols, a + cols + 1);
    }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.setIndex(tri);
  geo.computeVertexNormals();
  return { geo, pos, col, n };
}
