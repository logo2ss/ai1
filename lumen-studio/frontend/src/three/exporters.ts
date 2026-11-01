// GLB / PLY export of the reconstructed scene and the body model. Shared by the GPU viewers and the
// CPU (engine-rendered) viewers: none of this needs a WebGL context (GLTFExporter uses a 2D canvas).
import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import { bakePositions, buildGeometry, ParsedScene } from "./scene";
import type { BodyData } from "./body";

export interface SceneProjection {
  fov: number; // horizontal field of view of the original photo (degrees)
  zfar: number; // depth range
}

const unprojectParams = (scene: ParsedScene, o: SceneProjection) => ({
  tanHalfX: Math.tan((o.fov * Math.PI) / 360),
  aspect: scene.header.aspect,
  extent: [scene.header.extent_x, scene.header.extent_y] as [number, number],
  zfar: o.zfar,
});

/** Grid geometries of the visible surface and of the (sparse) synthesised background layer. */
export function sceneGeometries(scene: ParsedScene): { gMain: THREE.BufferGeometry; gBg: THREE.BufferGeometry } {
  const h = scene.header;
  return {
    gMain: buildGeometry(h.grid_w, h.grid_h, scene.disp, scene.keep),
    gBg: buildGeometry(h.grid_w, h.grid_h, scene.bgDisp, scene.keepBg, true),
  };
}

/** Textured mesh with world positions baked from the per-vertex (u, v, disparity) attribute. */
export function bakedSceneMesh(geo: THREE.BufferGeometry, tex: THREE.Texture, name: string, scene: ParsedScene, o: SceneProjection): THREE.Mesh {
  const uvd = geo.getAttribute("uvd").array as Float32Array;
  const pos = bakePositions(uvd, unprojectParams(scene, o));
  const uv = new Float32Array((uvd.length / 3) * 2);
  for (let i = 0, j = 0; i < uvd.length; i += 3, j += 2) {
    uv[j] = uvd[i];
    uv[j + 1] = 1 - uvd[i + 1];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.setIndex(geo.getIndex());
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex }));
  m.name = name;
  return m;
}

/** glTF binary with the visible surface and the estimated occluded background as textured meshes. */
export async function exportSceneGLB(
  scene: ParsedScene,
  layers: { gMain: THREE.BufferGeometry; gBg: THREE.BufferGeometry; mainTex: THREE.Texture; bgTex: THREE.Texture },
  o: SceneProjection,
): Promise<Blob> {
  const root = new THREE.Scene();
  root.add(bakedSceneMesh(layers.gMain, layers.mainTex, "visible_surface", scene, o));
  root.add(bakedSceneMesh(layers.gBg, layers.bgTex, "occluded_background_estimate", scene, o));
  const exporter = new GLTFExporter();
  const res = (await exporter.parseAsync(root, { binary: true, maxTextureSize: 4096 })) as ArrayBuffer;
  return new Blob([res], { type: "model/gltf-binary" });
}

/** Binary PLY point cloud of the visible surface with colours from the texture. */
export function exportScenePLY(scene: ParsedScene, gMain: THREE.BufferGeometry, image: CanvasImageSource, o: SceneProjection): Blob {
  const uvd = gMain.getAttribute("uvd").array as Float32Array;
  const pos = bakePositions(uvd, unprojectParams(scene, o));
  const c = document.createElement("canvas");
  c.width = scene.header.grid_w;
  c.height = scene.header.grid_h;
  const g = c.getContext("2d")!;
  g.drawImage(image, 0, 0, c.width, c.height);
  const px = g.getImageData(0, 0, c.width, c.height).data;
  const n = uvd.length / 3;
  const head = `ply\nformat binary_little_endian 1.0\ncomment Lumen Studio single-image reconstruction (estimated geometry)\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  const hb = new TextEncoder().encode(head);
  const body = new ArrayBuffer(n * 15);
  const dv = new DataView(body);
  for (let i = 0; i < n; i++) {
    dv.setFloat32(i * 15, pos[i * 3], true);
    dv.setFloat32(i * 15 + 4, pos[i * 3 + 1], true);
    dv.setFloat32(i * 15 + 8, pos[i * 3 + 2], true);
    dv.setUint8(i * 15 + 12, px[i * 4]);
    dv.setUint8(i * 15 + 13, px[i * 4 + 1]);
    dv.setUint8(i * 15 + 14, px[i * 4 + 2]);
  }
  return new Blob([hb, body], { type: "application/octet-stream" });
}

/** ASCII PLY of the kept body vertices (positions as built by buildBodyGeometry). */
export function exportBodyPLY(data: BodyData, geo: THREE.BufferGeometry): Blob {
  const p = geo.getAttribute("position").array as Float32Array;
  const ids: number[] = [];
  for (let i = 0; i < data.keep.length; i++) if (data.keep[i]) ids.push(i);
  const head = `ply\nformat ascii 1.0\ncomment Lumen Studio 3D body model (estimated from one photo)\nelement vertex ${ids.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`;
  const lines = ids.map((i) => `${p[i * 3].toFixed(5)} ${p[i * 3 + 1].toFixed(5)} ${p[i * 3 + 2].toFixed(5)}`);
  return new Blob([head, lines.join("\n")], { type: "text/plain" });
}

/** glTF binary of the body surface textured with the photo. */
export async function exportBodyGLB(geo: THREE.BufferGeometry, tex: THREE.Texture): Promise<Blob> {
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: tex }));
  const res = (await new GLTFExporter().parseAsync(mesh, { binary: true })) as ArrayBuffer;
  return new Blob([res], { type: "model/gltf-binary" });
}

/** Load an image as a texture (an <img> underneath: works without WebGL). */
export async function loadTexture(url: string): Promise<THREE.Texture> {
  const t = await new THREE.TextureLoader().loadAsync(url);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
