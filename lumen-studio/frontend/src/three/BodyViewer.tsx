import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { BodyData, BodyViewerHandle, BodyViewOptions, buildBodyGeometry } from "./body";
import { exportBodyGLB, exportBodyPLY } from "./exporters";

export type { BodyData, BodyHeader, BodyViewerHandle, BodyViewOptions } from "./body";
export { parseBody } from "./body";

const asError = (e: unknown) => (e instanceof Error ? e : new Error(String(e)));

interface Props {
  data: BodyData;
  texture: string;
  options: BodyViewOptions;
  /** The WebGL renderer could not be created or failed while drawing (e.g. no WebGL in this browser). */
  onRendererError?: (e: Error) => void;
  /** The browser took the WebGL context away. */
  onContextLost?: () => void;
}

/** Interactive renderer of the dense 3D body model (wireframe, scan lines, points, solid). */
export const BodyViewer = forwardRef<BodyViewerHandle, Props>(function BodyViewer({ data, texture, options, onRendererError, onContextLost }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const st = useRef<any>({});
  const cbs = useRef({ onRendererError, onContextLost });
  cbs.current = { onRendererError, onContextLost };

  // ---- scene setup (once per model)
  useEffect(() => {
    const el = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    } catch (e) {
      // without a handler, let the surrounding error boundary show the problem
      if (cbs.current.onRendererError) {
        cbs.current.onRendererError(asError(e));
        return;
      }
      throw asError(e);
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x07090d);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 50);
    camera.position.set(0, 0.05, 2.3);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controls.minDistance = 0.4;
    controls.maxDistance = 6;
    scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242c, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(1.2, 1.5, 2.0);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x8f83ff, 0.8);
    rim.position.set(-1.5, 0.5, -1.5);
    scene.add(rim);
    // floor grid for spatial reference
    const grid = new THREE.GridHelper(4, 40, 0x2b3242, 0x161b25);
    grid.position.y = -0.52;
    scene.add(grid);
    const tex = new THREE.TextureLoader().load(texture);
    tex.colorSpace = THREE.SRGBColorSpace;
    Object.assign(st.current, { renderer, scene, camera, controls, tex, group: new THREE.Group() });
    scene.add(st.current.group);
    const resize = () => {
      const r = el.getBoundingClientRect();
      renderer.setSize(r.width, r.height);
      camera.aspect = r.width / Math.max(1, r.height);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    const onLost = (e: Event) => {
      e.preventDefault();
      cbs.current.onContextLost?.();
    };
    renderer.domElement.addEventListener("webglcontextlost", onLost);
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      try {
        controls.update();
        renderer.render(scene, camera);
      } catch (e) {
        cancelAnimationFrame(raf);
        console.error("[Lumen] 3D body viewer frame failed:", e);
        cbs.current.onRendererError?.(asError(e));
      }
    };
    tick();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("webglcontextlost", onLost);
      controls.dispose();
      renderer.dispose();
      tex.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [data, texture]);

  // ---- (re)build geometry when the model or style options change
  useEffect(() => {
    const s = st.current;
    if (!s.group) return;
    s.group.clear();
    const { cols, rows } = data.header;
    const { geo, pos, col, n } = buildBodyGeometry(data, options.colors, options.relief);
    const k = data.keep;
    s.geo = geo;
    const usePhoto = options.colors === "photo";
    if (options.style === "solid") {
      const mat = usePhoto ? new THREE.MeshStandardMaterial({ map: s.tex, roughness: 0.85, metalness: 0 }) : new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0.05 });
      s.group.add(new THREE.Mesh(geo, mat));
    } else if (options.style === "points") {
      const pts = new THREE.BufferGeometry();
      const idx: number[] = [];
      for (let i = 0; i < n; i++) if (k[i]) idx.push(i);
      const p2 = new Float32Array(idx.length * 3);
      const c2 = new Float32Array(idx.length * 3);
      // photo colours are sampled on the GPU via a texture-mapped points shader would be overkill: use vertex colours
      const img: HTMLImageElement | undefined = s.tex.image;
      let px: Uint8ClampedArray | null = null;
      if (usePhoto && img && img.width) {
        const cv = document.createElement("canvas");
        cv.width = cols;
        cv.height = rows;
        const g = cv.getContext("2d")!;
        g.drawImage(img, 0, 0, cols, rows);
        px = g.getImageData(0, 0, cols, rows).data;
      }
      idx.forEach((i, j) => {
        p2.set(pos.subarray(i * 3, i * 3 + 3), j * 3);
        if (px) c2.set([px[i * 4] / 255, px[i * 4 + 1] / 255, px[i * 4 + 2] / 255], j * 3);
        else c2.set(col.subarray(i * 3, i * 3 + 3), j * 3);
      });
      pts.setAttribute("position", new THREE.BufferAttribute(p2, 3));
      pts.setAttribute("color", new THREE.BufferAttribute(c2, 3));
      const size = Math.max(0.0025, 1.6 / Math.max(cols, rows));
      s.group.add(new THREE.Points(pts, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true })));
    } else {
      // wireframe / scan lines along the surface, optionally occluded by the body itself
      const step = Math.max(1, Math.round(options.lineStep));
      const li: number[] = [];
      for (let y = 0; y < rows; y += options.style === "scanlines" ? Math.max(1, Math.round(step / 2)) : step)
        for (let x = 0; x < cols - 1; x++) {
          const a = y * cols + x;
          if (k[a] && k[a + 1]) li.push(a, a + 1);
        }
      if (options.style === "wireframe")
        for (let x = 0; x < cols; x += step)
          for (let y = 0; y < rows - 1; y++) {
            const a = y * cols + x;
            if (k[a] && k[a + cols]) li.push(a, a + cols);
          }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute("position", geo.getAttribute("position"));
      lg.setAttribute("color", geo.getAttribute("color"));
      lg.setIndex(li);
      const lmat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: options.colors === "hologram" ? 0.85 : 0.95 });
      s.group.add(new THREE.LineSegments(lg, lmat));
      if (options.hideHidden) {
        const occ = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x07090d, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
        s.group.add(occ);
      }
    }
  }, [data, options.style, options.colors, options.lineStep, options.hideHidden, options.relief]);

  useEffect(() => {
    if (st.current.controls) {
      st.current.controls.autoRotate = options.autoRotate;
      st.current.controls.autoRotateSpeed = 1.2;
    }
  }, [options.autoRotate, data]);

  useImperativeHandle(ref, () => ({
    reset: () => {
      const s = st.current;
      s.camera.position.set(0, 0.05, 2.3);
      s.controls.target.set(0, 0, 0);
    },
    screenshot: () => st.current.renderer?.domElement.toDataURL("image/png") ?? "",
    exportPLY: () => exportBodyPLY(data, st.current.geo),
    exportGLB: () => exportBodyGLB(st.current.geo, st.current.tex),
  }));

  return <div ref={host} className="absolute inset-0" data-testid="body-viewer" />;
});
