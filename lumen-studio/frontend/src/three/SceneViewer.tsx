import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { buildGeometry, envFragmentShader, envVertexShader, fragmentShader, ParsedScene, STRETCH_MIN_COS, vertexShader } from "./scene";
import { exportSceneGLB, exportScenePLY } from "./exporters";
import { medianDisparity, NavController } from "./navigation";
import { postBlob } from "../api";

export interface ViewerOptions {
  fov: number; // horizontal field of view of the original photo (degrees)
  zfar: number; // depth range (far / near)
  mode: "color" | "depth" | "layers";
  pointCloud: boolean;
  showBackground: boolean;
  constrain: boolean;
  controls: "orbit" | "walk";
  motion: "none" | "sway" | "dolly" | "circle";
  marginFade: number;
}

export interface ViewerHandle {
  reset: () => void;
  exportGLB: () => Promise<Blob>;
  exportPLY: () => Promise<Blob>;
  screenshot: () => string;
}

interface Props {
  scene: ParsedScene;
  /** Engine file names of the reconstruction: used to fetch the far environment panorama. */
  files?: { scene: string; main_texture: string; bg_texture?: string } | null;
  mainTexture: string;
  bgTexture: string;
  backTexture?: string;
  options: ViewerOptions;
  onStatus?: (s: { outside: boolean; distance: number }) => void;
  /** The WebGL renderer could not be created or failed while drawing (e.g. no WebGL in this browser). */
  onRendererError?: (e: Error) => void;
  /** The browser took the WebGL context away (GPU reset, too many contexts, ...). */
  onContextLost?: () => void;
}

const asError = (e: unknown) => (e instanceof Error ? e : new Error(String(e)));

/** Interactive renderer for a reconstructed layered-depth scene. */
export const SceneViewer = forwardRef<ViewerHandle, Props>(function SceneViewer({ scene, files, mainTexture, bgTexture, backTexture, options, onStatus, onRendererError, onContextLost }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<any>({});
  const opts = useRef(options);
  opts.current = options;
  const cbs = useRef({ onRendererError, onContextLost });
  cbs.current = { onRendererError, onContextLost };
  const statusRef = useRef(onStatus);
  statusRef.current = onStatus;
  const filesRef = useRef(files);
  filesRef.current = files;

  useEffect(() => {
    const el = host.current!;
    const fail = (e: unknown) => {
      // without a handler, let the surrounding error boundary show the problem
      if (cbs.current.onRendererError) cbs.current.onRendererError(asError(e));
      else throw asError(e);
    };
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    } catch (e) {
      fail(e);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0a0c10);
    el.appendChild(renderer.domElement);
    const s3 = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 500);
    const h = scene.header;
    const loader = new THREE.TextureLoader();
    const mk = (url: string) => {
      const t = loader.load(url);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = renderer.capabilities.getMaxAnisotropy();
      t.generateMipmaps = true;
      t.minFilter = THREE.LinearMipmapLinearFilter;
      return t;
    };
    // far environment panorama (fetched below); a 1x1 placeholder until it arrives
    const envTex = { current: new THREE.DataTexture(new Uint8Array([10, 12, 16, 255]), 1, 1) as THREE.Texture };
    envTex.current.needsUpdate = true;
    const uniforms = (map: THREE.Texture, tint: number) => ({
      map: { value: map },
      tanHalfX: { value: Math.tan((options.fov * Math.PI) / 360) },
      aspect: { value: h.aspect },
      extent: { value: new THREE.Vector2(h.extent_x, h.extent_y) },
      zfar: { value: options.zfar },
      mode: { value: 0 },
      layerTint: { value: tint },
      marginFade: { value: options.marginFade },
      pointSize: { value: 2.0 },
      dim: { value: 1.0 },
      envMap: { value: envTex.current },
      hasEnv: { value: 0 },
      stretchCos: { value: tint < 0.8 ? STRETCH_MIN_COS : 0 }, // main + background layer, not the backdrop
    });
    const mainTex = mk(mainTexture);
    const bgTex = mk(bgTexture);
    const matMain = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms: uniforms(mainTex, 0) });
    const matBg = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms: uniforms(bgTex, 0.55) });
    const gMain = buildGeometry(h.grid_w, h.grid_h, scene.disp, scene.keep);
    const gBg = buildGeometry(h.grid_w, h.grid_h, scene.bgDisp, scene.keepBg, true);
    const meshMain = new THREE.Mesh(gMain, matMain);
    const meshBg = new THREE.Mesh(gBg, matBg);
    meshMain.frustumCulled = meshBg.frustumCulled = false;
    // all-vertex point cloud of the main layer
    const gPts = new THREE.BufferGeometry();
    gPts.setAttribute("uvd", gMain.getAttribute("uvd"));
    gPts.setAttribute("position", gMain.getAttribute("position"));
    const matPts = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms: uniforms(mainTex, 0) });
    matPts.uniforms.stretchCos.value = 0;
    const points = new THREE.Points(gPts, matPts);
    points.frustumCulled = false;
    s3.add(meshMain, meshBg, points);
    // backdrop: coarse far surface with a blurred texture behind everything
    let meshBack: THREE.Mesh | null = null;
    let backTex: THREE.Texture | null = null;
    if (scene.backDisp && h.back_w && h.back_h) {
      backTex = mk(backTexture || bgTexture);
      const matBack = new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms: uniforms(backTex, 0.85) });
      matBack.uniforms.dim.value = 0.96;
      const keepAll = new Uint8Array((h.back_w - 1) * (h.back_h - 1)).fill(1);
      meshBack = new THREE.Mesh(buildGeometry(h.back_w, h.back_h, scene.backDisp, keepAll), matBack);
      meshBack.frustumCulled = false;
      s3.add(meshBack);
    }

    // environment sphere: drawn first, at the far plane, following the camera
    const matEnv = new THREE.ShaderMaterial({
      vertexShader: envVertexShader,
      fragmentShader: envFragmentShader,
      uniforms: { envMap: { value: envTex.current }, hasEnv: { value: 0 }, mode: { value: 0 } },
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    });
    const meshEnv = new THREE.Mesh(new THREE.SphereGeometry(100, 48, 24), matEnv);
    meshEnv.frustumCulled = false;
    meshEnv.renderOrder = -1000;
    meshEnv.visible = false;
    s3.add(meshEnv);
    const allEnvMats = () => [matEnv, matMain, matBg, matPts, ...(meshBack ? [meshBack.material as THREE.ShaderMaterial] : [])];
    let envFov = -1;
    let envBusy = false;
    let envUrl: string | null = null;
    let disposed = false;
    const loadEnv = (fov: number) => {
      const f = filesRef.current;
      if (!f || envBusy || Math.abs(fov - envFov) < 0.5) return;
      envBusy = true;
      envFov = fov;
      postBlob("/api/recon/environment", { files: { scene: f.scene, main_texture: f.main_texture, bg_texture: f.bg_texture }, fov })
        .then(({ blob }) => {
          if (disposed) return;
          const url = URL.createObjectURL(blob);
          loader.load(url, (t) => {
            if (disposed) {
              t.dispose();
              URL.revokeObjectURL(url);
              return;
            }
            t.colorSpace = THREE.SRGBColorSpace;
            t.wrapS = THREE.RepeatWrapping;
            t.generateMipmaps = false; // the panorama is soft; no mip seam at the longitude wrap
            t.minFilter = THREE.LinearFilter;
            const old = envTex.current;
            envTex.current = t;
            for (const mt of allEnvMats()) {
              mt.uniforms.envMap.value = t;
              mt.uniforms.hasEnv.value = 1;
            }
            old.dispose();
            if (envUrl) URL.revokeObjectURL(envUrl);
            envUrl = url;
          });
        })
        .catch((e) => console.warn("[Lumen] far environment unavailable:", e))
        .finally(() => (envBusy = false));
    };

    const medD = medianDisparity(scene.disp);
    const st = state.current;
    const nav = new NavController(renderer.domElement, () => opts.current, h.safe_radius, medD);
    Object.assign(st, {
      renderer,
      camera,
      s3,
      materials: [matMain, matBg, matPts, ...(meshBack ? [meshBack.material] : [])],
      meshMain,
      meshBg,
      points,
      gMain,
      gBg,
      medD,
      nav,
    });

    const resize = () => {
      const r = el.getBoundingClientRect();
      renderer.setSize(r.width, r.height);
      camera.aspect = r.width / Math.max(r.height, 1);
      // match the original camera: the photo fills the view at the start
      const tanX = Math.tan((opts.current.fov * Math.PI) / 360);
      const tanY = tanX / h.aspect;
      const fitTanY = camera.aspect > h.aspect ? tanY : tanX / camera.aspect;
      camera.fov = (2 * Math.atan(fitTanY) * 180) / Math.PI;
      camera.updateProjectionMatrix();
      el.dataset.fov = camera.fov.toFixed(2);
      el.dataset.aspect = camera.aspect.toFixed(3);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    const dom = renderer.domElement;
    dom.tabIndex = 0;
    dom.style.outline = "none";

    const onLost = (e: Event) => {
      e.preventDefault();
      cbs.current.onContextLost?.();
    };
    dom.addEventListener("webglcontextlost", onLost);

    // ---- render loop
    let raf = 0;
    let fovSince = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      try {
        frame();
      } catch (e) {
        cancelAnimationFrame(raf);
        console.error("[Lumen] 3D viewer frame failed:", e);
        if (cbs.current.onRendererError) cbs.current.onRendererError(asError(e));
      }
    };
    const frame = () => {
      const o = opts.current;
      const pose = nav.update();
      camera.position.copy(pose.position);
      if (pose.lookAt) camera.lookAt(pose.lookAt);
      else camera.rotation.set(pose.pitch, pose.yaw, 0, "YXZ");
      meshEnv.position.copy(camera.position);
      // the environment follows the field of view (re-fetched after the slider settles)
      const now = performance.now();
      if (Math.abs(o.fov - envFov) >= 0.5) {
        if (!fovSince) fovSince = now;
        if (now - fovSince > 350) {
          fovSince = 0;
          loadEnv(o.fov);
        }
      } else fovSince = 0;
      // uniforms
      const tanX = Math.tan((o.fov * Math.PI) / 360);
      for (const mat of st.materials as THREE.ShaderMaterial[]) {
        mat.uniforms.tanHalfX.value = tanX;
        mat.uniforms.zfar.value = o.zfar;
        mat.uniforms.mode.value = o.mode === "depth" ? 1 : o.mode === "layers" ? 2 : 0;
        mat.uniforms.marginFade.value = o.marginFade;
      }
      (st.materials[0] as THREE.ShaderMaterial).uniforms.layerTint.value = 0;
      matEnv.uniforms.mode.value = o.mode === "depth" ? 1 : o.mode === "layers" ? 2 : 0;
      meshEnv.visible = matEnv.uniforms.hasEnv.value > 0.5 && o.mode !== "depth" && o.showBackground;
      meshMain.visible = !o.pointCloud;
      meshBg.visible = !o.pointCloud && o.showBackground;
      points.visible = o.pointCloud;
      if (meshBack) meshBack.visible = !o.pointCloud && o.showBackground;
      renderer.render(s3, camera);
      statusRef.current?.({ outside: pose.outside, distance: pose.distance });
    };
    loadEnv(opts.current.fov);
    tick();
    st.resize = resize;

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      nav.dispose();
      dom.removeEventListener("webglcontextlost", onLost);
      envTex.current.dispose();
      if (envUrl) URL.revokeObjectURL(envUrl);
      meshEnv.geometry.dispose();
      matEnv.dispose();
      renderer.dispose();
      gMain.dispose();
      gBg.dispose();
      gPts.dispose();
      mainTex.dispose();
      bgTex.dispose();
      backTex?.dispose();
      meshBack?.geometry.dispose();
      el.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, mainTexture, bgTexture, backTexture]);

  useEffect(() => {
    state.current.resize?.();
  }, [options.fov]);

  const projection = () => ({ fov: opts.current.fov, zfar: opts.current.zfar });

  useImperativeHandle(ref, () => ({
    reset: () => {
      state.current.nav?.reset();
    },
    screenshot: () => state.current.renderer?.domElement.toDataURL("image/jpeg", 0.92) ?? "",
    exportGLB: async () => {
      const st = state.current;
      if (!st.gMain) throw new Error("The 3D view is not ready yet.");
      return exportSceneGLB(scene, { gMain: st.gMain, gBg: st.gBg, mainTex: st.materials[0].uniforms.map.value, bgTex: st.materials[1].uniforms.map.value }, projection());
    },
    exportPLY: async () => {
      // binary PLY point cloud of the visible surface with colours from the texture
      const st = state.current;
      if (!st.gMain) throw new Error("The 3D view is not ready yet.");
      return exportScenePLY(scene, st.gMain, st.materials[0].uniforms.map.value.image, projection());
    },
  }));

  return <div ref={host} className="absolute inset-0 outline-none" data-testid="viewer" />;
});
