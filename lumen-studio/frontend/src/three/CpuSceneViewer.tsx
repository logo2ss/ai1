import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { postBlob } from "../api";
import type { ParsedScene } from "./scene";
import type { ViewerHandle, ViewerOptions } from "./SceneViewer";
import { exportSceneGLB, exportScenePLY, loadTexture, sceneGeometries } from "./exporters";
import { CpuFrameOverlay, describeRenderError, FrameStats, FrameStream } from "./cpuFrames";
import { medianDisparity, NavController } from "./navigation";

export interface SceneFiles {
  scene: string;
  main_texture: string;
  bg_texture: string;
  back_texture?: string | null;
}

interface Props {
  scene: ParsedScene;
  /** Engine file names of the reconstruction (the engine renders from these). */
  files: SceneFiles;
  mainTexture: string;
  bgTexture: string;
  backTexture?: string;
  options: ViewerOptions;
  onStatus?: (s: { outside: boolean; distance: number }) => void;
}

const r5 = (v: number) => Math.round(v * 1e5) / 1e5;

/**
 * The reconstructed scene drawn by the engine's CPU renderer, for browsers without WebGL.
 * Same controls, camera math and imperative handle as SceneViewer; frames come from
 * POST /api/recon/render and are shown in an <img> stretched over the viewer (the frame is
 * rendered at the viewer's aspect ratio, so the framing matches the GPU viewer).
 */
export const CpuSceneViewer = forwardRef<ViewerHandle, Props>(function CpuSceneViewer({ scene, files, mainTexture, bgTexture, options, onStatus }, ref) {
  const host = useRef<HTMLDivElement>(null);
  const img = useRef<HTMLImageElement>(null);
  const state = useRef<any>({});
  const stream = useRef<FrameStream | null>(null);
  const opts = useRef(options);
  opts.current = options;
  const statusCb = useRef(onStatus);
  statusCb.current = onStatus;
  const filesRef = useRef(files);
  filesRef.current = files;
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<FrameStats | null>(null);

  useEffect(() => {
    const el = host.current!;
    const h = scene.header;
    setReady(false);
    setError(null);
    setStats(null);
    const frames = new FrameStream(() => img.current, {
      frame: (f) => {
        setStats(f);
        setReady(true);
      },
      error: (e) => setError(e ? describeRenderError(e) : null),
    });
    stream.current = frames;

    const nav = new NavController(el, () => opts.current, h.safe_radius, medianDisparity(scene.disp));
    state.current = { nav };

    let cw = 0;
    let ch = 0;
    const measure = () => {
      const r = el.getBoundingClientRect();
      cw = r.width;
      ch = r.height;
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();

    // ---- pose loop (same NavController as SceneViewer), then ask the engine for a frame
    let raf = 0;
    let lastStatus = { outside: false, distance: -1 };
    const tick = () => {
      raf = requestAnimationFrame(tick);
      try {
        frame();
      } catch (e) {
        console.error("[Lumen] CPU 3D viewer update failed:", e);
      }
    };
    const frame = () => {
      const o = opts.current;
      const pose = nav.update();
      const cam = pose.position;
      const camera = {
        position: [r5(cam.x), r5(cam.y), r5(cam.z)],
        yaw: pose.lookAt ? 0 : r5(pose.yaw),
        pitch: pose.lookAt ? 0 : r5(pose.pitch),
        look_at: pose.lookAt ? [r5(pose.lookAt.x), r5(pose.lookAt.y), r5(pose.lookAt.z)] : null,
      };
      if (pose.outside !== lastStatus.outside || Math.abs(pose.distance - lastStatus.distance) > 0.005) {
        lastStatus = { outside: pose.outside, distance: pose.distance };
        statusCb.current?.(lastStatus);
      }
      const f = filesRef.current;
      const key = JSON.stringify([camera, o.fov, o.zfar, o.mode, o.showBackground, o.marginFade, f.scene, f.main_texture, f.bg_texture, f.back_texture ?? null]);
      frames.update(key, cw, ch, pose.moving, (quality, width, height) =>
        postBlob("/api/recon/render", {
          files: { scene: f.scene, main_texture: f.main_texture, bg_texture: f.bg_texture, back_texture: f.back_texture ?? null },
          camera,
          fov: o.fov,
          zfar: o.zfar,
          width,
          height,
          quality,
          show_background: o.showBackground,
          mode: o.mode,
          margin_fade: o.marginFade,
          environment: true,
        }),
      );
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      nav.dispose();
      frames.dispose();
      if (stream.current === frames) stream.current = null;
    };
  }, [scene, files.scene, files.main_texture, files.bg_texture, files.back_texture]);

  const projection = () => ({ fov: opts.current.fov, zfar: opts.current.zfar });

  useImperativeHandle(ref, () => ({
    reset: () => {
      state.current.nav?.reset();
    },
    screenshot: () => stream.current?.snapshot("image/jpeg", 0.92) ?? "",
    exportGLB: async () => {
      const { gMain, gBg } = sceneGeometries(scene);
      const [mainTex, bgTex] = await Promise.all([loadTexture(mainTexture), loadTexture(bgTexture)]);
      try {
        return await exportSceneGLB(scene, { gMain, gBg, mainTex, bgTex }, projection());
      } finally {
        gMain.dispose();
        gBg.dispose();
        mainTex.dispose();
        bgTex.dispose();
      }
    },
    exportPLY: async () => {
      // binary PLY point cloud of the visible surface with colours from the texture
      const { gMain, gBg } = sceneGeometries(scene);
      const tex = await loadTexture(mainTexture);
      try {
        return exportScenePLY(scene, gMain, tex.image as HTMLImageElement, projection());
      } finally {
        gMain.dispose();
        gBg.dispose();
        tex.dispose();
      }
    },
  }));

  return (
    <div ref={host} tabIndex={0} className="absolute inset-0 outline-none bg-[#0a0c10] touch-none cursor-grab active:cursor-grabbing" data-testid="viewer" data-renderer="cpu">
      <img ref={img} alt="3D view" draggable={false} className="absolute inset-0 w-full h-full pointer-events-none select-none" style={{ objectFit: "fill", visibility: ready ? "visible" : "hidden" }} data-testid="cpu-frame" />
      <CpuFrameOverlay
        ready={ready}
        error={error}
        stats={stats}
        onRetry={() => {
          setError(null);
          stream.current?.retryNow();
        }}
      />
    </div>
  );
});
