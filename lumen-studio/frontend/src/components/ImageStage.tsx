import { ReactNode, useEffect, useRef, useState } from "react";

/** Displays an image fitted into its container with an SVG overlay in image pixel coordinates. */
export function ImageStage({
  src,
  width,
  height,
  children,
  onPointer,
  cursor,
}: {
  src: string;
  width: number;
  height: number;
  children?: ReactNode;
  onPointer?: (type: "down" | "move" | "up", p: [number, number], e: React.PointerEvent) => void;
  cursor?: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [sz, setSz] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const ro = new ResizeObserver(() => {
      const r = box.current!.getBoundingClientRect();
      const s = Math.min(r.width / width, r.height / height);
      setSz({ w: width * s, h: height * s });
    });
    ro.observe(box.current!);
    return () => ro.disconnect();
  }, [width, height]);
  const toImg = (e: React.PointerEvent): [number, number] => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height];
  };
  return (
    <div ref={box} className="w-full h-full flex items-center justify-center overflow-hidden">
      <div
        className="relative"
        style={{ width: sz.w, height: sz.h, cursor }}
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture?.(e.pointerId);
          onPointer?.("down", toImg(e), e);
        }}
        onPointerMove={(e) => onPointer?.("move", toImg(e), e)}
        onPointerUp={(e) => onPointer?.("up", toImg(e), e)}
      >
        <img src={src} className="absolute inset-0 w-full h-full select-none" draggable={false} alt="" />
        <svg className="absolute inset-0 w-full h-full overflow-visible" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {children}
        </svg>
      </div>
    </div>
  );
}
