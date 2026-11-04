"use client";

/**
 * One schematic per technique.
 *
 * These are diagrams of *what gets measured*, not decoration: each shows the transform, the
 * quantity taken from it, and the failure that raises the score. They are hand-drawn SVG rather
 * than illustrations because they have to stay legible in both themes and at 320 px wide, and
 * because a schematic that shares the page's colour tokens can be read as part of the instrument.
 *
 * `stroke="currentColor"` throughout, with the accent used once per figure — the point is the
 * relationship, not the palette.
 */

import * as React from "react";
import { cn } from "@/lib/utils";

const FRAME = "0 0 220 96";

function Svg({ children, label, className }: { children: React.ReactNode; label: string; className?: string }) {
  return (
    // Height follows width rather than being pinned to a row of 6rem. With a fixed height,
    // `preserveAspectRatio` fits the 220x96 frame to whichever dimension is tighter, so on desktop
    // the figure rendered at its literal 220 units inside a 700px card and left the rest empty, and
    // its 6.5-unit labels stayed 6.5px on every screen — illegible at exactly the width the note
    // above promises to handle. Scaling with the container makes the labels grow with the card.
    <svg viewBox={FRAME} role="img" aria-label={label}
         className={cn("h-auto w-full max-w-[26rem] text-muted-foreground", className)}>
      {children}
    </svg>
  );
}

const Box = ({ x, y, w = 44, h = 26, text }: { x: number; y: number; w?: number; h?: number; text: string }) => (
  <g>
    <rect x={x} y={y} width={w} height={h} rx={3} fill="var(--muted)" stroke="currentColor" strokeWidth={0.8} opacity={0.55} />
    <text x={x + w / 2} y={y + h / 2 + 3} textAnchor="middle" fontSize={7.5} fill="currentColor">
      {text}
    </text>
  </g>
);

const Arrow = ({ x1, x2, y }: { x1: number; x2: number; y: number }) => (
  <g stroke="currentColor" strokeWidth={0.9}>
    <line x1={x1} y1={y} x2={x2 - 3} y2={y} />
    <path d={`M ${x2} ${y} l -4 -2.2 v 4.4 z`} fill="currentColor" stroke="none" />
  </g>
);

const OUT = "var(--primary)";

export function TechniqueDiagram({ name, className }: { name: string; className?: string }) {
  const figure = FIGURES[name];
  if (!figure) {
    return (
      <Svg label="Generic measurement flow" className={className}>
        <Box x={6} y={35} text="image" />
        <Arrow x1={52} x2={88} y={48} />
        <Box x={90} y={35} w={50} text="measure" />
        <Arrow x1={142} x2={172} y={48} />
        <Box x={174} y={35} w={40} text="score" />
      </Svg>
    );
  }
  return figure(className);
}

type Figure = (className?: string) => React.ReactElement;

const FIGURES: Record<string, Figure> = {
  metadata: (className) => (
    <Svg label="Metadata: text chunks are read and matched against generator signatures" className={className}>
      <Box x={4} y={35} w={40} text="file" />
      <Arrow x1={46} x2={74} y={48} />
      <g>
        <rect x={76} y={14} width={62} height={13} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <text x={107} y={23.5} textAnchor="middle" fontSize={6.5} fill="currentColor">EXIF Make=Canon</text>
        <rect x={76} y={31} width={62} height={13} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <text x={107} y={40.5} textAnchor="middle" fontSize={6.5} fill="currentColor">XMP Software=…</text>
        <rect x={76} y={48} width={62} height={13} rx={2} fill="none" stroke={OUT} strokeWidth={1} />
        <text x={107} y={57.5} textAnchor="middle" fontSize={6.5} fill={OUT}>prompt= "…"</text>
        <rect x={76} y={65} width={62} height={13} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <text x={107} y={74.5} textAnchor="middle" fontSize={6.5} fill="currentColor">GPS (redacted)</text>
      </g>
      <Arrow x1={140} x2={168} y={48} />
      <Box x={170} y={35} w={44} text="flag" />
    </Svg>
  ),

  c2pa: (className) => (
    <Svg label="C2PA: a signed manifest is verified against its issuer certificate" className={className}>
      <Box x={4} y={35} w={40} text="jumbf" />
      <Arrow x1={46} x2={72} y={48} />
      <g stroke="currentColor" strokeWidth={0.8} fill="none">
        <rect x={74} y={20} width={54} height={17} rx={2} />
        <rect x={74} y={41} width={54} height={17} rx={2} />
        <rect x={74} y={62} width={54} height={17} rx={2} />
      </g>
      <text x={101} y={31} textAnchor="middle" fontSize={6.5} fill="currentColor">claim</text>
      <text x={101} y={52} textAnchor="middle" fontSize={6.5} fill="currentColor">assertions</text>
      <text x={101} y={73} textAnchor="middle" fontSize={6.5} fill="currentColor">signature</text>
      <Arrow x1={130} x2={156} y={48} />
      <g>
        <circle cx={172} cy={40} r={13} fill="none" stroke={OUT} strokeWidth={1.2} />
        <text x={172} y={43} textAnchor="middle" fontSize={7} fill={OUT}>verify</text>
        <text x={172} y={68} textAnchor="middle" fontSize={6.5} fill="currentColor">issuer + hash chain</text>
      </g>
    </Svg>
  ),

  ela: (className) => (
    <Svg label="ELA: the file is re-saved at its own quality and the difference is measured per region" className={className}>
      <Box x={4} y={12} w={42} text="original" />
      <Box x={4} y={58} w={42} text="re-save" />
      <g stroke="currentColor" strokeWidth={0.8} fill="none">
        <path d="M 48 24 q 14 18 28 22" />
        <path d="M 48 70 q 14 -18 28 -22" />
      </g>
      <text x={80} y={50} textAnchor="middle" fontSize={11} fill={OUT}>−</text>
      <Box x={94} y={35} w={52} text="abs diff" />
      <Arrow x1={148} x2={172} y={48} />
      <g>
        <rect x={174} y={26} width={40} height={20} rx={2} fill="currentColor" opacity={0.15} />
        <rect x={174} y={26} width={40} height={20} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <rect x={196} y={50} width={18} height={18} rx={2} fill={OUT} opacity={0.5} />
        <text x={194} y={80} textAnchor="middle" fontSize={6.5} fill="currentColor">hot region = spliced</text>
      </g>
    </Svg>
  ),

  jpeg_ghost: (className) => (
    <Svg label="JPEG ghost: DCT coefficient striation reveals a second compression history" className={className}>
      <g stroke="currentColor" strokeWidth={0.7} opacity={0.7}>
        {Array.from({ length: 9 }, (_, i) => (
          <line key={i} x1={8} x2={78} y1={16 + i * 8} y2={16 + i * 8} strokeDasharray={i % 3 === 0 ? "2 2" : undefined} />
        ))}
      </g>
      <text x={43} y={92} textAnchor="middle" fontSize={6.5} fill="currentColor">periodic row peaks</text>
      <Arrow x1={82} x2={106} y={48} />
      <g>
        <rect x={108} y={22} width={46} height={46} rx={2} fill="none" stroke="currentColor" strokeWidth={0.8} />
        <g fill={OUT}>
          {Array.from({ length: 6 }, (_, i) => (
            <rect key={i} x={112} y={26 + i * 7} width={38} height={2} />
          ))}
        </g>
        <text x={131} y={80} textAnchor="middle" fontSize={6.5} fill="currentColor">ghost mask</text>
      </g>
      <Arrow x1={156} x2={178} y={48} />
      <Box x={180} y={35} w={36} text="score" />
    </Svg>
  ),

  block_grid: (className) => (
    <Svg label="Block grid: two 8-by-8 lattices that do not share a phase" className={className}>
      <g stroke="currentColor" strokeWidth={0.6} opacity={0.7}>
        {Array.from({ length: 6 }, (_, i) => (
          <g key={i}>
            <line x1={10 + i * 14} x2={10 + i * 14} y1={14} y2={78} />
            <line x1={10} x2={80} y1={14 + i * 13} y2={14 + i * 13} />
          </g>
        ))}
      </g>
      <g stroke={OUT} strokeWidth={0.7} transform="translate(5,6)">
        {Array.from({ length: 5 }, (_, i) => (
          <g key={i}>
            <line x1={95 + i * 14} x2={95 + i * 14} y1={14} y2={78} />
            <line x1={95} x2={165} y1={14 + i * 13} y2={14 + i * 13} />
          </g>
        ))}
      </g>
      <text x={45} y={90} textAnchor="middle" fontSize={6.5} fill="currentColor">phase 0</text>
      <text x={130} y={90} textAnchor="middle" fontSize={6.5} fill={OUT}>phase offset</text>
      <Arrow x1={170} x2={186} y={48} />
      <Box x={188} y={35} w={28} text="Δφ" />
    </Svg>
  ),

  noise: (className) => (
    <Svg label="Noise: per-block MAD gives each region its own grain estimate" className={className}>
      <g>
        {Array.from({ length: 12 }, (_, i) => {
          const cx = 12 + (i % 4) * 24;
          const cy = 16 + Math.floor(i / 4) * 22;
          const grain = [0.9, 0.3, 0.5, 1.2, 0.4, 0.4, 1.4, 0.5, 0.6, 1.1, 0.4, 0.35][i];
          return (
            <g key={i}>
              <rect x={cx} y={cy} width={22} height={20} rx={2} fill="none" stroke="currentColor" strokeWidth={0.5} opacity={0.6} />
              {Array.from({ length: Math.round(grain * 12) }, (_, j) => (
                <circle
                  key={j}
                  cx={cx + 3 + ((j * 7) % 17)}
                  cy={cy + 3 + ((j * 5) % 14)}
                  r={0.8}
                  fill={grain > 1 ? OUT : "currentColor"}
                  opacity={grain > 1 ? 0.9 : 0.45}
                />
              ))}
            </g>
          );
        })}
      </g>
      <Arrow x1={112} x2={136} y={48} />
      <Box x={138} y={30} w={46} text="MAD spread" />
      <text x={161} y={70} textAnchor="middle" fontSize={6.5} fill={OUT}>one block is smoother</text>
    </Svg>
  ),

  copy_move: (className) => (
    <Svg label="Copy-move: keypoints inside one image match each other at a fixed displacement" className={className}>
      <rect x={8} y={14} width={70} height={64} rx={3} fill="none" stroke="currentColor" strokeWidth={0.8} />
      <rect x={20} y={24} width={22} height={18} rx={2} fill="currentColor" opacity={0.16} />
      <rect x={48} y={50} width={22} height={18} rx={2} fill={OUT} opacity={0.22} />
      <g stroke={OUT} strokeWidth={0.7} strokeDasharray="2 2">
        <line x1={26} y1={30} x2={54} y2={56} />
        <line x1={36} y1={34} x2={64} y2={60} />
        <line x1={30} y1={40} x2={58} y2={66} />
      </g>
      <Arrow x1={82} x2={106} y={46} />
      <g>
        <rect x={108} y={20} width={52} height={52} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        {[0, 1, 2, 3, 4].map((i) => (
          <rect key={i} x={112 + i * 9} y={60 - [6, 12, 30, 14, 8][i]} width={6} height={[6, 12, 30, 14, 8][i]} fill={i === 2 ? OUT : "currentColor"} opacity={i === 2 ? 0.85 : 0.4} />
        ))}
        <text x={134} y={82} textAnchor="middle" fontSize={6.5} fill="currentColor">displacement votes</text>
      </g>
      <Arrow x1={162} x2={180} y={46} />
      <Box x={182} y={33} w={32} text="pairs" />
    </Svg>
  ),

  face: (className) => (
    <Svg label="Face: landmarks, a blending seam, and asymmetric specular highlights" className={className}>
      <g fill="none" stroke="currentColor" strokeWidth={0.9}>
        <ellipse cx={62} cy={46} rx={34} ry={38} />
        <circle cx={49} cy={38} r={2} fill="currentColor" />
        <circle cx={75} cy={38} r={2} fill="currentColor" />
        <path d="M 44 32 q 5 -4 11 -2" />
        <path d="M 69 32 q 5 -3 11 1" />
        <path d="M 62 44 v 8" />
        <path d="M 54 60 q 8 5 16 0" />
      </g>
      <g stroke={OUT} strokeWidth={1} strokeDasharray="3 2">
        <path d="M 62 8 q 3 38 0 76" />
      </g>
      <text x={62} y={92} textAnchor="middle" fontSize={6.5} fill={OUT}>seam at the jaw line</text>
      <Arrow x1={100} x2={120} y={46} />
      <g>
        <circle cx={140} cy={30} r={9} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <circle cx={137} cy={27} r={2} fill={OUT} />
        <circle cx={166} cy={30} r={9} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <circle cx={169} cy={34} r={2} fill="currentColor" />
        <text x={153} y={54} textAnchor="middle" fontSize={6.5} fill="currentColor">highlight side differs</text>
        <text x={153} y={72} textAnchor="middle" fontSize={6.5} fill="currentColor">lighting symmetry</text>
      </g>
    </Svg>
  ),

  frequency: (className) => (
    <Svg label="Frequency: the radial spectrum decays like a natural image, or it does not" className={className}>
      <g>
        <rect x={8} y={12} width={70} height={70} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <circle cx={43} cy={47} r={3} fill={OUT} />
        <g stroke="currentColor" strokeWidth={0.5} opacity={0.5}>
          <circle cx={43} cy={47} r={12} />
          <circle cx={43} cy={47} r={22} />
          <circle cx={43} cy={47} r={32} />
        </g>
        {Array.from({ length: 4 }, (_, i) => (
          <line key={i} x1={43} y1={14} x2={43} y2={80} stroke={OUT} strokeWidth={0.5} opacity={0.6} transform={`rotate(${i * 45} 43 47)`} />
        ))}
        <text x={43} y={92} textAnchor="middle" fontSize={6.5} fill="currentColor">axis peaks = upsample</text>
      </g>
      <Arrow x1={82} x2={104} y={46} />
      <g>
        <rect x={106} y={16} width={60} height={60} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <polyline points="110,22 122,40 140,54 162,68" fill="none" stroke="currentColor" strokeWidth={1} />
        <polyline points="110,20 126,34 144,58 162,72" fill="none" stroke={OUT} strokeWidth={1} strokeDasharray="3 2" />
        <text x={136} y={86} textAnchor="middle" fontSize={6.5} fill="currentColor">log-log slope fit</text>
      </g>
      <Arrow x1={168} x2={184} y={46} />
      <Box x={186} y={33} w={28} text="slope" />
    </Svg>
  ),

  cnn: (className) => (
    <Svg label="Learned detector: one logit, its gradient mapped back onto the image" className={className}>
      <Box x={4} y={33} w={34} text="224²" />
      <Arrow x1={40} x2={56} y={46} />
      <g stroke="currentColor" strokeWidth={0.7} fill="var(--muted)">
        <rect x={58} y={30} width={16} height={32} rx={2} />
        <rect x={78} y={26} width={16} height={40} rx={2} />
        <rect x={98} y={22} width={16} height={48} rx={2} />
      </g>
      <text x={86} y={82} textAnchor="middle" fontSize={6.5} fill="currentColor"> EfficientNet-B0</text>
      <Arrow x1={116} x2={132} y={46} />
      <Box x={134} y={34} w={34} text="logit" />
      <Arrow x1={170} x2={184} y={46} />
      <g>
        <rect x={186} y={26} width={28} height={28} rx={2} fill="none" stroke="currentColor" strokeWidth={0.7} />
        <rect x={194} y={34} width={12} height={12} fill={OUT} opacity={0.55} />
        <text x={200} y={68} textAnchor="middle" fontSize={6.5} fill={OUT}>Grad-CAM</text>
      </g>
      <text x={151} y={20} textAnchor="middle" fontSize={6.5} fill="currentColor">σ(·) → p(synthetic)</text>
    </Svg>
  ),
};
