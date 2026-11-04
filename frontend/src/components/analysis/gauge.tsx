"use client";

/**
 * The headline gauge.
 *
 * Likelihood and reliability are two different numbers and are drawn as two different instruments:
 * a gauge for the probability and a separate meter for how much of the evidence actually arrived.
 * Merging them — the obvious shortcut — would let a 0.9 computed from one working check look
 * identical to a 0.9 computed from ten, which is the exact confusion this build exists to avoid.
 *
 * The arc is an SVG path with `strokeDasharray`, animated by Framer Motion. It is not a rotated
 * needle: a needle on a scale whose middle is meaningless (0.5 is not "medium risk", it is the
 * crossover between two different claims) implies a reading the number does not support.
 */

import * as React from "react";
import { motion, useReducedMotion } from "framer-motion";
import { BAND_COLORS, bandFor, pct, prob } from "@/lib/format";
import type { RiskBand } from "@/lib/types";
import { cn } from "@/lib/utils";

const R = 78;
const CX = 100;
const CY = 100;
/** Half-circle arc, drawn left to right, so the sweep direction matches the number. */
const ARC = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`;

export interface GaugeProps {
  /** 0..1 overall likelihood of the reported claim. `null` renders an empty instrument. */
  value: number | null;
  band?: RiskBand;
  label: string;
  caption?: string;
  size?: "sm" | "lg";
  className?: string;
}

export function Gauge({ value, band, label, caption, size = "lg", className }: GaugeProps) {
  const reduce = useReducedMotion();
  const measured = value != null && !Number.isNaN(value);
  const clamped = measured ? Math.min(1, Math.max(0, value as number)) : 0;
  const tone = band ?? bandFor(value);
  const color = BAND_COLORS[tone];
  const length = Math.PI * R;

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <svg
        viewBox="0 0 200 118"
        className={cn(size === "lg" ? "w-56" : "w-32")}
        role="img"
        aria-label={`${label}: ${measured ? pct(clamped) : "not measured"}`}
      >
        <path d={ARC} stroke="var(--muted)" strokeWidth={size === "lg" ? 13 : 11} fill="none" strokeLinecap="round" />
        {/* Band boundaries tick the arc at 0.35 and 0.60 — the same ceilings the server uses, shown
            so a reader can see how far the value is from a boundary rather than which side it fell on. */}
        {[0.35, 0.6].map((mark) => (
          <line
            key={mark}
            x1={CX - R + mark * 2 * R}
            x2={CX - R + mark * 2 * R}
            y1={CY - (size === "lg" ? 8 : 6)}
            y2={CY + (size === "lg" ? 8 : 6)}
            stroke="var(--border)"
            strokeWidth={1}
          />
        ))}
        <motion.path
          d={ARC}
          stroke={color}
          strokeWidth={size === "lg" ? 13 : 11}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={length}
          initial={{ strokeDashoffset: reduce ? length * (1 - clamped) : length }}
          animate={{ strokeDashoffset: length * (1 - clamped) }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
        <text
          x={CX}
          y={CY - 14}
          textAnchor="middle"
          className="tabular"
          fill="var(--foreground)"
          fontSize={size === "lg" ? 30 : 22}
          fontWeight={600}
        >
          {measured ? pct(clamped) : "—"}
        </text>
        <text x={CX} y={CY + 6} textAnchor="middle" fill="var(--muted-foreground)" fontSize={10}>
          {measured ? prob(clamped) : "not measured"}
        </text>
      </svg>
      <p className="mt-1 text-center text-xs font-medium" style={{ color }}>
        {label}
      </p>
      {caption && (
        <p className="mt-0.5 max-w-56 text-center text-[0.7rem] leading-snug text-muted-foreground">
          {caption}
        </p>
      )}
    </div>
  );
}

/**
 * Reliability, drawn as a bar rather than a gauge: it is a coverage figure ("how many of the checks
 * that could have spoken, did"), and a second radial dial would compete with the first for the
 * reader's attention while meaning something entirely different.
 */
export function ReliabilityMeter({
  score,
  note,
  className,
}: {
  score: number;
  note?: string;
  className?: string;
}) {
  const clamped = Math.min(1, Math.max(0, score));
  const tone = clamped >= 0.7 ? "var(--risk-low)" : clamped >= 0.4 ? "var(--risk-moderate)" : "var(--risk-high)";
  return (
    <div className={cn("flex w-full flex-col gap-1.5", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[0.7rem] font-medium uppercase tracking-wider text-muted-foreground">
          Evidence reliability
        </span>
        <span className="tabular text-xs" style={{ color: tone }}>
          {pct(clamped)}
        </span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ backgroundColor: tone }}
          initial={{ width: 0 }}
          animate={{ width: `${clamped * 100}%` }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        />
      </div>
      {note && <p className="text-[0.7rem] leading-snug text-muted-foreground">{note}</p>}
    </div>
  );
}
