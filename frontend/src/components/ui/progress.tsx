"use client";

import * as React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import { cn } from "@/lib/utils";

/**
 * An *indeterminate* bar for "a check is running".
 *
 * The job's own progress number is reported elsewhere as text; a bar that creeps toward a number
 * the server has no reason to be honest about would be theatre. This component is for the case
 * where a worker has accepted a group but has not yet emitted anything for it.
 */
export function Progress({
  className,
  value,
  indeterminate = false,
  ...props
}: React.ComponentProps<typeof ProgressPrimitive.Root> & { indeterminate?: boolean }) {
  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
      {...props}
    >
      <ProgressPrimitive.Indicator
        className={cn(
          "h-full w-full flex-1 rounded-full bg-primary transition-transform duration-300 ease-out",
          indeterminate && "animate-pulse-soft",
        )}
        style={{
          transform: `translateX(-${100 - (indeterminate ? 100 : (value ?? 0))}%)`,
        }}
      />
    </ProgressPrimitive.Root>
  );
}

/** A score bar: width is the score, colour is the server's band, and "not measured" is a hollow track. */
export function ScoreBar({
  score,
  bandClass,
  className,
}: {
  score: number | null;
  bandClass: string;
  className?: string;
}) {
  const measured = score != null && !Number.isNaN(score);
  return (
    <div
      className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
      role="presentation"
    >
      {measured ? (
        <div
          className={cn("h-full rounded-full transition-[width] duration-500 ease-out", bandClass)}
          style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }}
        />
      ) : (
        <div className="h-full w-full rounded-full bg-[repeating-linear-gradient(45deg,transparent,transparent_5px,var(--grid-line)_5px,var(--grid-line)_10px)]" />
      )}
    </div>
  );
}
