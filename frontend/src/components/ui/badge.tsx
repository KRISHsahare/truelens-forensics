import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import type { RiskBand } from "@/lib/types";
import { BAND_CLASSES } from "@/lib/format";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[0.7rem] font-medium leading-tight transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary/15 text-primary",
        secondary: "border-border bg-secondary text-secondary-foreground",
        outline: "border-border text-foreground",
        muted: "border-transparent bg-muted text-muted-foreground",
        destructive: "border-transparent bg-destructive/15 text-destructive",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

/**
 * A band-tinted label.
 *
 * Colour is never the only carrier — the words are always present — because the risk bands are
 * distinguishable by copy alone and a red/green-only reading fails WCAG 1.4.1 for a colour-blind
 * reviewer.
 */
export function BandBadge({
  band,
  children,
  className,
}: {
  band: RiskBand;
  children: React.ReactNode;
  className?: string;
}) {
  const c = BAND_CLASSES[band];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-border/60 px-2.5 py-1 text-xs font-medium",
        c.bg,
        c.text,
        className,
      )}
    >
      <span aria-hidden className={cn("size-1.5 rounded-full", c.bar)} />
      {children}
    </span>
  );
}
