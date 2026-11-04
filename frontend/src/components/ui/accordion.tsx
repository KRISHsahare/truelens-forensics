"use client";

import * as React from "react";
import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export const Accordion = AccordionPrimitive.Root;

export function AccordionItem({
  className,
  ...props
}: React.ComponentProps<typeof AccordionPrimitive.Item>) {
  return (
    <AccordionPrimitive.Item className={cn("border-b border-border/70", className)} {...props} />
  );
}

/**
 * The chevron rotates instead of the panel sliding open from zero height, because the raw-details
 * block is a JSON tree of unknown length: a `max-height` transition on it either clips the content
 * or finishes before the layout does.
 */
export function AccordionTrigger({
  className,
  children,
  ...props
}: React.ComponentProps<typeof AccordionPrimitive.Trigger>) {
  return (
    <AccordionPrimitive.Header className="flex">
      <AccordionPrimitive.Trigger
        className={cn(
          "flex flex-1 items-center justify-between gap-3 py-2.5 text-left text-xs font-medium transition-colors",
          "hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          "[&[data-state=open]>svg]:rotate-180",
          className,
        )}
        {...props}
      >
        <span className="flex items-center gap-2">{children}</span>
        <ChevronDown className="size-3.5 shrink-0 text-muted-foreground transition-transform duration-200" />
      </AccordionPrimitive.Trigger>
    </AccordionPrimitive.Header>
  );
}

export function AccordionContent({
  className,
  children,
  ...props
}: React.ComponentProps<typeof AccordionPrimitive.Content>) {
  return (
    <AccordionPrimitive.Content
      // The animation interpolates `grid-template-rows`, so the animated element must itself be the
      // grid: a wrapper div carrying `grid` one level down would be sized by its content and refuse
      // to collapse to zero. The inner div needs `min-height: 0` for the same reason.
      className={cn(
        "grid overflow-hidden text-xs",
        "data-[state=closed]:grid-rows-[0fr] data-[state=open]:animate-accordion-down",
        "data-[state=closed]:transition-[grid-template-rows] data-[state=closed]:duration-150",
        className,
      )}
      {...props}
    >
      <div className="min-h-0 pb-3 pt-0">{children}</div>
    </AccordionPrimitive.Content>
  );
}
