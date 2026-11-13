import { useEffect, useRef } from "react";

/**
 * The page's h1. It takes focus when the page mounts, so after a route change keyboard
 * and screen-reader users land on the new page's title instead of the old link.
 */
export function PageHeading({ children, id }: { children: React.ReactNode; id?: string }) {
  const ref = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (document.activeElement === document.body || document.activeElement?.closest("nav")) {
      ref.current?.focus({ preventScroll: true });
    }
  }, []);
  return (
    <h1 ref={ref} id={id} tabIndex={-1}>
      {children}
    </h1>
  );
}
