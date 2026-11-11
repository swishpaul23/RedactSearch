import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

// jsdom has no layout: give scrolling APIs a no-op so components can call them.
Element.prototype.scrollIntoView = vi.fn();
