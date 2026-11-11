/* Inline icons (stroke, 24px grid). Decorative unless given a title. */
type Props = { className?: string };
const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
  focusable: false,
};

export const AskIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z" />
    <path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.1" />
    <path d="M12 16.5h.01" />
  </svg>
);
export const CallsIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <path d="M4 6h16M4 12h16M4 18h10" />
  </svg>
);
export const QualityIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
  </svg>
);
export const SearchIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </svg>
);
export const FilterIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <path d="M4 5h16l-6 7.5V19l-4 1v-7.5L4 5Z" />
  </svg>
);
export const ChevronIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <path d="m9 6 6 6-6 6" />
  </svg>
);
export const CloseIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
);
export const SortIcon = ({ dir, className = "icon" }: Props & { dir: "ascending" | "descending" | "none" }) => (
  <svg {...base} className={className}>
    {dir !== "descending" && <path d="m8 10 4-4 4 4" opacity={dir === "none" ? 0.45 : 1} />}
    {dir !== "ascending" && <path d="m8 14 4 4 4-4" opacity={dir === "none" ? 0.45 : 1} />}
  </svg>
);
export const ThemeIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <circle cx="12" cy="12" r="8" />
    <path d="M12 4a8 8 0 0 0 0 16Z" fill="currentColor" stroke="none" />
  </svg>
);
export const ExternalIcon = ({ className = "icon" }: Props) => (
  <svg {...base} className={className}>
    <path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
  </svg>
);
