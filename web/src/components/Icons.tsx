/* The one icon set for the app: Lucide (see .ux-profile.md). Pages import from here, so
   swapping a glyph happens in one place. Icons are decorative unless the caller labels them. */
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  ChartColumn,
  ChevronRight,
  ChevronsUpDown,
  CircleAlert,
  CircleCheck,
  FileText,
  List,
  LoaderCircle,
  MessageSquareText,
  Search,
  SlidersHorizontal,
  SunMoon,
  Upload,
  X,
  type LucideIcon,
} from "lucide-react";

type Props = { className?: string; size?: number };

function make(Glyph: LucideIcon, defaultSize = 16) {
  return function Icon({ className = "icon", size = defaultSize }: Props) {
    return <Glyph className={className} size={size} strokeWidth={2} aria-hidden="true" focusable="false" />;
  };
}

export const AskIcon = make(MessageSquareText, 18);
export const CallsIcon = make(List, 18);
export const QualityIcon = make(ChartColumn, 18);
export const UploadIcon = make(Upload, 18);
export const SearchIcon = make(Search);
export const FilterIcon = make(SlidersHorizontal);
export const ChevronIcon = make(ChevronRight);
export const CloseIcon = make(X);
export const ThemeIcon = make(SunMoon);
export const ExternalIcon = make(ArrowUpRight);
export const FileIcon = make(FileText);
export const DoneIcon = make(CircleCheck);
export const AlertIcon = make(CircleAlert);
export const SpinnerIcon = make(LoaderCircle);

export function SortIcon({ dir, className = "icon" }: Props & { dir: "ascending" | "descending" | "none" }) {
  const Glyph = dir === "ascending" ? ArrowUp : dir === "descending" ? ArrowDown : ChevronsUpDown;
  return (
    <Glyph
      className={dir === "none" ? `${className} is-muted` : className}
      size={14}
      strokeWidth={2}
      aria-hidden="true"
      focusable="false"
    />
  );
}
