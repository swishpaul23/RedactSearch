import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { NavLink, Outlet } from "react-router";
import { api, call } from "../api/client";
import { usePersistentState } from "../lib/hooks";
import { AskIcon, CallsIcon, QualityIcon, ThemeIcon } from "./Icons";

type Theme = "system" | "light" | "dark";
const NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
const THEME_LABEL: Record<Theme, string> = { system: "Match system", light: "Light", dark: "Dark" };

const NAV = [
  { to: "/", label: "Ask", icon: AskIcon, end: true },
  { to: "/calls", label: "Calls", icon: CallsIcon, end: false },
  { to: "/quality", label: "Quality", icon: QualityIcon, end: false },
];

function SystemStatus() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => call(() => api.GET("/api/health")),
    refetchInterval: 60_000,
  });
  if (health.isPending) return <p className="status-line">Checking status…</p>;
  if (health.isError) {
    return (
      <p className="status-line">
        <span className="status-dot is-warn" aria-hidden="true" />
        API unreachable
      </p>
    );
  }
  const h = health.data;
  return (
    <div>
      <p className="status-line">
        <span className={h.index.available ? "status-dot is-ok" : "status-dot is-warn"} aria-hidden="true" />
        {h.index.available ? `Index ready, ${h.index.n_chunks?.toLocaleString()} turns` : "Index not built"}
      </p>
      <p className="status-line">
        <span className={h.llm ? "status-dot is-ok" : "status-dot is-warn"} aria-hidden="true" />
        {h.llm ? `Answers by ${h.llm.model}` : "No answer model set"}
      </p>
    </div>
  );
}

export function AppShell() {
  const [theme, setTheme] = usePersistentState<Theme>("redactsearch.theme", "system");
  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <div className="shell">
        <header className="sidebar">
          <NavLink to="/" className="brand" aria-label="RedactSearch home">
            <span className="brand-mark" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
            RedactSearch
          </NavLink>
          <nav aria-label="Main">
            <ul className="nav">
              {NAV.map(({ to, label, icon: Icon, end }) => (
                <li key={to}>
                  <NavLink to={to} end={end} className="nav-link">
                    <Icon className="nav-icon" />
                    {label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="sidebar-foot">
            <SystemStatus />
            <button
              type="button"
              className="btn btn-sm btn-quiet"
              onClick={() => setTheme(NEXT[theme])}
              aria-label={`Theme: ${THEME_LABEL[theme]}. Switch to ${THEME_LABEL[NEXT[theme]]}`}
            >
              <ThemeIcon />
              Theme: {THEME_LABEL[theme]}
            </button>
          </div>
        </header>
        <main id="main" className="main" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </>
  );
}
