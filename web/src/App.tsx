import { Route, Routes } from "react-router";
import { AppShell } from "./components/AppShell";
import { AskPage } from "./pages/AskPage";
import { CallPage } from "./pages/CallPage";
import { CallsPage } from "./pages/CallsPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { QualityPage } from "./pages/QualityPage";
import { UploadPage } from "./pages/UploadPage";

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<AskPage />} />
        <Route path="calls" element={<CallsPage />} />
        <Route path="calls/:callId" element={<CallPage />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="quality" element={<QualityPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
